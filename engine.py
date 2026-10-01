# splice/engine.py
"""
여러 backbone(biotransmamba, caduceus, dnabert2, hyenadna, ...)이 공유하는
학습/평가 엔진.

각 backbone 스크립트가 준비해야 할 것은 딱 하나, forward_fn 입니다.

  - splice3 태스크: forward_fn(seq_bytes, device) -> logits            (B,L,3)
  - tissue  태스크: forward_fn(seq_bytes, device) -> (logits, logits_tis)

나머지(배치 루프, loss 계산, optimizer step, 메트릭 누적/집계, RC augmentation
/ensemble)는 이 파일이 전담합니다.

새 backbone을 추가할 때 할 일:
  1) forward_fn만 그 모델에 맞게 작성
  2) run_splice3_epoch / run_tissue_epoch 호출
그 외 아무것도 복붙할 필요가 없습니다.
"""
from typing import Callable, List, Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from metrics import (
    masked_ce_loss, masked_bce_with_logits_loss,
    init_topkl_sums, update_topkl_sums, finalize_topkl,
    init_pr_auc_buf, accumulate_pr_auc, finalize_pr_auc,
    init_pr_auc_tissue_buf, accumulate_pr_auc_tissue, finalize_pr_auc_tissue,
)

ForwardFn = Callable[[List[bytes], torch.device], Any]


# =========================================================
# Reverse-complement 유틸 (splice3 태스크에서만 의미가 있음:
# acceptor(1)/donor(2) 라벨이 RC에서 서로 뒤바뀜)
# =========================================================
_COMP = {ord('A'): ord('T'), ord('T'): ord('A'), ord('C'): ord('G'), ord('G'): ord('C'), ord('N'): ord('N')}

def reverse_complement_bytes(b: bytes) -> bytes:
    return bytes(_COMP.get(x, ord('N')) for x in b[::-1])

def rc_transform_labels(y: torch.Tensor) -> torch.Tensor:
    """y: (B,SL) values in {ignore_index,0,1,2}. 위치 flip + acceptor/donor swap."""
    y = torch.flip(y, dims=[-1])
    y2 = y.clone()
    y2[y == 1] = 2
    y2[y == 2] = 1
    return y2

def rc_align_logits_to_forward(logits_rc: torch.Tensor) -> torch.Tensor:
    """RC 시퀀스로부터 나온 logits (B,L,3)를 forward 기준으로 정렬 (위치 flip + class swap)."""
    x = torch.flip(logits_rc, dims=[1])
    x2 = x.clone()
    x2[..., 1] = x[..., 2]
    x2[..., 2] = x[..., 1]
    return x2


# =========================================================
# splice3 (3-class: non-splice / acceptor / donor)
# =========================================================
class Splice3MetricsTracker:
    def __init__(self, crop: int, sl: int, ignore_index: int, topkl_ks, topkl_tol: int, topkl_positive_only: bool):
        self.crop = crop
        self.sl = sl
        self.ignore_index = ignore_index
        self.topkl_ks = topkl_ks
        self.topkl_tol = topkl_tol
        self.topkl_positive_only = topkl_positive_only

        self.pr_buf = init_pr_auc_buf()
        self.topkl_acc = init_topkl_sums(topkl_ks)
        self.topkl_don = init_topkl_sums(topkl_ks)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, y: torch.Tensor):
        accumulate_pr_auc(self.pr_buf, logits, y, crop=self.crop, sl=self.sl, ignore_index=self.ignore_index)
        update_topkl_sums(
            self.topkl_acc, logits, y, class_id=1,
            crop=self.crop, sl=self.sl, ks=self.topkl_ks, tol=self.topkl_tol,
            ignore_index=self.ignore_index, positive_only=self.topkl_positive_only,
        )
        update_topkl_sums(
            self.topkl_don, logits, y, class_id=2,
            crop=self.crop, sl=self.sl, ks=self.topkl_ks, tol=self.topkl_tol,
            ignore_index=self.ignore_index, positive_only=self.topkl_positive_only,
        )

    def finalize(self) -> Dict[str, Any]:
        ap_acc, ap_don = finalize_pr_auc(self.pr_buf)
        rec_kl_acc, _ = finalize_topkl(self.topkl_acc, self.topkl_ks)
        rec_kl_don, _ = finalize_topkl(self.topkl_don, self.topkl_ks)
        return {
            "pr_auc_acc": ap_acc,
            "pr_auc_don": ap_don,
            "rec_kl_acc": rec_kl_acc,
            "rec_kl_don": rec_kl_don,
        }

    @staticmethod
    def log_line(prefix: str, out: Dict[str, Any]) -> str:
        return (
            f"{prefix}loss={out['loss']:.4f} | "
            f"PR-AUC(acc)={out['pr_auc_acc']:.4f} PR-AUC(don)={out['pr_auc_don']:.4f} | "
            f"Recall@1L(acc)={out['rec_kl_acc'][1]:.4f} Recall@1L(don)={out['rec_kl_don'][1]:.4f}"
        )


def run_splice3_epoch(
    model: torch.nn.Module,
    dl: DataLoader,
    forward_fn: ForwardFn,
    cfg,
    optimizer: Optional[torch.optim.Optimizer] = None,
    rc_aug: bool = False,
    rc_aug_prob: float = 0.5,
    rc_ensemble: bool = False,
    max_batches: int = 0,
    grad_clip_norm: float = 1.0,
) -> Dict[str, Any]:
    """
    optimizer가 주어지면 train 모드(backward+step), 없으면 eval 모드(메트릭 집계).
    rc_aug: train 시 50%(rc_aug_prob) 확률로 RC 시퀀스로 바꿔서 augmentation.
    rc_ensemble: eval 시 forward/RC 양쪽 logits를 평균 (caduceus 스타일).
    grad_clip_norm: train 시 gradient norm을 이 값으로 clip. Mamba2 계열은 학습 초반
        gradient가 튀는 경우가 흔해서 기본으로 켜둠. 0 이하로 주면 비활성화.
    """
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    tracker = None if train_mode else Splice3MetricsTracker(
        crop=cfg.CROP, sl=cfg.SL, ignore_index=cfg.IGNORE_INDEX,
        topkl_ks=cfg.TOPKL_KS, topkl_tol=cfg.TOPKL_TOL, topkl_positive_only=cfg.TOPKL_POSITIVE_ONLY,
    )

    running_loss = 0.0
    nb = 0
    device = cfg.DEVICE

    grad_ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with grad_ctx:
        for bi, (seq_bytes, y) in enumerate(dl, start=1):
            y = y.to(device)

            if train_mode and rc_aug and np.random.rand() < rc_aug_prob:
                seq_bytes = [reverse_complement_bytes(b) for b in seq_bytes]
                y = rc_transform_labels(y)

            logits = forward_fn(seq_bytes, device)

            if (not train_mode) and rc_ensemble:
                seq_rc = [reverse_complement_bytes(b) for b in seq_bytes]
                logits_rc = forward_fn(seq_rc, device)
                logits = 0.5 * (logits + rc_align_logits_to_forward(logits_rc))

            loss = masked_ce_loss(logits, y, crop=cfg.CROP, sl=cfg.SL, ignore_index=cfg.IGNORE_INDEX)
            if loss is None:
                continue
            if not torch.isfinite(loss):
                print(f"[engine] non-finite loss at batch {bi} (train={train_mode}), skipping this batch")
                if train_mode:
                    optimizer.zero_grad(set_to_none=True)
                continue

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()
            else:
                tracker.update(logits, y)

            running_loss += float(loss.item())
            nb += 1

            if max_batches and bi >= max_batches:
                break

    out: Dict[str, Any] = {"loss": running_loss / max(nb, 1), "batches": nb}
    if not train_mode:
        out.update(tracker.finalize())
    return out


# =========================================================
# tissue (GTEx tissue-specific, multi-label BCE)
# =========================================================
class TissueMetricsTracker:
    def __init__(self, crop: int, sl: int, num_tissues: int, thr: float = 0.5):
        self.crop = crop
        self.sl = sl
        self.thr = thr
        self.pr_tis = init_pr_auc_tissue_buf(num_tissues)

    @torch.no_grad()
    def update(self, logits_tis: torch.Tensor, y_tis: torch.Tensor):
        # (B, SL, T) -> center crop -> (B, T, SL)
        logits_center = logits_tis[:, self.crop:self.crop + self.sl, :].permute(0, 2, 1).contiguous()
        accumulate_pr_auc_tissue(self.pr_tis, logits_center, y_tis, crop=0, sl=self.sl, thr=self.thr)

    def finalize(self) -> Dict[str, Any]:
        ap_tissues = finalize_pr_auc_tissue(self.pr_tis)
        return {
            "pr_auc_tissue": ap_tissues,
            "pr_auc_tissue_mean": float(np.nanmean(ap_tissues)),
        }

    @staticmethod
    def log_line(prefix: str, out: Dict[str, Any]) -> str:
        ap_str = ", ".join("%.4f" % x for x in out["pr_auc_tissue"])
        return f"{prefix}loss={out['loss']:.4f} | Tissue AP=[{ap_str}] | Mean Tissue AP={out['pr_auc_tissue_mean']:.4f}"


def run_tissue_epoch(
    model: torch.nn.Module,
    dl: DataLoader,
    forward_fn: ForwardFn,
    cfg,
    num_tissues: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    max_batches: int = 0,
    grad_clip_norm: float = 1.0,
) -> Dict[str, Any]:
    """optimizer가 주어지면 train 모드, 없으면 eval 모드(tissue PR-AUC 집계)."""
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    tracker = None if train_mode else TissueMetricsTracker(crop=cfg.CROP, sl=cfg.SL, num_tissues=num_tissues)

    running_loss = 0.0
    nb = 0
    device = cfg.DEVICE

    grad_ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with grad_ctx:
        for bi, (seq_bytes, y, y_tis) in enumerate(dl, start=1):
            y_tis = y_tis.to(device)

            logits, logits_tis = forward_fn(seq_bytes, device)

            loss = masked_bce_with_logits_loss(logits_tis, y_tis, crop=cfg.CROP, sl=cfg.SL)
            if loss is None:
                continue
            if not torch.isfinite(loss):
                print(f"[engine] non-finite loss at batch {bi} (train={train_mode}), skipping this batch")
                if train_mode:
                    optimizer.zero_grad(set_to_none=True)
                continue

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()
            else:
                tracker.update(logits_tis, y_tis)

            running_loss += float(loss.item())
            nb += 1

            if max_batches and bi >= max_batches:
                break

    out: Dict[str, Any] = {"loss": running_loss / max(nb, 1), "batches": nb}
    if not train_mode:
        out.update(tracker.finalize())
    return out