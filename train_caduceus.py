import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CFG
from data import NpzShardIterable, collate_fn
import metrics
from models.caduceus import CaduceusSplice


# -------------------------
# DNA RC helpers
# -------------------------
_COMP = {ord('A'): ord('T'), ord('T'): ord('A'), ord('C'): ord('G'), ord('G'): ord('C'), ord('N'): ord('N')}
def reverse_complement_bytes(b: bytes) -> bytes:
    # reverse + complement
    return bytes(_COMP.get(x, ord('N')) for x in b[::-1])

def rc_transform_labels(y: torch.Tensor) -> torch.Tensor:
    """
    y: (B,500) values in {-1,0,1,2}
    RC하면:
      - 위치는 reverse
      - acceptor(1) <-> donor(2) swap
    """
    y = torch.flip(y, dims=[-1])
    y2 = y.clone()
    y2[y == 1] = 2
    y2[y == 2] = 1
    return y2

def rc_align_logits_to_forward(logits_rc_1500: torch.Tensor) -> torch.Tensor:
    """
    logits_rc_1500: RC sequence로부터 나온 (B,1500,3)
    forward 기준으로 평균내려면:
      - 위치 flip
      - class 1/2 swap
    """
    x = torch.flip(logits_rc_1500, dims=[1])
    x2 = x.clone()
    x2[..., 1] = x[..., 2]
    x2[..., 2] = x[..., 1]
    return x2


# -------------------------
# Tokenize helper
# -------------------------
def tokenize_batch(tokenizer, seq_bytes_list, device, max_len):
    # bytes -> str
    seq_str = [s.decode("ascii") for s in seq_bytes_list]
    enc = tokenizer(
        seq_str,
        add_special_tokens=False,     # 길이 1500 그대로 맞추기
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    return enc["input_ids"].to(device), enc.get("attention_mask", None).to(device) if "attention_mask" in enc else None


@torch.no_grad()
def evaluate(model, tokenizer, split: str, cfg: CFG, rc_ensemble: bool = True):
    model.eval()

    dl = DataLoader(
        NpzShardIterable(cfg.DATA_DIR, split, cfg.SEED, shuffle_files=False, shuffle_within=False),
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn
    )

    total_loss = 0.0
    n_batches = 0

    pr_buf = {"acc_y": [], "acc_s": [], "don_y": [], "don_s": []}
    topk_acc = metrics.init_topk_sums(cfg.TOPK_KS)
    topk_don = metrics.init_topk_sums(cfg.TOPK_KS)
    topkl_acc = metrics.init_topkl_sums(cfg.TOPKL_KS)
    topkl_don = metrics.init_topkl_sums(cfg.TOPKL_KS)

    for seq_bytes, y in dl:
        y = y.to(cfg.DEVICE)

        input_ids, attn = tokenize_batch(tokenizer, seq_bytes, cfg.DEVICE, cfg.INPUT_LEN)

        logits = model(input_ids=input_ids, attention_mask=attn)  # (B,1500,3)

        if rc_ensemble:
            seq_rc = [reverse_complement_bytes(b) for b in seq_bytes]
            ids_rc, attn_rc = tokenize_batch(tokenizer, seq_rc, cfg.DEVICE, cfg.INPUT_LEN)
            logits_rc = model(input_ids=ids_rc, attention_mask=attn_rc)
            logits = 0.5 * (logits + rc_align_logits_to_forward(logits_rc))

        loss = metrics.masked_ce_loss(logits, y, crop=cfg.CROP, sl=cfg.SL, ignore_index=cfg.IGNORE_INDEX)
        if loss is not None:
            total_loss += float(loss)
            n_batches += 1

        metrics.accumulate_pr_auc(pr_buf, logits, y, crop=cfg.CROP, sl=cfg.SL, ignore_index=cfg.IGNORE_INDEX)

        metrics.update_topk_sums(
            topk_acc, logits, y, class_id=1,
            crop=cfg.CROP, sl=cfg.SL, ks=cfg.TOPK_KS, tol=cfg.TOPK_TOL,
            ignore_index=cfg.IGNORE_INDEX, positive_only=cfg.TOPK_POSITIVE_ONLY
        )
        metrics.update_topk_sums(
            topk_don, logits, y, class_id=2,
            crop=cfg.CROP, sl=cfg.SL, ks=cfg.TOPK_KS, tol=cfg.TOPK_TOL,
            ignore_index=cfg.IGNORE_INDEX, positive_only=cfg.TOPK_POSITIVE_ONLY
        )

        metrics.update_topkl_sums(
            topkl_acc, logits, y, class_id=1,
            crop=cfg.CROP, sl=cfg.SL, ks=cfg.TOPKL_KS, tol=cfg.TOPKL_TOL,
            ignore_index=cfg.IGNORE_INDEX, positive_only=cfg.TOPKL_POSITIVE_ONLY
        )
        metrics.update_topkl_sums(
            topkl_don, logits, y, class_id=2,
            crop=cfg.CROP, sl=cfg.SL, ks=cfg.TOPKL_KS, tol=cfg.TOPKL_TOL,
            ignore_index=cfg.IGNORE_INDEX, positive_only=cfg.TOPKL_POSITIVE_ONLY
        )

    ap_acc, ap_don = metrics.finalize_pr_auc(pr_buf)
    hit_acc, rec_acc, _ = metrics.finalize_topk(topk_acc, cfg.TOPK_KS)
    hit_don, rec_don, _ = metrics.finalize_topk(topk_don, cfg.TOPK_KS)
    rkl_acc, _ = metrics.finalize_topkl(topkl_acc, cfg.TOPKL_KS)
    rkl_don, _ = metrics.finalize_topkl(topkl_don, cfg.TOPKL_KS)

    return {
        "loss": total_loss / max(1, n_batches),
        "pr_auc_acc": ap_acc,
        "pr_auc_don": ap_don,
        "hit_acc": hit_acc,
        "hit_don": hit_don,
        "rec_acc": rec_acc,
        "rec_don": rec_don,
        "rec_kl_acc": rkl_acc,
        "rec_kl_don": rkl_don,
    }


def main():
    cfg = CFG()
    cfg.EPOCHS = 3  # 필요하면 config에서 조절
    cfg.DEVICE = cfg.DEVICE if torch.cuda.is_available() else "cpu"

    # ✅ 모델명만 바꿔서 실험 반복 가능
    CADUCEUS_NAME = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"

    model = CaduceusSplice(CADUCEUS_NAME, num_classes=cfg.NUM_CLASSES, dropout=0.1).to(cfg.DEVICE)
    tokenizer = model.tokenizer

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    train_dl = DataLoader(
        NpzShardIterable(cfg.DATA_DIR, "train", cfg.SEED, shuffle_files=True, shuffle_within=True),
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn
    )

    for ep in range(1, cfg.EPOCHS + 1):
        model.train()
        running = 0.0
        nb = 0

        for seq_bytes, y in train_dl:
            # (선택) RC augmentation 50%
            if np.random.rand() < 0.5:
                seq_bytes = [reverse_complement_bytes(b) for b in seq_bytes]
                y = rc_transform_labels(y)

            y = y.to(cfg.DEVICE)
            input_ids, attn = tokenize_batch(tokenizer, seq_bytes, cfg.DEVICE, cfg.INPUT_LEN)

            logits = model(input_ids=input_ids, attention_mask=attn)
            loss = metrics.masked_ce_loss(logits, y, crop=cfg.CROP, sl=cfg.SL, ignore_index=cfg.IGNORE_INDEX)
            if loss is None:
                continue

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += float(loss)
            nb += 1

        val = evaluate(model, tokenizer, "val", cfg, rc_ensemble=True)
        print(
            f"[Epoch {ep}] train_loss={running/max(1,nb):.4f} | "
            f"val_loss={val['loss']:.4f} | "
            f"PR-AUC(acc)={val['pr_auc_acc']:.4f} PR-AUC(don)={val['pr_auc_don']:.4f} | "
            f"Hit@10(acc)={val['hit_acc'][10]:.4f} Hit@10(don)={val['hit_don'][10]:.4f} | "
            f"Recall@1L(acc)={val['rec_kl_acc'][1]:.4f} Recall@1L(don)={val['rec_kl_don'][1]:.4f}"
        )

    test = evaluate(model, tokenizer, "test", cfg, rc_ensemble=True)
    print("\n[TEST]")
    print(
        f"loss={test['loss']:.4f} | "
        f"PR-AUC(acc)={test['pr_auc_acc']:.4f} PR-AUC(don)={test['pr_auc_don']:.4f} | "
        f"Hit@1(acc)={test['hit_acc'][1]:.4f} Hit@1(don)={test['hit_don'][1]:.4f} | "
        f"Hit@10(acc)={test['hit_acc'][10]:.4f} Hit@10(don)={test['hit_don'][10]:.4f} | "
        f"Recall@1L(acc)={test['rec_kl_acc'][1]:.4f} Recall@1L(don)={test['rec_kl_don'][1]:.4f}"
    )


if __name__ == "__main__":
    main()
