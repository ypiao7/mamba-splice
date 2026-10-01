# metrics.py  (top-k 제거 버전: top-kL + PR-AUC(AP)만 사용)
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

import torch
import torch.nn.functional as F

def masked_bce_with_logits_loss(logits, targets, crop, sl):
    """
    logits : (B, L, C)  e.g. (B,3000,15)
    targets: (B, C, SL) e.g. (B,15,1000)
    crop/sl : center slice on logits (e.g. crop=1000, sl=1000)
    """
    # center slice on logits: (B,SL,C) -> (B,C,SL)
    x = logits[:, crop:crop + sl, :].permute(0, 2, 1).contiguous()
    return F.binary_cross_entropy_with_logits(x, targets)

def masked_ce_loss(logits, y, crop: int, sl: int, ignore_index: int):
    logits = logits[:, crop:crop+sl, :]  # (B, SL, 3)
    logits = logits.permute(0, 2, 1)          # (B, 3, SL)

    if (y != ignore_index).sum().item() == 0:
        return None

    return F.cross_entropy(logits, y, ignore_index=ignore_index)


# -------------------------
# Helpers
# -------------------------
def _expand_positions(pos_set, tol: int, L: int):
    if tol <= 0:
        return pos_set
    out = set()
    for p in pos_set:
        for d in range(-tol, tol + 1):
            q = p + d
            if 0 <= q < L:
                out.add(q)
    return out


# -------------------------
# top-kL (SpliceAI-style)
# -------------------------
def init_topkl_sums(ks):
    return {"rec_sum": {k: 0.0 for k in ks}, "n_eval": 0}


@torch.no_grad()
def update_topkl_sums(
    sums,
    logits_1500,
    y_500,
    class_id: int,
    crop: int,
    sl: int,
    ks,
    tol: int,
    ignore_index: int,
    positive_only: bool = True,
):
    """
    For each sample:
      L_true = # of true splice sites (positions where y == class_id)
      pick top ceil(k * L_true) predictions, measure recall with tolerance

    returns accumulated recall over evaluated samples.
    """
    logits = logits_1500[:, crop:crop + sl, :]            # (B, 500, 3)
    prob = torch.softmax(logits, dim=-1)[..., class_id]   # (B, 500)

    B = y_500.size(0)
    for b in range(B):
        yb = y_500[b]
        valid = (yb != ignore_index)
        avail = int(valid.sum().item())
        if avail == 0:
            continue

        true_pos = set(torch.where(yb == class_id)[0].tolist())
        L_true = len(true_pos)
        if positive_only and L_true == 0:
            continue

        sums["n_eval"] += 1

        scores = prob[b].clone()
        scores[~valid] = -float("inf")

        for k in ks:
            if L_true == 0:
                rec = 0.0
            else:
                kk = int(np.ceil(float(k) * L_true))
                kk = max(1, min(kk, avail))  # valid 범위 내로 제한

                top_idx = torch.topk(scores, kk, largest=True).indices.tolist()
                top_set = set(top_idx)

                matched = 0
                for p in true_pos:
                    if _expand_positions({p}, tol=tol, L=sl) & top_set:
                        matched += 1
                rec = matched / L_true

            sums["rec_sum"][k] += rec


def finalize_topkl(sums, ks):
    n = sums["n_eval"]
    if n == 0:
        return {k: float("nan") for k in ks}, 0
    return {k: sums["rec_sum"][k] / n for k in ks}, n


# -------------------------
# PR-AUC (Average Precision)
# -------------------------
def init_pr_auc_buf():
    return {"acc_y": [], "acc_s": [], "don_y": [], "don_s": []}


@torch.no_grad()
def accumulate_pr_auc(acc_buf, logits_1500, y_500, crop: int, sl: int, ignore_index: int):
    """
    Position-wise AP for acceptor(class=1) and donor(class=2) over all valid positions.
    """
    logits = logits_1500[:, crop:crop + sl, :]
    prob = torch.softmax(logits, dim=-1).detach().cpu().numpy()

    y = y_500.detach().cpu().numpy()
    valid = (y != ignore_index)
    if valid.sum() == 0:
        return

    acc_y = (y == 1).astype(np.int32)
    don_y = (y == 2).astype(np.int32)

    acc_buf["acc_y"].append(acc_y[valid])
    acc_buf["acc_s"].append(prob[..., 1][valid])
    acc_buf["don_y"].append(don_y[valid])
    acc_buf["don_s"].append(prob[..., 2][valid])

@torch.no_grad()
def accumulate_pr_auc_hyena(acc_buf, logits_1500, y_500, crop: int, sl: int, ignore_index: int):
    logits = logits_1500[:, crop:crop + sl, :]  # (B, SL, 3)

    # ✅ logits/prob 유한성 체크
    prob_t = torch.softmax(logits, dim=-1)  # (B, SL, 3)
    y = y_500

    valid = (y != ignore_index)

    # ✅ score가 finite인 위치만 사용
    acc_score = prob_t[..., 1]
    don_score = prob_t[..., 2]

    finite_acc = torch.isfinite(acc_score)
    finite_don = torch.isfinite(don_score)

    m_acc = valid & finite_acc
    m_don = valid & finite_don

    if m_acc.sum().item() > 0:
        acc_y = (y == 1).to(torch.int32)
        acc_buf["acc_y"].append(acc_y[m_acc].detach().cpu().numpy())
        acc_buf["acc_s"].append(acc_score[m_acc].detach().cpu().numpy())

    if m_don.sum().item() > 0:
        don_y = (y == 2).to(torch.int32)
        acc_buf["don_y"].append(don_y[m_don].detach().cpu().numpy())
        acc_buf["don_s"].append(don_score[m_don].detach().cpu().numpy())


def finalize_pr_auc(acc_buf):
    acc_y = np.concatenate(acc_buf["acc_y"]) if acc_buf["acc_y"] else np.array([], dtype=np.int32)
    acc_s = np.concatenate(acc_buf["acc_s"]) if acc_buf["acc_s"] else np.array([], dtype=np.float32)
    don_y = np.concatenate(acc_buf["don_y"]) if acc_buf["don_y"] else np.array([], dtype=np.int32)
    don_s = np.concatenate(acc_buf["don_s"]) if acc_buf["don_s"] else np.array([], dtype=np.float32)

    ap_acc = average_precision_score(acc_y, acc_s) if acc_y.sum() > 0 else float("nan")
    ap_don = average_precision_score(don_y, don_s) if don_y.sum() > 0 else float("nan")
    return ap_acc, ap_don

# metrics.py
# metrics.py (tissue PR-AUC 부분만 교체)
import numpy as np
import torch
from sklearn.metrics import average_precision_score

def init_pr_auc_tissue_buf(n_tissues: int):
    # 각 tissue마다 y_list, s_list를 모으는 버퍼
    return {
        "y": [[] for _ in range(n_tissues)],
        "s": [[] for _ in range(n_tissues)],
        "n_tissues": int(n_tissues),
    }

@torch.no_grad()
def accumulate_pr_auc_tissue(buf, logits_tis, y_tis, crop: int, sl: int, thr: float = 0.5):
    """
    position-wise tissue AP
    logits_tis: (B,15,1000)  (권장: 이미 center slice + permute된 상태)
    y_tis:      (B,15,1000)  float target (0~1)
    crop/sl:    여기서는 보통 crop=0, sl=1000
    thr:        y_tis를 binary로 만들 임계값
    """
    # shape 통일: (B,15,L)
    if logits_tis.dim() != 3 or y_tis.dim() != 3:
        raise ValueError(f"logits_tis/y_tis must be 3D, got {logits_tis.shape}, {y_tis.shape}")

    # crop/sl 적용 (보통 crop=0)
    logits = logits_tis[:, :, crop:crop+sl]
    y = y_tis[:, :, crop:crop+sl]

    # score: sigmoid(logits)
    s = torch.sigmoid(logits)

    # label: (y > thr) 로 binary화
    yb = (y > thr).to(torch.int32)

    # tissue별로 flatten해서 리스트에 축적
    # 각 tissue마다 길이 = B*sl
    for t in range(buf["n_tissues"]):
        buf["y"][t].append(yb[:, t, :].reshape(-1).detach().cpu().numpy())
        buf["s"][t].append(s[:, t, :].reshape(-1).detach().cpu().numpy())

def finalize_pr_auc_tissue(buf):
    aps = []
    for t in range(buf["n_tissues"]):
        if len(buf["y"][t]) == 0:
            aps.append(float("nan"))
            continue
        y = np.concatenate(buf["y"][t], axis=0)
        s = np.concatenate(buf["s"][t], axis=0)
        # positive가 하나도 없으면 AP 정의가 애매하니 NaN
        ap = average_precision_score(y, s) if y.sum() > 0 else float("nan")
        aps.append(float(ap))
    return aps
