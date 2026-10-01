# train.py (요지 수정본)
import torch
from torch.utils.data import DataLoader
from config import CFG
from data import NpzShardIterable, collate_fn
from metrics import (
    masked_ce_loss,
    init_topkl_sums, update_topkl_sums, finalize_topkl,
    init_pr_auc_buf, accumulate_pr_auc, finalize_pr_auc
)
from models.dnabert2 import DNABERT2ForSplice


def evaluate(model, split: str):
    model.eval()
    dl = DataLoader(
        NpzShardIterable(CFG.DATA_DIR, split=split, seed=CFG.SEED, shuffle_files=False, shuffle_within=False),
        batch_size=CFG.BATCH_SIZE,
        num_workers=CFG.NUM_WORKERS,
        collate_fn=collate_fn
    )

    total_loss, n_batches = 0.0, 0
    pr_buf = init_pr_auc_buf()

    topkl_acc = init_topkl_sums(CFG.TOPKL_KS)
    topkl_don = init_topkl_sums(CFG.TOPKL_KS)

    for seq_bytes, y in dl:
        dna = [s.decode("ascii") for s in seq_bytes]
        y = y.to(CFG.DEVICE)

        with torch.no_grad():                    # ✅ 중요!
            logits = model(dna)                  # (B,1500,3)

        loss = masked_ce_loss(logits, y, CFG.CROP, CFG.SL, CFG.IGNORE_INDEX)
        if loss is not None:
            total_loss += float(loss.item()); n_batches += 1

        accumulate_pr_auc(pr_buf, logits, y, CFG.CROP, CFG.SL, CFG.IGNORE_INDEX)

        update_topkl_sums(topkl_acc, logits, y, 1, CFG.CROP, CFG.SL, CFG.TOPKL_KS, CFG.TOPKL_TOL, CFG.IGNORE_INDEX, CFG.TOPKL_POSITIVE_ONLY)
        update_topkl_sums(topkl_don, logits, y, 2, CFG.CROP, CFG.SL, CFG.TOPKL_KS, CFG.TOPKL_TOL, CFG.IGNORE_INDEX, CFG.TOPKL_POSITIVE_ONLY)

    val_loss = total_loss / max(n_batches, 1)
    ap_acc, ap_don = finalize_pr_auc(pr_buf)
    recKL_acc, _ = finalize_topkl(topkl_acc, CFG.TOPKL_KS)
    recKL_don, _ = finalize_topkl(topkl_don, CFG.TOPKL_KS)

    return val_loss, ap_acc, ap_don, recKL_acc, recKL_don


def main():
    torch.manual_seed(CFG.SEED)
    model = DNABERT2ForSplice(num_classes=CFG.NUM_CLASSES, dropout=0.1).to(CFG.DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)

    train_dl = DataLoader(
        NpzShardIterable(CFG.DATA_DIR, split="train", seed=CFG.SEED, shuffle_files=True, shuffle_within=True),
        batch_size=CFG.BATCH_SIZE,
        num_workers=CFG.NUM_WORKERS,
        collate_fn=collate_fn
    )

    model.train()
    running, nb = 0.0, 0
    for seq_bytes, y in train_dl:
        dna = [s.decode("ascii") for s in seq_bytes]
        y = y.to(CFG.DEVICE)

        logits = model(dna)
        loss = masked_ce_loss(logits, y, CFG.CROP, CFG.SL, CFG.IGNORE_INDEX)
        if loss is None:
            continue

        opt.zero_grad()
        loss.backward()
        opt.step()

        running += float(loss.item()); nb += 1

    train_loss = running / max(nb, 1)
    val_loss, ap_acc, ap_don, recKL_acc, recKL_don = evaluate(model, "val")

    print(
        f"[1 epoch] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
        f"PR-AUC(acc)={ap_acc:.4f} PR-AUC(don)={ap_don:.4f} | "
        f"Recall@1L(acc)={recKL_acc[1]:.4f} Recall@1L(don)={recKL_don[1]:.4f}"
    )

if __name__ == "__main__":
    main()
