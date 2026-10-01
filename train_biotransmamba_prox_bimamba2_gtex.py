# splice/train_biotransmamba_prox_bimamba2_gtex.py
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import time

from config import CFG
from data import NpzShardIterable, collate_fn
from metrics import (
    masked_ce_loss, masked_bce_with_logits_loss,
    init_topkl_sums, update_topkl_sums, finalize_topkl,
    init_pr_auc_buf, accumulate_pr_auc, finalize_pr_auc,
    init_pr_auc_tissue_buf, accumulate_pr_auc_tissue, finalize_pr_auc_tissue
)
from models.mymodel import BioTransMambaProxBiMamba2
from data_h5_gtex import GTExH5_Center3000_SL1000, collate_fn_h5

TRAIN_H5 = [
    "/data/piao/gtex/dataset_train_1.h5",
    "/data/piao/gtex/dataset_train_2.h5",
    "/data/piao/gtex/dataset_train_3.h5",
]
VAL_H5 = [
    "/data/piao/gtex/dataset_test_3.h5",
]

# =========================================================
# ✅ 여기만 바꿔서 실험
# =========================================================
USE_BPE = False
BPE_TOKENIZER = "zhihan1996/DNABERT-2-117M"

LOCAL_MARGIN = 0
ENABLE_MEMORY_INJECT = True

D_MODEL = 256
N_LAYERS = 12
N_HEADS = 8
ATTN_WINDOW = 128

# BiMamba2 SSD-ish defaults
D_STATE = 64
D_CONV = 4
EXPAND = 2

MAX_TRAIN_BATCHES = 0
MAX_EVAL_BATCHES = 0
# =========================================================


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def format_secs(sec: float) -> str:
    sec = int(round(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


@torch.no_grad()
def evaluate(model, split: str, device: str, cfg: CFG, max_batches: int = 0):
    model.eval()

    # ds = NpzShardIterable(cfg.DATA_DIR, split=split, seed=cfg.SEED, shuffle_files=False, shuffle_within=False)
    # dl = DataLoader(
    #     ds,
    #     batch_size=cfg.BATCH_SIZE,
    #     num_workers=cfg.NUM_WORKERS,
    #     collate_fn=collate_fn,
    #     pin_memory = (device.type == "cuda")
    # )

    ds = GTExH5_Center3000_SL1000(VAL_H5, ignore_index=cfg.IGNORE_INDEX)
    dl = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                          num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn_h5,
                          pin_memory=(device.type == "cuda"))
    total_loss = 0.0
    n_loss_batches = 0

    topkl_acc = init_topkl_sums(cfg.TOPKL_KS)
    topkl_don = init_topkl_sums(cfg.TOPKL_KS)
    pr_buf = init_pr_auc_buf()
    pr_tis = init_pr_auc_tissue_buf(15)

    seen = 0
    for seq_bytes, y, y_tis in dl:
        seen += 1
        y = y.to(device)
        y_tis = y_tis.to(device)

        if USE_BPE:
            dna = [s.decode("ascii") for s in seq_bytes]
            logits, logits_tis = model(seq_bytes_list=None, dna_list=dna, device=device)
        else:
            logits, logits_tis = model(seq_bytes_list=seq_bytes, dna_list=None, device=device)

        #loss = masked_ce_loss(logits, y, crop=cfg.CROP, sl=cfg.SL, ignore_index=cfg.IGNORE_INDEX)

        # tissue-specific  prediction
        loss = masked_bce_with_logits_loss(logits_tis, y_tis, crop=cfg.CROP, sl=cfg.SL)

        if loss is not None:
            total_loss += float(loss.item())
            n_loss_batches += 1

        # accumulate_pr_auc(pr_buf, logits, y, crop=cfg.CROP, sl=cfg.SL, ignore_index=cfg.IGNORE_INDEX)
        #
        # update_topkl_sums(
        #     topkl_acc, logits, y, class_id=1,
        #     crop=cfg.CROP, sl=cfg.SL,
        #     ks=cfg.TOPKL_KS, tol=cfg.TOPKL_TOL,
        #     ignore_index=cfg.IGNORE_INDEX,
        #     positive_only=cfg.TOPKL_POSITIVE_ONLY
        # )
        # update_topkl_sums(
        #     topkl_don, logits, y, class_id=2,
        #     crop=cfg.CROP, sl=cfg.SL,
        #     ks=cfg.TOPKL_KS, tol=cfg.TOPKL_TOL,
        #     ignore_index=cfg.IGNORE_INDEX,
        #     positive_only=cfg.TOPKL_POSITIVE_ONLY
        # )
        logits_tis_center = logits_tis[:, cfg.CROP:cfg.CROP + cfg.SL, :].permute(0, 2, 1).contiguous()  # (B,15,1000)
        accumulate_pr_auc_tissue(pr_tis, logits_tis_center, y_tis, crop=0, sl=cfg.SL, thr=0.5)

        if max_batches and seen >= max_batches:
            break

    mean_loss = total_loss / max(n_loss_batches, 1)

    # ap_acc, ap_don = finalize_pr_auc(pr_buf)
    # rec_kl_acc, _ = finalize_topkl(topkl_acc, cfg.TOPKL_KS)
    # rec_kl_don, _ = finalize_topkl(topkl_don, cfg.TOPKL_KS)

    ap_tissues = finalize_pr_auc_tissue(pr_tis)

    # return {
    #     "loss": mean_loss,
    #     "pr_auc_acc": ap_acc,
    #     "pr_auc_don": ap_don,
    #     "rec_kl_acc": rec_kl_acc,
    #     "rec_kl_don": rec_kl_don,
    #     "batches": seen,
    # }

    return {
        "loss": mean_loss,
        "pr_auc_acc": ap_tissues,
        "batches": seen,
    }

def main():
    t0 = time.perf_counter()
    cfg = CFG()

    device = cfg.DEVICE

    if "cuda" in device:
        idx = int(device.split(":")[1]) if ":" in device else 0
        torch.cuda.set_device(idx)
        device = torch.device(f"cuda:{idx}")
    else:
        device = torch.device("cpu")

    set_seed(cfg.SEED)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    model = BioTransMambaProxBiMamba2(
        input_len=cfg.INPUT_LEN,
        sl=cfg.SL,
        crop=cfg.CROP,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        attn_window=ATTN_WINDOW,
        dropout=0.1,
        d_state=D_STATE,
        d_conv=D_CONV,
        expand=EXPAND,
        local_margin=LOCAL_MARGIN,
        enable_memory_inject=ENABLE_MEMORY_INJECT,
        use_bpe=USE_BPE,
        bpe_tokenizer=BPE_TOKENIZER,
        num_classes=cfg.NUM_CLASSES,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    # train_ds = NpzShardIterable(cfg.DATA_DIR, split="train", seed=cfg.SEED, shuffle_files=True, shuffle_within=True)
    # train_dl = DataLoader(
    #     train_ds,
    #     batch_size=cfg.BATCH_SIZE,
    #     num_workers=cfg.NUM_WORKERS,
    #     collate_fn=collate_fn,
    #     pin_memory = (device.type == "cuda")
    # )

    train_ds = GTExH5_Center3000_SL1000(TRAIN_H5, ignore_index=cfg.IGNORE_INDEX)
    train_dl = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                          num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn_h5,
                          pin_memory=(device.type == "cuda"))


    for ep in range(1, cfg.EPOCHS + 1):
        model.train()
        running = 0.0
        nb = 0

        for bi, (seq_bytes, y, y_tis) in enumerate(train_dl, start=1):
            y = y.to(device)
            y_tis = y_tis.to(device)

            # s = y_tis.sum(dim=1)  # (B,1000)
            # print(s.min().item(), s.max().item(), s.mean().item())
            #
            # break

            if USE_BPE:
                dna = [s.decode("ascii") for s in seq_bytes]
                logits, logits_tis = model(seq_bytes_list=None, dna_list=dna, device=device)
            else:
                logits, logits_tis = model(seq_bytes_list=seq_bytes, dna_list=None, device=device)

            #loss = masked_ce_loss(logits, y, crop=cfg.CROP, sl=cfg.SL, ignore_index=cfg.IGNORE_INDEX)

            #tissu-specific  prediction
            loss = masked_bce_with_logits_loss(logits_tis, y_tis, crop=cfg.CROP, sl=cfg.SL)

            if loss is None:
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.item())
            nb += 1

            if MAX_TRAIN_BATCHES and bi >= MAX_TRAIN_BATCHES:
                break

        train_loss = running / max(nb, 1)

        val_metrics = evaluate(model, "val", device, cfg, max_batches=MAX_EVAL_BATCHES)
        # print(
        #     f"[Epoch {ep}] "
        #     f"train_loss={train_loss:.4f} | "
        #     f"val_loss={val_metrics['loss']:.4f} | "
        #     f"PR-AUC(acc)={val_metrics['pr_auc_acc']:.4f} PR-AUC(don)={val_metrics['pr_auc_don']:.4f} | "
        #     f"Recall@1L(acc)={val_metrics['rec_kl_acc'][1]:.4f} Recall@1L(don)={val_metrics['rec_kl_don'][1]:.4f} | "
        #     f"train_batches={bi}"
        # )

        print("Tissue AP:", ["%.4f" % x for x in val_metrics['pr_auc_acc']])
        print("Mean Tissue AP:", np.nanmean(val_metrics['pr_auc_acc']))

    test_metrics = evaluate(model, "test", device, cfg, max_batches=MAX_EVAL_BATCHES)
    print("\n[TEST]")
    # print(
    #     f"loss={test_metrics['loss']:.4f} | "
    #     f"PR-AUC(acc)={test_metrics['pr_auc_acc']:.4f} PR-AUC(don)={test_metrics['pr_auc_don']:.4f} | "
    #     f"Recall@1L(acc)={test_metrics['rec_kl_acc'][1]:.4f} Recall@1L(don)={test_metrics['rec_kl_don'][1]:.4f} | "
    #     f"eval_batches={test_metrics['batches']}"
    # )

    print("Tissue AP:", ["%.4f" % x for x in test_metrics['pr_auc_acc']])
    print("Mean Tissue AP:", np.nanmean(test_metrics['pr_auc_acc']))

    elapsed = time.perf_counter() - t0
    h = int(elapsed) // 3600
    m = (int(elapsed) % 3600) // 60
    s = int(elapsed) % 60
    print(f"\n[Done] Total runtime: {elapsed:.2f} sec ({h:02d}:{m:02d}:{s:02d})")


if __name__ == "__main__":
    main()
