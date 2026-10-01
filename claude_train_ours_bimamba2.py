import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CFG
from data import NpzShardIterable, collate_fn
from engine import run_splice3_epoch, Splice3MetricsTracker
from models.mymodel import BioTransMambaProxBiMamba2


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

# ---- Ablation 플래그 (기본값 = baseline, 전부 켜짐) ----
# 예시:
#   memory injection 제거          -> ENABLE_MEMORY_INJECT = False
#   local attention 통째로 제거     -> USE_LOCAL_ATTENTION = False
#   gate 대신 고정 비율 0.5로 mix   -> GATE_MODE = "fixed"
#   crop 구간을 attn으로 완전 대체  -> GATE_MODE = "hard"
#   BiMamba2 단방향(정방향만)       -> BIDIRECTIONAL_MAMBA = False
#   CNN motif stem 제거            -> USE_CONV_STEM = False
USE_LOCAL_ATTENTION = True
GATE_MODE = "learned"          # "learned" | "fixed" | "hard"
FIXED_GATE_VALUE = 0.5
BIDIRECTIONAL_MAMBA = True
USE_CONV_STEM = True

MAX_TRAIN_BATCHES = 0
MAX_EVAL_BATCHES = 0
# =========================================================


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_forward_fn(model, use_bpe: bool):
    def forward_fn(seq_bytes, device):
        if use_bpe:
            dna = [s.decode("ascii") for s in seq_bytes]
            return model(seq_bytes_list=None, dna_list=dna, device=device)
        return model(seq_bytes_list=seq_bytes, dna_list=None, device=device)
    return forward_fn


def main():
    t0 = time.perf_counter()
    cfg = CFG()

    device = cfg.DEVICE
    if "cuda" in device:
        idx = int(device.split(":")[1]) if ":" in device else 0
        torch.cuda.set_device(idx)
        cfg.DEVICE = torch.device(f"cuda:{idx}")
    else:
        cfg.DEVICE = torch.device("cpu")
    device = cfg.DEVICE

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
        use_tissue_head=False,  # hg38 실험: tissue head 사용 안 함
        use_local_attention=USE_LOCAL_ATTENTION,
        gate_mode=GATE_MODE,
        fixed_gate_value=FIXED_GATE_VALUE,
        bidirectional_mamba=BIDIRECTIONAL_MAMBA,
        use_conv_stem=USE_CONV_STEM,
    ).to(device)
    forward_fn = make_forward_fn(model, USE_BPE)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[Config] use_local_attention={USE_LOCAL_ATTENTION} enable_memory_inject={ENABLE_MEMORY_INJECT} "
        f"gate_mode={GATE_MODE} bidirectional_mamba={BIDIRECTIONAL_MAMBA} use_conv_stem={USE_CONV_STEM} "
        f"| n_params={n_params:,}"
    )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    train_dl = DataLoader(
        NpzShardIterable(cfg.DATA_DIR, split="train", seed=cfg.SEED, shuffle_files=True, shuffle_within=True),
        batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    def eval_dl(split):
        return DataLoader(
            NpzShardIterable(cfg.DATA_DIR, split=split, seed=cfg.SEED, shuffle_files=False, shuffle_within=False),
            batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn,
            pin_memory=(device.type == "cuda"),
        )

    for ep in range(1, cfg.EPOCHS + 1):
        train_out = run_splice3_epoch(model, train_dl, forward_fn, cfg, optimizer=opt, max_batches=MAX_TRAIN_BATCHES)
        val_out = run_splice3_epoch(model, eval_dl("val"), forward_fn, cfg, optimizer=None, max_batches=MAX_EVAL_BATCHES)
        print(Splice3MetricsTracker.log_line(f"[Epoch {ep}] train_loss={train_out['loss']:.4f} | val_", val_out))

    test_out = run_splice3_epoch(model, eval_dl("test"), forward_fn, cfg, optimizer=None, max_batches=MAX_EVAL_BATCHES)
    print("\n[TEST]")
    print(Splice3MetricsTracker.log_line("", test_out))

    elapsed = time.perf_counter() - t0
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    print(f"\n[Done] Total runtime: {elapsed:.2f} sec ({h:02d}:{m:02d}:{s:02d})")


if __name__ == "__main__":
    main()