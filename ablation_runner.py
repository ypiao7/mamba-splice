"""
BioTransMambaProxBiMamba2의 핵심 구조 ablation을 7개 조합, seed 1개로 순차 실행.

- 조합마다 동일 seed로 모델을 새로 초기화 (공정 비교)
- train/val/test dataloader는 전체 실행에서 한 번만 만들어 재사용
- 조합 하나 끝날 때마다 결과를 ablation_results.csv에 즉시 append
  (중간에 죽어도 그 전까지 결과는 안 날아감)
- 조합 하나가 에러 나도(OOM 등) 잡아서 CSV에 error로 기록하고 다음 조합 계속 진행

실행:
    python ablation_runner.py
결과:
    ./ablation_results.csv
"""
import csv
import os
import random
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CFG
from data import NpzShardIterable, collate_fn
from engine import run_splice3_epoch, Splice3MetricsTracker
from models.mymodel import BioTransMambaProxBiMamba2

# =========================================================
# baseline과 동일하게 고정할 공통 하이퍼파라미터
# =========================================================
USE_BPE = False
BPE_TOKENIZER = "zhihan1996/DNABERT-2-117M"
LOCAL_MARGIN = 0

D_MODEL = 256
N_LAYERS = 12
N_HEADS = 8
ATTN_WINDOW = 1
D_STATE = 64
D_CONV = 4
EXPAND = 2

# =========================================================
# ✅ 빠른 스크리닝 스위치
#    True  -> epoch 1개, train/eval 배치 수 제한 (7조합 감 잡기용, 몇 분 내로 끝남)
#    False -> config.py의 EPOCHS 그대로, 배치 제한 없음 (진짜 본 실험용)
# =========================================================
QUICK_SCREEN = True
QUICK_SCREEN_EPOCHS = 1
QUICK_SCREEN_MAX_TRAIN_BATCHES = 4000
QUICK_SCREEN_MAX_EVAL_BATCHES = 500

MAX_TRAIN_BATCHES = QUICK_SCREEN_MAX_TRAIN_BATCHES if QUICK_SCREEN else 0
MAX_EVAL_BATCHES = QUICK_SCREEN_MAX_EVAL_BATCHES if QUICK_SCREEN else 0

# =========================================================
# 실행할 7개 조합: (이름, BioTransMambaProxBiMamba2에 덮어쓸 kwargs)
# =========================================================
ABLATIONS = [
    ("baseline",             dict()),
    ("no_local_attention",   dict(use_local_attention=False)),
    ("no_memory_inject",     dict(enable_memory_inject=False)),
    ("gate_fixed_0.5",       dict(gate_mode="fixed", fixed_gate_value=0.5)),
    ("gate_hard",            dict(gate_mode="hard")),
    ("unidirectional_mamba", dict(bidirectional_mamba=False)),
    ("no_conv_stem",         dict(use_conv_stem=False)),
]


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


def build_dataloaders(cfg):
    pin = (cfg.DEVICE.type == "cuda")
    train_dl = DataLoader(
        NpzShardIterable(cfg.DATA_DIR, split="train", seed=cfg.SEED, shuffle_files=True, shuffle_within=True),
        batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=pin,
    )
    val_dl = DataLoader(
        NpzShardIterable(cfg.DATA_DIR, split="val", seed=cfg.SEED, shuffle_files=False, shuffle_within=False),
        batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=pin,
    )
    test_dl = DataLoader(
        NpzShardIterable(cfg.DATA_DIR, split="test", seed=cfg.SEED, shuffle_files=False, shuffle_within=False),
        batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=pin,
    )
    return train_dl, val_dl, test_dl


def run_one(name: str, kwargs: dict, cfg, epochs: int, train_dl, val_dl, test_dl) -> dict:
    print(f"\n{'='*70}\n[Ablation] {name}  kwargs={kwargs}\n{'='*70}")
    set_seed(cfg.SEED)  # 조합마다 동일 seed로 초기화 -> 조합 간 공정 비교

    model = BioTransMambaProxBiMamba2(
        input_len=cfg.INPUT_LEN, sl=cfg.SL, crop=cfg.CROP,
        d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS, attn_window=ATTN_WINDOW,
        dropout=0.1, d_state=D_STATE, d_conv=D_CONV, expand=EXPAND,
        local_margin=LOCAL_MARGIN, use_bpe=USE_BPE, bpe_tokenizer=BPE_TOKENIZER,
        num_classes=cfg.NUM_CLASSES, use_tissue_head=False,
        **kwargs,
    ).to(cfg.DEVICE)
    forward_fn = make_forward_fn(model, USE_BPE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  n_params={n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        train_out = run_splice3_epoch(model, train_dl, forward_fn, cfg, optimizer=opt, max_batches=MAX_TRAIN_BATCHES)
        val_out = run_splice3_epoch(model, val_dl, forward_fn, cfg, optimizer=None, max_batches=MAX_EVAL_BATCHES)
        print(Splice3MetricsTracker.log_line(f"  [{name}][Epoch {ep}] train_loss={train_out['loss']:.4f} | val_", val_out))

    test_out = run_splice3_epoch(model, test_dl, forward_fn, cfg, optimizer=None, max_batches=MAX_EVAL_BATCHES)
    elapsed = time.perf_counter() - t0
    print(Splice3MetricsTracker.log_line(f"  [{name}][TEST] ", test_out))

    # 다음 조합을 위해 GPU 메모리 정리
    del model, opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "name": name,
        "kwargs": str(kwargs),
        "seed": cfg.SEED,
        "n_params": n_params,
        "epochs": epochs,
        "quick_screen": QUICK_SCREEN,
        "max_train_batches": MAX_TRAIN_BATCHES,
        "max_eval_batches": MAX_EVAL_BATCHES,
        "val_loss": round(val_out["loss"], 6),
        "val_pr_auc_acc": round(val_out["pr_auc_acc"], 6),
        "val_pr_auc_don": round(val_out["pr_auc_don"], 6),
        "val_rec1L_acc": round(val_out["rec_kl_acc"][1], 6),
        "val_rec1L_don": round(val_out["rec_kl_don"][1], 6),
        "test_loss": round(test_out["loss"], 6),
        "test_pr_auc_acc": round(test_out["pr_auc_acc"], 6),
        "test_pr_auc_don": round(test_out["pr_auc_don"], 6),
        "test_rec1L_acc": round(test_out["rec_kl_acc"][1], 6),
        "test_rec1L_don": round(test_out["rec_kl_don"][1], 6),
        "elapsed_sec": round(elapsed, 1),
        "error": "",
    }


def append_row_to_csv(row: dict, path: str):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    cfg = CFG()

    device = cfg.DEVICE
    if "cuda" in device:
        idx = int(device.split(":")[1]) if ":" in device else 0
        torch.cuda.set_device(idx)
        cfg.DEVICE = torch.device(f"cuda:{idx}")
    else:
        cfg.DEVICE = torch.device("cpu")

    epochs = QUICK_SCREEN_EPOCHS if QUICK_SCREEN else cfg.EPOCHS

    # 결과 파일명에 seed를 포함 -> config.py의 SEED만 바꿔서 다른 GPU/seed로
    # 동시에 돌려도 서로 다른 CSV에 쓰여서 충돌하지 않음.
    suffix = "_quickscreen" if QUICK_SCREEN else ""
    out_csv = f"ablation_results{suffix}_seed{cfg.SEED}.csv"

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    train_dl, val_dl, test_dl = build_dataloaders(cfg)

    print(
        f"[Ablation Runner] QUICK_SCREEN={QUICK_SCREEN} | device={cfg.DEVICE} | "
        f"{len(ABLATIONS)} configs x seed={cfg.SEED} "
        f"x {epochs} epochs x (train<={MAX_TRAIN_BATCHES or 'all'}, eval<={MAX_EVAL_BATCHES or 'all'}) -> {out_csv}"
    )

    for name, kwargs in ABLATIONS:
        try:
            row = run_one(name, kwargs, cfg, epochs, train_dl, val_dl, test_dl)
        except Exception as e:
            print(f"[ERROR] {name} failed: {e}")
            traceback.print_exc()
            row = {
                "name": name, "kwargs": str(kwargs), "seed": cfg.SEED,
                "n_params": "", "epochs": epochs, "quick_screen": QUICK_SCREEN,
                "max_train_batches": MAX_TRAIN_BATCHES, "max_eval_batches": MAX_EVAL_BATCHES,
                "val_loss": "", "val_pr_auc_acc": "", "val_pr_auc_don": "",
                "val_rec1L_acc": "", "val_rec1L_don": "",
                "test_loss": "", "test_pr_auc_acc": "", "test_pr_auc_don": "",
                "test_rec1L_acc": "", "test_rec1L_don": "",
                "elapsed_sec": "", "error": str(e),
            }
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        append_row_to_csv(row, out_csv)
        print(f"[Saved] {name} -> {out_csv}")

    print(f"\n[Done] All {len(ABLATIONS)} ablation runs finished. Results in {out_csv}")


if __name__ == "__main__":
    main()