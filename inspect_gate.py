# inspect_gate.py
"""
baseline 설정으로 짧게 학습시킨 뒤, 각 레이어의 gate(sigmoid(Linear(h))) 값이
crop 구간에서 실제로 얼마나 attention 쪽으로 열려 있는지 직접 확인.

- gate 값이 0 근처에 몰려있으면 -> attention 브랜치가 학습 중 거의 안 쓰이고
  있다는 뜻 (branch starvation 가설 지지)
- gate 값이 0.3~0.7 등 골고루 퍼져있거나 0.5 이상으로 몰려있으면 -> attention이
  실제로 섞여 쓰이고 있는데도 성능 기여가 적다는 뜻 (태스크 특성 가설 지지)

실행:
    python inspect_gate.py
"""
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CFG
from data import NpzShardIterable, collate_fn
from engine import run_splice3_epoch
from models.mymodel import BioTransMambaProxBiMamba2

TRAIN_BATCHES = 4000  # quickscreen과 동일 스케일로 짧게 학습
D_MODEL = 256
N_LAYERS = 12
N_HEADS = 8
ATTN_WINDOW = 128
D_STATE = 64
D_CONV = 4
EXPAND = 2


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_forward_fn(model):
    def forward_fn(seq_bytes, device):
        return model(seq_bytes_list=seq_bytes, dna_list=None, device=device)
    return forward_fn


@torch.no_grad()
def collect_gate_stats(model, dl, cfg, n_batches: int = 50):
    """각 레이어 gate의 crop 구간 내 평균/표준편차, 레이어별 분포를 수집."""
    model.eval()
    device = cfg.DEVICE
    lo, hi = model._local_range()

    per_layer_vals = [[] for _ in range(len(model.layers))]

    it = iter(dl)
    for _ in range(n_batches):
        seq_bytes, y = next(it)
        x = model.encoder(seq_bytes, device=device)
        x = x.permute(0, 2, 1).contiguous()
        if model.use_conv_stem:
            for blk in model.conv_blocks:
                x = blk(x)
        x = x.permute(0, 2, 1).contiguous()

        for li, layer in enumerate(model.layers):
            h = layer["norm"](x)
            if model.use_local_attention and model.gate_mode == "learned":
                g = torch.sigmoid(layer["gate"](h))[:, lo:hi, :]  # (B, SL, 1)
                per_layer_vals[li].append(g.flatten().cpu())

            out_mamba = layer["mamba"](h)
            if model.use_local_attention:
                h_loc = h[:, lo:hi, :]
                out_loc, (K, V) = layer["attn"](h_loc)
                out_attn_full = torch.zeros_like(out_mamba)
                out_attn_full[:, lo:hi, :] = out_loc
                if model.enable_memory_inject:
                    inj = layer["mem"](K, V)
                    out_mamba = out_mamba + inj
                if model.gate_mode == "learned":
                    g_full = torch.sigmoid(layer["gate"](h))
                elif model.gate_mode == "fixed":
                    g_full = torch.full(
                        (x.size(0), x.size(1), 1), model.fixed_gate_value,
                        device=x.device, dtype=x.dtype,
                    )
                else:
                    g_full = torch.ones((x.size(0), x.size(1), 1), device=x.device, dtype=x.dtype)
                x_new = out_mamba.clone()
                x_new[:, lo:hi, :] = out_mamba[:, lo:hi, :] + g_full[:, lo:hi, :] * (
                    out_attn_full[:, lo:hi, :] - out_mamba[:, lo:hi, :]
                )
            else:
                x_new = out_mamba

            x = x + layer["drop"](x_new)
            x = x + layer["ffn"](layer["ffn_norm"](x))

    print("\n[레이어별 gate 값 통계] (crop 구간, learned gate 기준)")
    print(f"{'layer':>6} {'mean':>8} {'std':>8} {'median':>8} {'%<0.1':>8} {'%>0.9':>8}")
    for li, vals in enumerate(per_layer_vals):
        if not vals:
            print("  gate_mode가 'learned'가 아니거나 use_local_attention=False라 gate가 없습니다.")
            return
        v = torch.cat(vals).numpy()
        pct_low = (v < 0.1).mean() * 100
        pct_high = (v > 0.9).mean() * 100
        print(f"{li:6d} {v.mean():8.4f} {v.std():8.4f} {np.median(v):8.4f} {pct_low:7.1f}% {pct_high:7.1f}%")


def main():
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
        input_len=cfg.INPUT_LEN, sl=cfg.SL, crop=cfg.CROP,
        d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS, attn_window=ATTN_WINDOW,
        dropout=0.1, d_state=D_STATE, d_conv=D_CONV, expand=EXPAND,
        num_classes=cfg.NUM_CLASSES, use_tissue_head=False,
        use_local_attention=True, gate_mode="learned",
    ).to(device)
    forward_fn = make_forward_fn(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    train_dl = DataLoader(
        NpzShardIterable(cfg.DATA_DIR, split="train", seed=cfg.SEED, shuffle_files=True, shuffle_within=True),
        batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    print(f"[1/2] baseline을 {TRAIN_BATCHES}배치만 학습 (quickscreen과 동일 스케일)...")
    train_out = run_splice3_epoch(model, train_dl, forward_fn, cfg, optimizer=opt, max_batches=TRAIN_BATCHES)
    print(f"  train_loss={train_out['loss']:.4f}")

    print("\n[2/2] 학습된 모델의 gate 값 분포 확인 (val 50배치)...")
    val_dl = DataLoader(
        NpzShardIterable(cfg.DATA_DIR, split="val", seed=cfg.SEED, shuffle_files=False, shuffle_within=False),
        batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    collect_gate_stats(model, val_dl, cfg, n_batches=50)

    print("\n[해석 가이드]")
    print("  - mean이 전반적으로 낮고(<0.2) %<0.1 비율이 높으면: attention 브랜치가 거의 꺼져있음")
    print("    -> (B) branch starvation 가능성. gate bias 초기화를 바꿔서 재실험 고려.")
    print("  - mean이 0.3~0.7 사이로 고르게 분포하면: attention이 실제로 섞여 쓰이고 있음")
    print("    -> (A) 섞여 쓰이는데도 성능 기여가 적은 것 -> 태스크 특성상 중복 정보일 가능성.")


if __name__ == "__main__":
    main()