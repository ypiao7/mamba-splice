# splice/bench_speed.py
"""
전체 ablation을 돌리기 전에 실제 소요 시간을 추정하기 위한 벤치마크.

측정하는 것:
  1) train/val/test 각 split의 총 배치 수 (현재 BATCH_SIZE 기준)
  2) 지금 config.py 설정(BATCH_SIZE, NUM_WORKERS)에서 배치당 학습 시간
  3) BATCH_SIZE / NUM_WORKERS를 바꿨을 때 배치당 시간이 어떻게 변하는지
     (같은 총 샘플 수를 처리하는 데 걸리는 시간을 비교하기 위해
      "샘플당 시간"으로도 환산해서 보여줌)

실행:
    python bench_speed.py
"""
import time

import torch
from torch.utils.data import DataLoader

from config import CFG
from data import NpzShardIterable, collate_fn
from metrics import masked_ce_loss
from models.mymodel import BioTransMambaProxBiMamba2

N_WARMUP = 3   # Triton/Mamba2 JIT 컴파일 워밍업 (측정에서 제외)
N_TIMED = 20   # 실제 시간 측정에 쓸 배치 수


def count_total_batches(cfg, split: str, batch_size: int) -> int:
    dl = DataLoader(
        NpzShardIterable(cfg.DATA_DIR, split=split, seed=cfg.SEED, shuffle_files=False, shuffle_within=False),
        batch_size=batch_size, num_workers=0, collate_fn=collate_fn,
    )
    n = 0
    for _ in dl:
        n += 1
    return n


def build_model(cfg, device):
    torch.manual_seed(cfg.SEED)
    model = BioTransMambaProxBiMamba2(
        input_len=cfg.INPUT_LEN, sl=cfg.SL, crop=cfg.CROP,
        d_model=256, n_layers=12, n_heads=8, attn_window=128,
        dropout=0.1, d_state=64, d_conv=4, expand=2,
        num_classes=cfg.NUM_CLASSES, use_tissue_head=False,
    ).to(device)
    return model


def bench_one_config(cfg, device, batch_size: int, num_workers: int):
    model = build_model(cfg, device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    dl = DataLoader(
        NpzShardIterable(cfg.DATA_DIR, split="train", seed=cfg.SEED, shuffle_files=True, shuffle_within=True),
        batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    it = iter(dl)
    model.train()

    # 워밍업 (JIT 컴파일 등 1회성 비용 제외)
    for _ in range(N_WARMUP):
        seq_bytes, y = next(it)
        y = y.to(device)
        logits = model(seq_bytes_list=seq_bytes, dna_list=None, device=device)
        loss = masked_ce_loss(logits, y, crop=cfg.CROP, sl=cfg.SL, ignore_index=cfg.IGNORE_INDEX)
        if loss is None:
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    n_done = 0
    while n_done < N_TIMED:
        seq_bytes, y = next(it)
        y = y.to(device)
        logits = model(seq_bytes_list=seq_bytes, dna_list=None, device=device)
        loss = masked_ce_loss(logits, y, crop=cfg.CROP, sl=cfg.SL, ignore_index=cfg.IGNORE_INDEX)
        if loss is None:
            continue  # 이 배치는 valid label이 하나도 없어서 시간 측정에서 제외
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        n_done += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    sec_per_batch = elapsed / n_done
    sec_per_sample = sec_per_batch / batch_size

    del model, opt, dl, it
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return sec_per_batch, sec_per_sample


def main():
    cfg = CFG()
    device = cfg.DEVICE
    if "cuda" in device:
        idx = int(device.split(":")[1]) if ":" in device else 0
        torch.cuda.set_device(idx)
        device = torch.device(f"cuda:{idx}")
    else:
        device = torch.device("cpu")
    cfg.DEVICE = device

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    print(f"[Device] {device}")
    print(f"[Config] current BATCH_SIZE={cfg.BATCH_SIZE}, NUM_WORKERS={cfg.NUM_WORKERS}, EPOCHS={cfg.EPOCHS}")

    # 1) 총 배치 수 (현재 BATCH_SIZE 기준)
    print("\n[1/3] 전체 배치 수 세는 중 (한 번은 전체 스캔이라 시간이 좀 걸릴 수 있음)...")
    t0 = time.perf_counter()
    n_train = count_total_batches(cfg, "train", cfg.BATCH_SIZE)
    n_val = count_total_batches(cfg, "val", cfg.BATCH_SIZE)
    n_test = count_total_batches(cfg, "test", cfg.BATCH_SIZE)
    print(f"  train batches={n_train}, val batches={n_val}, test batches={n_test} "
          f"(counting took {time.perf_counter()-t0:.1f}s)")

    # 2) 현재 설정으로 배치당 시간
    print(f"\n[2/3] 현재 설정(BATCH_SIZE={cfg.BATCH_SIZE}, NUM_WORKERS={cfg.NUM_WORKERS})으로 "
          f"{N_WARMUP}배치 워밍업 + {N_TIMED}배치 측정 중...")
    sec_per_batch, sec_per_sample = bench_one_config(cfg, device, cfg.BATCH_SIZE, cfg.NUM_WORKERS)
    epoch_sec = sec_per_batch * n_train
    print(f"  sec/batch={sec_per_batch:.3f}  sec/sample={sec_per_sample:.4f}")
    print(f"  -> train 1 epoch 추정: {epoch_sec/60:.1f} 분 ({epoch_sec/3600:.2f} 시간)")
    print(f"  -> baseline {cfg.EPOCHS} epoch 추정(학습만, eval 제외): {epoch_sec*cfg.EPOCHS/3600:.2f} 시간")

    # 3) BATCH_SIZE / NUM_WORKERS 조합 비교
    print(f"\n[3/3] BATCH_SIZE / NUM_WORKERS 조합별 비교 (sec/sample 기준, 낮을수록 좋음)...")
    candidates = [
        (cfg.BATCH_SIZE, cfg.NUM_WORKERS),
        (8, 0),
        (8, 4),
        (16, 4),
        (16, 8),
    ]
    # 중복 제거
    seen = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    results = []
    for bs, nw in candidates:
        try:
            spb, sps = bench_one_config(cfg, device, bs, nw)
            results.append((bs, nw, spb, sps))
            print(f"  BATCH_SIZE={bs:3d} NUM_WORKERS={nw:2d} -> sec/batch={spb:.3f}  sec/sample={sps:.4f}")
        except RuntimeError as e:
            print(f"  BATCH_SIZE={bs:3d} NUM_WORKERS={nw:2d} -> 실패 (OOM 등): {e}")

    if results:
        best = min(results, key=lambda r: r[3])
        baseline_sps = sec_per_sample
        speedup = baseline_sps / best[3]
        print(f"\n  최적 조합: BATCH_SIZE={best[0]}, NUM_WORKERS={best[1]} "
              f"(현재 대비 {speedup:.2f}배 빠름)")
        est_epoch_sec_best = best[3] * cfg.BATCH_SIZE * n_train  # n_train은 현재 batch_size 기준 배치 수라 샘플수로 환산
        total_samples = n_train * cfg.BATCH_SIZE
        est_epoch_sec_best = best[3] * total_samples
        print(f"  이 조합으로 1 epoch 추정: {est_epoch_sec_best/60:.1f} 분")


if __name__ == "__main__":
    main()