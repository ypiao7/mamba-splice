# data.py
import os, glob
import numpy as np
import torch
from torch.utils.data import IterableDataset

class NpzShardIterable(IterableDataset):
    def __init__(self, data_dir: str, split: str, seed: int, shuffle_files=True, shuffle_within=True):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.seed = seed
        self.shuffle_files = shuffle_files
        self.shuffle_within = shuffle_within
        self.epoch = 0  # ✅ 추가

        self.files = sorted(glob.glob(os.path.join(data_dir, f"{split}_*.npz")))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No npz shards found for split={split} in {data_dir}")

    def set_epoch(self, epoch: int):
        # ✅ DataLoader 밖에서 epoch마다 호출
        self.epoch = int(epoch)

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = 0 if info is None else info.id

        # ✅ epoch를 seed에 섞어서 매 epoch마다 shuffle 결과가 달라지게
        base = (
            self.seed
            + (0 if self.split == "train" else 123)
            + 1000 * wid
            + 100_000 * self.epoch
        )
        rng = np.random.default_rng(base)

        files = self.files.copy()
        if self.shuffle_files:
            rng.shuffle(files)

        for fp in files:
            z = np.load(fp)
            seq = z["seq"]  # (N,)
            y   = z["y"]    # (N,SL)

            n = y.shape[0]
            idx = np.arange(n)
            if self.shuffle_within:
                rng.shuffle(idx)

            for i in idx:
                yield bytes(seq[i]), torch.from_numpy(y[i].astype(np.int64))


def collate_fn(batch):
    seq_bytes, y = zip(*batch)
    y = torch.stack(y, dim=0)
    return list(seq_bytes), y
