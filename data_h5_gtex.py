# splice/data_h5_gtex.py
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# 0,1,2,3,4 = N,A,C,G,T
_ID2CH = np.array([ord("N"), ord("A"), ord("C"), ord("G"), ord("T")], dtype=np.uint8)

def tokens_to_bytes(tokens_0_4: np.ndarray) -> bytes:
    return _ID2CH[tokens_0_4.astype(np.int64)].tobytes()

class GTExH5_Center3000_SL1000(Dataset):
    """
    Returns:
      seq_bytes: bytes length 3000
      y: LongTensor (1000,) with values in {0,1,2} (or ignore_index if you set some)
    """
    def __init__(self, h5_paths, ignore_index=-1):
        self.h5_paths = list(h5_paths)
        self.ignore_index = int(ignore_index)

        self.L_full = 9000
        self.L_used = 3000
        self.SL = 1000

        # full(9000)에서 중앙 3000
        self.crop_start_full = 3000
        self.crop_end_full = 6000

        self._h5 = None

        self.index = []
        for fid, path in enumerate(self.h5_paths):
            with h5py.File(path, "r") as f:
                xkeys = sorted([k for k in f.keys() if k.startswith("X")], key=lambda x: int(x[1:]))
                for xk in xkeys:
                    n = f[xk].shape[0]
                    for r in range(n):
                        self.index.append((fid, xk, r))

    def __len__(self):
        return len(self.index)

    def _get_h5(self, fid):
        if self._h5 is None:
            self._h5 = {}
        if fid not in self._h5:
            self._h5[fid] = h5py.File(self.h5_paths[fid], "r")
        return self._h5[fid]

    def __getitem__(self, i):
        fid, xk, r = self.index[i]
        f = self._get_h5(fid)
        yk = xk.replace("X", "Y")

        X = np.array(f[xk][r], dtype=np.int64)      # (9000,) values 0..4
        Y = np.array(f[yk][r], dtype=np.float32)    # (18,1000)

        # 중앙 3000만 사용
        x_used = X[self.crop_start_full:self.crop_end_full]          # (3000,)
        seq_bytes = tokens_to_bytes(x_used)

        # 중앙 1000 라벨 (3-class)
        y = np.argmax(Y[:3, :], axis=0).astype(np.int64)             # (1000,)

        y_tissue = Y[3:, :].astype(np.float32)  # (15,1000)

        return seq_bytes, torch.from_numpy(y), torch.from_numpy(y_tissue)

def collate_fn_h5(batch):
    seq_bytes = [b for (b, _, _) in batch]
    y = torch.stack([y for (_, y, _) in batch], dim=0)  # (B,1000)
    y_tis = torch.stack([t for (_, _, t) in batch], dim=0)
    return seq_bytes, y, y_tis
