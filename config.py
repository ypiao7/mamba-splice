#config.py
from dataclasses import dataclass

@dataclass
class CFG:
    DATA_DIR: str = "/data/piao/data/data/spliceai_cut_SL1000_CL2000/"
    #DATA_DIR: str = "/home/yjpiao/data/spliceai_cut_SL1000_CL2000/"
    SL: int = 1000
    CL: int = 2000
    CROP: int = CL // 2
    INPUT_LEN: int = SL + CL

    NUM_CLASSES: int = 3
    IGNORE_INDEX: int = -1

    BATCH_SIZE: int = 16
    NUM_WORKERS: int = 4
    SEED: int = 1

    DEVICE: str = "cuda:1"
    EPOCHS: int = 2
    LR: float = 1e-4 #1e-4 my model
    WEIGHT_DECAY: float = 1e-2

    # TopK
    TOPK_KS: tuple = (1, 5, 10, 20)
    TOPK_TOL: int = 1
    TOPK_POSITIVE_ONLY: bool = True

    # SpliceAI-style top-kL
    TOPKL_KS: tuple = (0.5, 1, 2, 4)
    TOPKL_TOL: int = 1  # SpliceAI 완전 동일 비교면 0 추천
    TOPKL_POSITIVE_ONLY: bool = True
