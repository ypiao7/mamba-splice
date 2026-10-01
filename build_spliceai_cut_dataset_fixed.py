import os
import numpy as np

# =========================================================
# Global config
# =========================================================
GRCH38_PATH = "/data/piao/data/data/grch38_para.txt"
FASTA_PATH  = "/data/piao/data/data/genome/hg38.fa"
OUT_DIR     = "/data/piao/data/data/spliceai_cut_SL1000_CL2000"

SL = 1000
CL = 2000
SHARD_SIZE = 50000

# ---------------------------------------------------------
# Split config
# ---------------------------------------------------------
# 테스트/밸리데이션은 항상 고정
TEST_CHROMS = {f"chr{i}" for i in [1, 3, 5, 7, 9]}
VAL_CHROMS  = {"chr11"}

# 빠른 학습(임시) 모드
QUICK_TRAIN_ONLY = False
QUICK_TRAIN_CHROMS = {f"chr{i}" for i in [2, 4, 6, 8, 10]}

# 최종 학습(전체) 모드: test/val 제외한 전부
ALL_AUTOSOMES = {f"chr{i}" for i in range(1, 23)}
TRAIN_CHROMS_FULL = (ALL_AUTOSOMES | {"chrX"}) - TEST_CHROMS - VAL_CHROMS

# test에서 homologous(=1) gene 제외
FILTER_TEST_HOMOLOGOUS = True

# single-exon 제거(강력 추천: 시간/용량 절약)
SKIP_SINGLE_EXON = True


# =========================================================
# Helpers
# =========================================================
def ensure_fai(fasta_path: str):
    import pysam
    fai = fasta_path + ".fai"
    if not os.path.exists(fai):
        pysam.faidx(fasta_path)

def parse_exon_list(s: str):
    s = s.strip()
    if s.endswith(","):
        s = s[:-1]
    if not s:
        return []
    return [int(v) for v in s.split(",") if v != ""]

def encode_seq_to_digits(seq: str) -> np.ndarray:
    seq = seq.upper()
    seq = (seq.replace("A", "1")
              .replace("C", "2")
              .replace("G", "3")
              .replace("T", "4")
              .replace("N", "0"))
    seq = "".join(ch if ch in "01234" else "0" for ch in seq)
    return np.fromiter((ord(c) - 48 for c in seq), dtype=np.int8, count=len(seq))

def rc_digits(x: np.ndarray) -> np.ndarray:
    return (5 - x[::-1]) % 5

def ceil_div(a, b):
    return (a + b - 1) // b

def which_split(chrom: str):
    # 1) test 고정
    if chrom in TEST_CHROMS:
        return "test"

    # 2) val 고정 (train/test에서 제외)
    if chrom in VAL_CHROMS:
        return "val"

    # 3) train은 QUICK/FULL로만 달라짐
    if QUICK_TRAIN_ONLY:
        return "train" if chrom in QUICK_TRAIN_CHROMS else None
    else:
        return "train" if chrom in TRAIN_CHROMS_FULL else None


def shard_state(split: str, input_len: int):
    os.makedirs(OUT_DIR, exist_ok=True)
    return {
        "split": split,
        "input_len": input_len,
        "shard_idx": 0,
        "buf_seq": [],
        "buf_y": [],
        "buf_mask": [],
        "buf_gene": [],
        "buf_chr": [],
        "buf_tx_start": [],
        "buf_tx_end": [],
        "buf_strand": [],
        "buf_block_i": [],
    }

def flush_shard(st):
    n = len(st["buf_seq"])
    if n == 0:
        return

    out_path = os.path.join(OUT_DIR, f"{st['split']}_{st['shard_idx']:03d}.npz")

    seq_arr = np.asarray(st["buf_seq"], dtype=f"S{st['input_len']}")
    y_arr   = np.stack(st["buf_y"]).astype(np.int8)
    m_arr   = np.stack(st["buf_mask"]).astype(np.uint8)

    gene_arr = np.asarray(st["buf_gene"], dtype="S64")
    chr_arr  = np.asarray(st["buf_chr"], dtype="S8")
    txs = np.asarray(st["buf_tx_start"], dtype=np.int32)
    txe = np.asarray(st["buf_tx_end"], dtype=np.int32)
    strand_arr = np.asarray(st["buf_strand"], dtype="S1")
    block_i = np.asarray(st["buf_block_i"], dtype=np.int32)

    np.savez_compressed(
        out_path,
        seq=seq_arr,
        y=y_arr,
        mask=m_arr,
        gene=gene_arr,
        chrom=chr_arr,
        tx_start=txs,
        tx_end=txe,
        strand=strand_arr,
        block_i=block_i,
    )

    st["shard_idx"] += 1
    for k in ["buf_seq","buf_y","buf_mask","buf_gene","buf_chr","buf_tx_start","buf_tx_end","buf_strand","buf_block_i"]:
        st[k].clear()

def add_example(st, seq_bytes, y_block, mask_block, gene, chrom, tx_start, tx_end, strand, block_i):
    st["buf_seq"].append(seq_bytes)
    st["buf_y"].append(y_block)
    st["buf_mask"].append(mask_block)
    st["buf_gene"].append(gene.encode("utf-8")[:64])
    st["buf_chr"].append(chrom.encode("utf-8")[:8])
    st["buf_tx_start"].append(tx_start)
    st["buf_tx_end"].append(tx_end)
    st["buf_strand"].append(strand.encode("utf-8"))
    st["buf_block_i"].append(block_i)

    if len(st["buf_seq"]) >= SHARD_SIZE:
        flush_shard(st)


# =========================================================
# Main
# =========================================================
def main():
    assert CL % 2 == 0, "CL must be even"
    pad = CL // 2
    input_len = SL + CL

    import pysam
    ensure_fai(FASTA_PATH)
    fa = pysam.FastaFile(FASTA_PATH)

    train_w = shard_state("train", input_len)
    val_w   = shard_state("val", input_len)   # 현재는 거의 안 쓰임
    test_w  = shard_state("test", input_len)
    writers = {"train": train_w, "val": val_w, "test": test_w}

    map_back = np.array([ord('N'), ord('A'), ord('C'), ord('G'), ord('T')], dtype=np.uint8)

    n_rows = 0
    n_skipped_fetch = 0
    n_filtered_test_hom = 0
    n_skipped_single_exon = 0
    n_points = {"train": 0, "val": 0, "test": 0}

    with open(GRCH38_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")

            # grch38_para.txt: 2번째 열이 homologous flag
            # 최소: gene, flag, chrom, strand, tx_start, tx_end, exon_starts, exon_ends
            if len(parts) < 8:
                continue

            gene = parts[0]
            hom_flag = int(parts[1])  # 1=homologous, 0=non-homologous

            chrom_raw = parts[2]
            strand = parts[3]
            tx_start = int(parts[4])
            tx_end = int(parts[5])
            exon_starts = parse_exon_list(parts[6])
            exon_ends = parse_exon_list(parts[7])

            chrom = chrom_raw if chrom_raw.startswith("chr") else f"chr{chrom_raw}"
            split = which_split(chrom)
            if split is None:
                continue

            # test에서 homologous=1 gene 제거
            if FILTER_TEST_HOMOLOGOUS and split == "test" and hom_flag == 1:
                n_filtered_test_hom += 1
                continue

            if strand not in ("+", "-"):
                continue
            if tx_end <= tx_start:
                continue
            if len(exon_starts) == 0 or len(exon_starts) != len(exon_ends):
                continue

            # sort exons by start
            exons = sorted(zip(exon_starts, exon_ends), key=lambda x: x[0])
            exon_starts = [a for a, _ in exons]
            exon_ends   = [b for _, b in exons]

            has_junc = len(exon_starts) >= 2
            if SKIP_SINGLE_EXON and (not has_junc):
                n_skipped_single_exon += 1
                continue

            # junction coordinates
            jn_start = exon_ends[:-1]   # donor on + (exon end)
            jn_end   = exon_starts[1:]  # acceptor on + (next exon start)

            # transcript sequence T = [tx_start, tx_end)
            try:
                T = fa.fetch(chrom, tx_start, tx_end)
            except Exception:
                n_skipped_fetch += 1
                continue

            S = ("N" * pad) + T + ("N" * pad)
            X0 = encode_seq_to_digits(S)

            L = tx_end - tx_start
            Y0 = -np.ones(L + 1, dtype=np.int8)  # SpliceAI uses +1

            # label
            Y0[:] = 0
            if strand == "+":
                for c in jn_start:
                    if tx_start <= c <= tx_end:
                        idx = c - tx_start
                        if 0 <= idx <= L:
                            Y0[idx] = 2  # donor
                for c in jn_end:
                    if tx_start <= c <= tx_end:
                        idx = c - tx_start
                        if 0 <= idx <= L:
                            Y0[idx] = 1  # acceptor
            else:
                X0 = rc_digits(X0)
                for c in jn_end:
                    if tx_start <= c <= tx_end:
                        idx = tx_end - c
                        if 0 <= idx <= L:
                            Y0[idx] = 2  # donor
                for c in jn_start:
                    if tx_start <= c <= tx_end:
                        idx = tx_end - c
                        if 0 <= idx <= L:
                            Y0[idx] = 1  # acceptor

            # slicing (SpliceAI style)
            num_points = ceil_div(len(Y0), SL)
            X0p = np.pad(X0, (0, SL), constant_values=0)
            Y0p = np.pad(Y0, (0, SL), constant_values=-1)

            w = writers[split]
            for i in range(num_points):
                x_block = X0p[SL * i : CL + SL * (i + 1)]   # len=SL+CL (here 512)
                y_block = Y0p[SL * i : SL * (i + 1)]        # len=SL (512)
                mask = (y_block != -1).astype(np.uint8)

                seq_bytes = map_back[x_block].tobytes()
                add_example(w, seq_bytes, y_block, mask, gene, chrom, tx_start, tx_end, strand, i)
                n_points[split] += 1

            n_rows += 1

    for w in writers.values():
        flush_shard(w)

    print("DONE")
    print("rows_processed:", n_rows)
    print("skipped_fetch:", n_skipped_fetch)
    print("skipped_single_exon:", n_skipped_single_exon)
    print("filtered_test_homologous:", n_filtered_test_hom)
    print("points:", n_points)
    print("OUT_DIR:", OUT_DIR)
    print("QUICK_TRAIN_ONLY:", QUICK_TRAIN_ONLY)

if __name__ == "__main__":
    main()

