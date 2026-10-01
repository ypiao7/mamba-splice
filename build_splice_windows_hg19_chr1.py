import os
import numpy as np
import pandas as pd
from pyfaidx import Fasta

# ===== 설정 =====
ANNOT_PATH = "/home/yjpiao/workspace/data/canonical_dataset.txt"  # hg19 기준
FASTA_PATH = "/home/yjpiao/data/genome/hg19.fa"
OUTPUT_TSV = "/home/yjpiao/workspace/data/splice_windows_hg19_chr1.tsv"

# 사용할 염색체 (빠른 테스트용으로 chr1만)
TARGET_CHROMS = {"1"}   # canonical_dataset에서 chr 이름이 '1', '2', ... 형식
WINDOW_RADIUS = 500     # 중심에서 ±500bp → 총 1001bp 윈도우

# ===== 헬퍼 함수 =====
def parse_exon_list(s: str):
    # "925737,925921,...," 이런 형태 → [925737, 925921, ...]
    s = s.strip()
    if s.endswith(","):
        s = s[:-1]
    if not s:
        return []
    return [int(x) for x in s.split(",")]

def get_splice_sites_from_line(fields):
    """
    canonical_dataset 한 줄을 받아 canonical donor/acceptor 좌표를 구한다.
    포맷: gene, chrom, strand, tx_start, tx_end, exonStarts, exonEnds
    """
    gene = fields[0]
    chrom = fields[1]
    strand = fields[2]
    tx_start = int(fields[3])
    tx_end = int(fields[4])
    exon_starts = parse_exon_list(fields[5])
    exon_ends = parse_exon_list(fields[6])

    if len(exon_starts) != len(exon_ends) or len(exon_starts) < 2:
        return []  # 스플라이스 없는 단일 엑손 유전자 등은 스킵

    sites = []

    # SpliceAI 방식에 맞게:
    # + strand: donor = exonEnd[:-1], acceptor = exonStart[1:]
    # - strand: donor = exonStart[1:], acceptor = exonEnd[:-1]
    if strand == "+":
        donors = exon_ends[:-1]
        acceptors = exon_starts[1:]
    else:
        donors = exon_starts[1:]
        acceptors = exon_ends[:-1]

    for pos in donors:
        sites.append((chrom, pos, strand, "donor", gene))
    for pos in acceptors:
        sites.append((chrom, pos, strand, "acceptor", gene))

    return sites

# ===== 1. canonical_dataset에서 chr1의 splice site 좌표 수집 =====
all_sites = []

with open(ANNOT_PATH) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 7:
            continue
        chrom = fields[1]
        if chrom not in TARGET_CHROMS:
            continue

        sites = get_splice_sites_from_line(fields)
        all_sites.extend(sites)

print(f"Collected {len(all_sites)} splice sites (donor+acceptor) on chroms {TARGET_CHROMS}")

# 중복 제거
# key: (chrom, pos, strand, label)
unique_sites = {}
for chrom, pos, strand, label, gene in all_sites:
    key = (chrom, pos, strand, label)
    if key not in unique_sites:
        unique_sites[key] = gene

sites_list = [
    (chrom, pos, strand, label, gene)
    for (chrom, pos, strand, label), gene in unique_sites.items()
]
print(f"Unique splice sites: {len(sites_list)}")

# ===== 2. FASTA에서 서열 추출 =====
genome = Fasta(FASTA_PATH)  # UCSC hg19: chr1, chr2, ...
chrom_lengths = {name: len(genome[name]) for name in genome.keys()}

def to_fasta_chrom(chrom_str: str) -> str:
    # canonical_dataset는 "1", "2" 형식. hg19.fa는 "chr1", "chr2".
    if chrom_str.startswith("chr"):
        return chrom_str
    return f"chr{chrom_str}"

def extract_window(chrom, pos, radius):
    """
    chrom: '1' 같은 canonical 이름
    pos:   0-based 기준 스플라이스 위치라고 가정
    """
    fasta_chrom = to_fasta_chrom(chrom)
    chr_len = chrom_lengths[fasta_chrom]

    start = max(0, pos - radius)
    end = min(chr_len, pos + radius + 1)  # end는 exclusive

    seq = genome[fasta_chrom][start:end].seq.upper()

    # 만약 윈도우가 양 끝에 걸려서 길이가 짧으면 패딩 대신 스킵
    expected_len = 2 * radius + 1
    if len(seq) != expected_len:
        return None
    return seq

# 양성(도너/억셉터) 윈도우 생성
pos_records = []  # (chrom, pos, strand, label_id, label_name, seq)

label_to_id = {"none": 0, "acceptor": 1, "donor": 2}

for chrom, pos, strand, label, gene in sites_list:
    seq = extract_window(chrom, pos, WINDOW_RADIUS)
    if seq is None:
        continue
    pos_records.append(
        (chrom, pos, strand, label_to_id[label], label, seq)
    )

print(f"Positive (donor/acceptor) windows: {len(pos_records)}")

# ===== 3. 음성(non-splice) 샘플 생성 =====
# chr1에서 랜덤 위치를 뽑되, splice site 주변 WINDOW_RADIUS bp 이내는 피한다.
import random

# splice site 주변 금지 영역 집합 만들기
blocked_positions = set()
for chrom, pos, strand, label, gene in sites_list:
    if chrom not in TARGET_CHROMS:
        continue
    for p in range(pos - WINDOW_RADIUS, pos + WINDOW_RADIUS + 1):
        if p >= 0:
            blocked_positions.add(p)

target_chrom = list(TARGET_CHROMS)[0]
fasta_chrom = to_fasta_chrom(target_chrom)
chr_len = chrom_lengths[fasta_chrom]

num_pos = len(pos_records)
num_neg_target = num_pos  # 양성 개수만큼 음성 생성 (balanced)

neg_records = []
attempts = 0
max_attempts = num_neg_target * 20  # 너무 오래 돌지 않도록 제한

while len(neg_records) < num_neg_target and attempts < max_attempts:
    attempts += 1
    pos = random.randint(WINDOW_RADIUS, chr_len - WINDOW_RADIUS - 1)
    if pos in blocked_positions:
        continue
    seq = extract_window(target_chrom, pos, WINDOW_RADIUS)
    if seq is None:
        continue
    # strand는 일단 '+'로 둔다 (NT는 서열만 보므로 여기선 크게 상관 없음)
    neg_records.append(
        (target_chrom, pos, "+", label_to_id["none"], "none", seq)
    )

print(f"Negative (none) windows: {len(neg_records)} (attempts={attempts})")

# ===== 4. TSV로 저장 =====
all_records = pos_records + neg_records
df = pd.DataFrame(
    all_records,
    columns=["chrom", "pos", "strand", "label_id", "label_name", "seq"],
)

print("Final dataset size:", df.shape)
os.makedirs(os.path.dirname(OUTPUT_TSV), exist_ok=True)
df.to_csv(OUTPUT_TSV, sep="\t", index=False)
print("Saved to:", OUTPUT_TSV)
