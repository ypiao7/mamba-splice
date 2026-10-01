import math
from typing import Tuple, Dict, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


# =========================================================
# RoPE
# =========================================================
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        assert dim % 2 == 0
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int, device=None, dtype=None):
        device = device or self.inv_freq.device
        t = torch.arange(seqlen, device=device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (L, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)            # (L, dim)
        cos = emb.cos()
        sin = emb.sin()
        if dtype is not None:
            cos = cos.to(dtype)
            sin = sin.to(dtype)
        return cos, sin


# =========================================================
# Norm
# =========================================================
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


# =========================================================
# Local attention mask
# =========================================================
def build_local_attn_mask(L: int, window: int, device, dtype):
    half = window // 2
    idx = torch.arange(L, device=device)
    dist = (idx[:, None] - idx[None, :]).abs()
    allowed = dist <= half
    mask = torch.full((L, L), float("-inf"), device=device, dtype=dtype)
    mask.masked_fill_(allowed, 0.0)
    return mask


# =========================================================
# Local Self-Attention (RoPE) + KV export
# =========================================================
class LocalSelfAttentionKV(nn.Module):
    def __init__(self, d_model: int, n_heads: int, window: int, dropout: float = 0.0, use_rope: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window = window
        self.use_rope = use_rope

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.d_head) if use_rope else None

        self._mask_cache: Dict[Tuple[torch.device, torch.dtype, int, int], torch.Tensor] = {}

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,L,Dh)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        if self.use_rope:
            cos, sin = self.rope(L, device=x.device, dtype=x.dtype)
            q, k = apply_rope(q, k, cos, sin)

        key = (x.device, x.dtype, L, self.window)
        if key not in self._mask_cache:
            self._mask_cache[key] = build_local_attn_mask(L, self.window, x.device, x.dtype)
        attn_mask = self._mask_cache[key]

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        y = self.out(y)
        y = self.drop(y)

        K_base = k.transpose(1, 2).contiguous().view(B, L, D)
        V_base = v.transpose(1, 2).contiguous().view(B, L, D)
        return y, (K_base, V_base)


# =========================================================
# BiMamba2 (Genome-friendly, bidirectional)
# =========================================================
def _try_import_mamba2():
    try:
        from mamba_ssm import Mamba2
        return Mamba2
    except Exception:
        return None

class BiMamba2(nn.Module):
    """
    bidirectional=True  (기본): y = mix([Mamba2_fwd(x), reverse(Mamba2_bwd(reverse(x)))])
    bidirectional=False (ablation용): y = Mamba2_fwd(x) 만 사용 (단방향)
    """
    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, expand: int = 2,
                 dropout: float = 0.1, bidirectional: bool = True):
        super().__init__()
        Mamba2 = _try_import_mamba2()
        if Mamba2 is None:
            raise ImportError("Mamba2 not available. pip install mamba-ssm")
        self.bidirectional = bool(bidirectional)
        self.fwd = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        if self.bidirectional:
            self.bwd = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.mix = nn.Linear(2 * d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,L,D)
        y_f = self.fwd(x)
        if not self.bidirectional:
            return self.drop(y_f)
        y_b = self.bwd(torch.flip(x, dims=[1]))
        y_b = torch.flip(y_b, dims=[1])
        y = self.mix(torch.cat([y_f, y_b], dim=-1))
        return self.drop(y)


# =========================================================
# Memory Converter (prototype)
# =========================================================
class MemoryConverter(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(2 * d_model, d_model, bias=False)
        self.to_inject = nn.Linear(d_model, d_model, bias=False)

    def forward(self, K: torch.Tensor, V: torch.Tensor):
        s = self.proj(torch.cat([K.mean(dim=1), V.mean(dim=1)], dim=-1))
        inj = self.to_inject(s).unsqueeze(1)
        return inj


# =========================================================
# CNN motif stem
# =========================================================
class ResidualConvBlock(nn.Module):
    def __init__(self, d: int, kernel: int = 7, dropout: float = 0.1):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(d, d, kernel_size=kernel, padding=pad)
        self.conv2 = nn.Conv1d(d, d, kernel_size=kernel, padding=pad)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(d)

    def forward(self, x):  # (B,D,L)
        r = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = x + r
        x = self.bn(x)
        return x


# =========================================================
# Encoders
# =========================================================
_LUT_NP = np.full(256, 0, dtype=np.int64)  # default N=0
for ch, v in [("A", 1), ("C", 2), ("G", 3), ("T", 4), ("N", 0),
              ("a", 1), ("c", 2), ("g", 3), ("t", 4), ("n", 0)]:
    _LUT_NP[ord(ch)] = v

class OneHotEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(5, d_model, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, seq_bytes_list, device):
        # seq_bytes_list: list[bytes], each length L
        # -> (B,L,D)

        # (B,L) int64 on CPU (fast)
        arr = np.stack(
            [_LUT_NP[np.frombuffer(s, dtype=np.uint8)] for s in seq_bytes_list],
            axis=0
        )

        x = torch.from_numpy(arr).to(device=device, dtype=torch.long)   # (B,L) on GPU
        x = F.one_hot(x, num_classes=5).to(torch.float32)               # (B,L,5)
        x = x.permute(0, 2, 1).contiguous()                             # (B,5,L)

        h = self.stem(x)                                                # (B,D,L)
        h = h.permute(0, 2, 1).contiguous()                             # (B,L,D)
        return h


class BPETwitterToBaseEncoder(nn.Module):
    def __init__(self, d_model: int, tokenizer_name_or_path: str, dropout: float = 0.1):
        super().__init__()
        if AutoTokenizer is None:
            raise ImportError("transformers is required for BPE mode. Install transformers.")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, trust_remote_code=True, use_fast=True
        )
        vocab = int(getattr(self.tokenizer, "vocab_size", 0)) or int(self.tokenizer.vocab_size)
        self.emb = nn.Embedding(vocab, d_model)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    @torch.no_grad()
    def _tokens_to_base(self, h_tok: torch.Tensor, offsets: torch.Tensor, L: int):
        B, T, D = h_tok.shape
        device = h_tok.device
        h_base = torch.zeros((B, L, D), device=device, dtype=h_tok.dtype)
        filled = torch.zeros((B, L), device=device, dtype=torch.bool)

        for b in range(B):
            for t in range(T):
                s = int(offsets[b, t, 0].item())
                e = int(offsets[b, t, 1].item())
                if e <= s:
                    continue
                s = max(0, min(s, L))
                e = max(0, min(e, L))
                if e <= s:
                    continue
                h_base[b, s:e, :] = h_tok[b, t, :].unsqueeze(0)
                filled[b, s:e] = True

            if (~filled[b]).any():
                last = torch.zeros((D,), device=device, dtype=h_tok.dtype)
                for i in range(L):
                    if filled[b, i]:
                        last = h_base[b, i]
                    else:
                        h_base[b, i] = last
        return h_base

    def forward(self, dna_list, device):
        enc = self.tokenizer(
            dna_list,
            return_tensors="pt",
            padding=True,
            truncation=False,
            return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping")
        input_ids = enc["input_ids"].to(device)
        offsets = offsets.to(device)

        h_tok = self.emb(input_ids)
        h_tok = self.drop(h_tok)
        h_tok = self.proj(h_tok)

        L = len(dna_list[0])
        h_base = self._tokens_to_base(h_tok, offsets, L)
        return h_base


# =========================================================
# Bio-TransMamba Prox (BiMamba2)
# =========================================================
class BioTransMambaProxBiMamba2(nn.Module):
    """
    - Mamba stream: BiMamba2 on full length
    - Local stream: attention on [crop, crop+SL) (+margin)
    - Memory inject: local KV -> inj added to mamba stream (broadcast)
    - Combine: scalar gated replacement on local zone

    두 실험(hg38 3-class splice site / GTEx tissue-specific)이 이 모델 하나를
    공유합니다. `use_tissue_head=False`(기본값)면 기존 hg38 실험과 완전히
    동일하게 동작하며 forward()는 logits 하나만 반환합니다.
    `use_tissue_head=True`로 켜면 tissue head가 추가되고 forward()는
    (logits, logits_tissue) 튜플을 반환합니다.

    Ablation용 플래그 (기본값은 전부 "논문 최종안"과 동일하게 켜져 있음):
      - use_local_attention: False면 (B)로컬 어텐션/(C)memory injection/(D)gate
        전부 제거하고 순수 BiMamba2 스트림만 사용 (파라미터 수도 그만큼 줄어듦)
      - enable_memory_inject: False면 (C)만 제거. use_local_attention=False면
        자동으로 함께 꺼짐 (K,V 자체가 없으므로).
      - gate_mode: "learned"(기본, 위치별 학습 게이트) / "fixed"(고정값
        fixed_gate_value로 mamba·attn을 항상 같은 비율로 섞음) / "hard"
        (crop 구간을 attn 출력으로 완전히 대체, gate=1 고정)
      - bidirectional_mamba: False면 BiMamba2를 정방향만 사용 (역방향 스트림 제거)
      - use_conv_stem: False면 앞단 CNN motif stem(ResidualConvBlock x4)을 제거
        하고 encoder 출력을 그대로 Hybrid stack에 흘림
    """
    def __init__(
        self,
        input_len: int,
        sl: int,
        crop: int,
        d_model: int = 256,
        n_layers: int = 12,
        n_heads: int = 8,
        attn_window: int = 128,
        dropout: float = 0.1,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        local_margin: int = 0,
        enable_memory_inject: bool = True,
        use_bpe: bool = False,
        bpe_tokenizer: str = "zhihan1996/DNABERT-2-117M",
        num_classes: int = 3,
        use_tissue_head: bool = False,
        num_tissues: int = 15,
        # ---- ablation 플래그 ----
        use_local_attention: bool = True,
        gate_mode: str = "learned",
        fixed_gate_value: float = 0.5,
        bidirectional_mamba: bool = True,
        use_conv_stem: bool = True,
    ):
        super().__init__()
        assert gate_mode in ("learned", "fixed", "hard"), f"unknown gate_mode: {gate_mode}"

        self.input_len = int(input_len)
        self.sl = int(sl)
        self.crop = int(crop)
        self.local_margin = int(local_margin)
        self.use_bpe = bool(use_bpe)
        self.use_tissue_head = bool(use_tissue_head)

        self.use_local_attention = bool(use_local_attention)
        self.gate_mode = gate_mode
        self.fixed_gate_value = float(fixed_gate_value)
        self.use_conv_stem = bool(use_conv_stem)

        # local attention이 없으면 memory injection도 물리적으로 불가능 (K,V가 없음)
        self.enable_memory_inject = bool(enable_memory_inject) and self.use_local_attention

        if self.use_bpe:
            self.encoder = BPETwitterToBaseEncoder(d_model=d_model, tokenizer_name_or_path=bpe_tokenizer, dropout=dropout)
        else:
            self.encoder = OneHotEncoder(d_model=d_model, dropout=dropout)

        if self.use_conv_stem:
            self.conv_blocks = nn.ModuleList([ResidualConvBlock(d_model, kernel=7, dropout=dropout) for _ in range(4)])
        else:
            self.conv_blocks = None

        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            layer = {
                "norm": RMSNorm(d_model),
                "mamba": BiMamba2(d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                                   dropout=dropout, bidirectional=bidirectional_mamba),
                "ffn_norm": RMSNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(4 * d_model, d_model),
                    nn.Dropout(dropout),
                ),
                "drop": nn.Dropout(dropout),
            }
            if self.use_local_attention:
                layer["attn"] = LocalSelfAttentionKV(d_model, n_heads=n_heads, window=attn_window, dropout=dropout, use_rope=True)
                if self.enable_memory_inject:
                    layer["mem"] = MemoryConverter(d_model)
                if self.gate_mode == "learned":
                    layer["gate"] = nn.Linear(d_model, 1)
            self.layers.append(nn.ModuleDict(layer))

        self.head_norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        # tissue-specific 실험(GTEx)에서만 사용
        self.tissue_head: Optional[nn.Linear] = (
            nn.Linear(d_model, num_tissues) if self.use_tissue_head else None
        )

    def _local_range(self) -> Tuple[int, int]:
        lo = max(0, self.crop - self.local_margin)
        hi = min(self.input_len, self.crop + self.sl + self.local_margin)
        return lo, hi

    def forward(self, seq_bytes_list=None, dna_list=None, device=None):
        assert device is not None

        # 1) base embedding (B,L,D)
        if self.use_bpe:
            assert dna_list is not None
            x = self.encoder(dna_list, device=device)
        else:
            assert seq_bytes_list is not None
            x = self.encoder(seq_bytes_list, device=device)

        # 2) CNN motif stage (ablation: use_conv_stem=False면 스킵)
        if self.use_conv_stem:
            x = x.permute(0, 2, 1).contiguous()  # (B,D,L)
            for blk in self.conv_blocks:
                x = blk(x)
            x = x.permute(0, 2, 1).contiguous()  # (B,L,D)

        lo, hi = self._local_range()

        # 3) Hybrid stack
        for layer in self.layers:
            h = layer["norm"](x)

            # (A) full-length BiMamba2
            out_mamba = layer["mamba"](h)  # (B,L,D)

            if self.use_local_attention:
                # (B) local attention
                h_loc = h[:, lo:hi, :]
                out_loc, (K, V) = layer["attn"](h_loc)

                out_attn_full = torch.zeros_like(out_mamba)
                out_attn_full[:, lo:hi, :] = out_loc

                # (C) memory injection
                if self.enable_memory_inject:
                    inj = layer["mem"](K, V)     # (B,1,D)
                    out_mamba = out_mamba + inj  # broadcast

                # (D) gated replacement on local
                if self.gate_mode == "learned":
                    g = torch.sigmoid(layer["gate"](h))  # (B,L,1)
                elif self.gate_mode == "fixed":
                    g = torch.full((x.size(0), x.size(1), 1), self.fixed_gate_value, device=x.device, dtype=x.dtype)
                else:  # "hard"
                    g = torch.ones((x.size(0), x.size(1), 1), device=x.device, dtype=x.dtype)

                x_new = out_mamba
                x_new_loc = out_mamba[:, lo:hi, :] + g[:, lo:hi, :] * (out_attn_full[:, lo:hi, :] - out_mamba[:, lo:hi, :])
                x_new = x_new.clone()
                x_new[:, lo:hi, :] = x_new_loc
            else:
                # 순수 BiMamba2 스트림만 사용 (local attention 완전 제거 ablation)
                x_new = out_mamba

            x = x + layer["drop"](x_new)
            x = x + layer["ffn"](layer["ffn_norm"](x))

        x = self.head_norm(x)
        logits = self.head(x)

        if self.use_tissue_head:
            logits_tis = self.tissue_head(x)
            return logits, logits_tis

        return logits