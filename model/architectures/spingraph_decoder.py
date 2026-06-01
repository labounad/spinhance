"""
model.architectures.spingraph_decoder
======================================
Structured spin-graph model (IDEAS north-star, Families C + E + G):

    spectrum
      -> ResNet1D conv stem            (B, Cc, L) global feature tokens
      -> proj to dim + ppm positional encoding + type embedding
      (+ optional support-region tokens, fused; Family D/E — Phase 2)
      -> Transformer encoder (pre-LN)  fused token memory
      -> 8 learned spin-group queries -> Transformer decoder (cross-attn)
      -> per-node heads (shift, degeneracy) + symmetric pairwise edge head (J)
      -> ModelOutput

The unordered 8-query output + symmetric edge head directly target the S8
permutation symmetry; train with the existing Hungarian set-matching loss
(Stage-1) then the frozen surrogate spectral-consistency loss (Stage-2). The
output is the standard ``ModelOutput`` (shifts standardized, couplings edge-list
in triu order, degeneracy logits (B,G,C)) so every existing loss/metric/probe
works unchanged.

Region tokens are consumed when ``batch.region_tokens`` is present and ignored
(global-only) otherwise, so Phase-1 (global) and Phase-2 (region) share one model.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from model.architectures.base import SpinArchitecture
from model.architectures.registry import ARCHITECTURES
from model.architectures.resnet1d import SIZE_PRESETS, ResNet1DEncoder
from model.heads.node_head import NodeHead
from model.heads.pairwise_edge_head import PairwiseEdgeHead
from model.schemas import ModelOutput, SpinBatch
from model.schemas.constants import DEFAULT_DEG_VOCAB, N_GROUPS, N_POINTS


def _gaussian_kernel(sigma: float) -> torch.Tensor:
    """Normalized 1-D Gaussian smoothing kernel, radius = ceil(3*sigma)."""
    r = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return (k / k.sum()).view(1, 1, -1)


def _sinusoidal_pe(length: int, dim: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding (L, dim). Token position is
    monotonic in ppm (the conv stem preserves spectral order)."""
    pe = torch.zeros(length, dim)
    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe


@ARCHITECTURES.register("spingraph_decoder")
class SpinGraphDecoderModel(SpinArchitecture):
    def __init__(self, size: str = "medium", n_groups: int = N_GROUPS,
                 n_deg_classes: int = len(DEFAULT_DEG_VOCAB),
                 dim: int = 256, enc_layers: int = 2, dec_layers: int = 4,
                 n_heads: int = 8, ffn_mult: int = 4, dropout: float = 0.1,
                 node_hidden: int = 256, edge_hidden: int = 256,
                 region_feat_dim: int = 80, use_peak_channel: bool = False,
                 peak_min_frac: float = 0.01, peak_sigma: float = 2.0,
                 **encoder_overrides):
        super().__init__()
        self.n_groups = n_groups
        self.use_peak_channel = use_peak_channel
        self.peak_min_frac = peak_min_frac
        in_channels = 2 if use_peak_channel else 1
        if use_peak_channel:
            self.register_buffer("peak_kernel", _gaussian_kernel(peak_sigma))
        stem, stages, blocks = SIZE_PRESETS[size]
        self.encoder = ResNet1DEncoder(stem_channels=stem, stage_channels=stages,
                                       blocks_per_stage=blocks, in_channels=in_channels,
                                       **encoder_overrides)
        self.proj_global = nn.Linear(self.encoder.out_dim, dim)
        self.proj_region = nn.Linear(region_feat_dim, dim)
        self.type_embed = nn.Embedding(2, dim)               # 0 = global, 1 = region

        # positional encoding sized to the actual token-sequence length L
        with torch.no_grad():
            L = self.encoder(self._encoder_input(torch.zeros(1, N_POINTS))).shape[-1]
        self.register_buffer("ppm_pos", _sinusoidal_pe(L, dim))

        enc_layer = nn.TransformerEncoderLayer(
            dim, n_heads, dim * ffn_mult, dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, enc_layers, enable_nested_tensor=False)
        dec_layer = nn.TransformerDecoderLayer(
            dim, n_heads, dim * ffn_mult, dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(dec_layer, dec_layers)

        self.queries = nn.Parameter(torch.randn(n_groups, dim) * dim ** -0.5)
        self.node_head = NodeHead(dim, n_deg_classes, hidden=node_hidden, dropout=dropout)
        self.edge_head = PairwiseEdgeHead(dim, n_groups, hidden=edge_hidden, dropout=dropout)

    @staticmethod
    def _region_of(x):
        return x.region_tokens if isinstance(x, SpinBatch) else None

    def _peak_channel(self, spec: torch.Tensor) -> torch.Tensor:
        """Peak-emphasis channel (B, P) derived from the input spectrum: local
        maxima above a per-sample threshold, set to their height and Gaussian
        smoothed. A cheap shift-localization prior — peak positions track the
        chemical shifts the node queries must regress (Idea 1)."""
        thr = self.peak_min_frac * spec.amax(dim=-1, keepdim=True)
        is_max = torch.ones_like(spec, dtype=torch.bool)
        is_max[:, 1:] &= spec[:, 1:] >= spec[:, :-1]
        is_max[:, :-1] &= spec[:, :-1] >= spec[:, 1:]
        peaks = torch.where(is_max & (spec > thr), spec, torch.zeros_like(spec))
        k = self.peak_kernel.to(spec.dtype)
        smoothed = nn.functional.conv1d(peaks.unsqueeze(1), k,
                                        padding=k.shape[-1] // 2).squeeze(1)
        return smoothed

    def _encoder_input(self, spec: torch.Tensor) -> torch.Tensor:
        if not self.use_peak_channel:
            return spec                                             # (B, P)
        return torch.stack([spec, self._peak_channel(spec)], dim=1)  # (B, 2, P)

    def forward(self, x) -> ModelOutput:
        spec = self.spectrum_of(x)                                  # (B, P)
        B = spec.shape[0]

        feat = self.encoder(self._encoder_input(spec))             # (B, Cc, L)
        g = self.proj_global(feat.transpose(1, 2))                  # (B, L, dim)
        g = g + self.ppm_pos[: g.shape[1]].unsqueeze(0) + self.type_embed.weight[0]
        glob_pad = torch.zeros(B, g.shape[1], dtype=torch.bool, device=spec.device)

        region = self._region_of(x)
        if region is not None:
            r = self.proj_region(region.features.to(g.dtype)) + self.type_embed.weight[1]
            tokens = torch.cat([g, r], dim=1)                       # (B, L+R, dim)
            pad = torch.cat([glob_pad, ~region.mask.bool()], dim=1)  # True = PAD
        else:
            tokens, pad = g, glob_pad

        mem = self.enc(tokens, src_key_padding_mask=pad)            # (B, L(+R), dim)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)             # (B, G, dim)
        node = self.dec(q, mem, memory_key_padding_mask=pad)        # (B, G, dim)

        shifts, deg_logits = self.node_head(node)                  # (B,G), (B,G,C)
        jmag, jpres, se_logits, edge_emb = self.edge_head(node)    # (B,E)x3, (B,E,h)
        return ModelOutput(
            shifts=shifts,
            coupling_values=jmag,                                  # (B,E) -> matrix auto
            coupling_presence_logits=jpres,                        # (B,E)
            degeneracy_logits=deg_logits,                          # (B,G,C)
            node_embeddings=node,
            edge_embeddings=edge_emb,
            auxiliary={"soft_equiv_logits": se_logits},            # (B,E) triu order
        )
