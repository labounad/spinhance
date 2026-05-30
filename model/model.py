"""
model.model
==============
ResNet-1D encoder + four typed heads (Decision 1 & 2).

Input : (B, P) normalized spectrum (P = 16384), treated as a 1-channel signal.
Output: dict of raw head outputs, in the STANDARDIZED space the dataset produces
        (shifts & j_mag are z-scored; presence & degeneracy are logits).

        shifts      (B, G)
        j_mag       (B, n_pairs)             n_pairs = G*(G-1)/2
        j_presence  (B, n_pairs)             logits
        deg_logits  (B, G, n_deg_classes)    logits

Encoder is swappable behind this interface (Decision 1 notes a transformer/hybrid
as a possible v2).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv(in_c, out_c, k=3, s=1):
    return nn.Conv1d(in_c, out_c, k, stride=s, padding=k // 2, bias=False)


class BasicBlock1D(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = _conv(in_c, out_c, 3, stride)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = _conv(out_c, out_c, 3, 1)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.act = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or in_c != out_c:
            self.down = nn.Sequential(
                nn.Conv1d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_c))

    def forward(self, x):
        idn = x if self.down is None else self.down(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + idn)


class ResNet1DEncoder(nn.Module):
    def __init__(self, stem_channels=32, stage_channels=(64, 128, 256, 512),
                 blocks_per_stage=(2, 2, 2, 2), stem_kernel=15, stem_stride=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, stem_channels, stem_kernel, stride=stem_stride,
                      padding=stem_kernel // 2, bias=False),
            nn.BatchNorm1d(stem_channels), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1))
        stages = []
        in_c = stem_channels
        for out_c, n in zip(stage_channels, blocks_per_stage):
            stages.append(BasicBlock1D(in_c, out_c, stride=2))
            for _ in range(n - 1):
                stages.append(BasicBlock1D(out_c, out_c, stride=1))
            in_c = out_c
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = in_c

    def forward(self, x):                      # x: (B, P)
        x = x.unsqueeze(1)                     # (B, 1, P)
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x).squeeze(-1)           # (B, C)
        return x


class SpinHanceModel(nn.Module):
    def __init__(self, n_groups=8, n_deg_classes=8, encoder=None,
                 head_hidden=512, dropout=0.1):
        super().__init__()
        self.G = n_groups
        self.n_pairs = n_groups * (n_groups - 1) // 2
        self.n_deg = n_deg_classes
        self.encoder = encoder or ResNet1DEncoder()
        emb = self.encoder.out_dim

        def head(out):
            return nn.Sequential(nn.Linear(emb, head_hidden), nn.ReLU(inplace=True),
                                 nn.Dropout(dropout), nn.Linear(head_hidden, out))

        self.shift_head = head(self.G)
        self.jmag_head = head(self.n_pairs)
        self.jpres_head = head(self.n_pairs)
        self.deg_head = head(self.G * self.n_deg)

    def forward(self, spectrum):               # (B, P)
        z = self.encoder(spectrum)
        B = spectrum.shape[0]
        return {
            "shifts": self.shift_head(z),                         # (B, G)
            "j_mag": self.jmag_head(z),                           # (B, n_pairs)
            "j_presence": self.jpres_head(z),                     # (B, n_pairs) logits
            "deg_logits": self.deg_head(z).view(B, self.G, self.n_deg),
        }


if __name__ == "__main__":   # shape smoke test (run in your env)
    m = SpinHanceModel()
    x = torch.randn(4, 16384)
    out = m(x)
    print({k: tuple(v.shape) for k, v in out.items()})
    print("params (M):", round(sum(p.numel() for p in m.parameters()) / 1e6, 2))
