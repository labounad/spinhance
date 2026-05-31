"""
modelv2.model — neural network
==============================
Two classes, nothing else (DESIGN.md):

  * ``ResNet1D``        — a 1-D residual encoder over the raw spectrum.
  * ``SpinHanceModel``  — that encoder + four typed heads.

Input  : (B, P) normalized spectrum (P = 16384), treated as a 1-channel signal.
Output : a dict of raw head outputs in the dataset's STANDARDIZED space
         (shifts & j_mag are z-scored; presence & degeneracy are logits):

        shifts        (B, G)                  z-scored shift mean (mu)
        shift_logvar  (B, G)                  predicted log-variance of shift
        j_mag         (B, n_pairs)            z-scored |J| mean (mu)
        jmag_logvar   (B, n_pairs)            predicted log-variance of |J|
        j_presence    (B, n_pairs)            coupling-presence logits
        deg_logits    (B, G, C)               degeneracy class logits

The two regression heads emit a log-variance channel alongside the mean (mu and
log sigma^2). It is used first as a diagnostic and only later, with care, as a
beta-NLL term (see train.py). log sigma^2 is produced in fp32 and clamped so
``exp`` can never overflow.

GroupNorm is used throughout (never BatchNorm): it is independent of batch
composition and batch size, and has no running buffers, so the EMA in train.py
tracks parameters only. The group count is asserted to divide every channel
width at build time.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["ResNet1D", "SpinHanceModel", "LOGVAR_MIN", "LOGVAR_MAX"]

# Clamp range for predicted log-variance (keeps exp(logvar) finite in fp32).
LOGVAR_MIN, LOGVAR_MAX = -10.0, 5.0


def _conv(in_c, out_c, k=3, s=1):
    return nn.Conv1d(in_c, out_c, k, stride=s, padding=k // 2, bias=False)


def _group_norm(channels, max_groups=32):
    """GroupNorm with the largest group count (<= max_groups) dividing channels.

    Asserts divisibility so a misconfigured width fails loudly at build time
    rather than silently changing normalization semantics.
    """
    g = min(max_groups, channels)
    while channels % g:
        g -= 1
    assert channels % g == 0, f"no GroupNorm group count divides {channels}"
    return nn.GroupNorm(g, channels)


class BasicBlock1D(nn.Module):
    """Conv-GN-ReLU-Conv-GN with a residual connection (stride on the first conv)."""

    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = _conv(in_c, out_c, 3, stride)
        self.norm1 = _group_norm(out_c)
        self.conv2 = _conv(out_c, out_c, 3, 1)
        self.norm2 = _group_norm(out_c)
        self.act = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or in_c != out_c:
            self.down = nn.Sequential(
                nn.Conv1d(in_c, out_c, 1, stride=stride, bias=False),
                _group_norm(out_c))

    def forward(self, x):
        idn = x if self.down is None else self.down(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + idn)


class ResNet1D(nn.Module):
    """1-D residual encoder: large-kernel stem, four stride-2 stages, global pool."""

    def __init__(self, stem_channels=32, stage_channels=(64, 128, 256, 512),
                 blocks_per_stage=(2, 2, 2, 2), stem_kernel=15, stem_stride=4):
        super().__init__()
        assert len(stage_channels) == len(blocks_per_stage) == 4, \
            "ResNet1D expects exactly four stages"
        self.stem = nn.Sequential(
            nn.Conv1d(1, stem_channels, stem_kernel, stride=stem_stride,
                      padding=stem_kernel // 2, bias=False),
            _group_norm(stem_channels), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1))
        stages = []
        in_c = stem_channels
        for out_c, n in zip(stage_channels, blocks_per_stage):
            stages.append(BasicBlock1D(in_c, out_c, stride=2))  # downsample first
            for _ in range(n - 1):
                stages.append(BasicBlock1D(out_c, out_c, stride=1))
            in_c = out_c
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = in_c

    def forward(self, x):                  # x: (B, P)
        x = x.unsqueeze(1)                 # (B, 1, P)
        x = self.stem(x)
        x = self.stages(x)
        return self.pool(x).squeeze(-1)    # (B, C)


class SpinHanceModel(nn.Module):
    """ResNet-1D encoder + four typed heads (shift / j-mag / presence / degeneracy)."""

    def __init__(self, n_groups=8, n_deg_classes=8, encoder=None,
                 head_hidden=512, dropout=0.1, encoder_kwargs=None):
        super().__init__()
        self.n_groups = int(n_groups)
        self.n_pairs = self.n_groups * (self.n_groups - 1) // 2
        self.n_deg = int(n_deg_classes)
        self.head_hidden = int(head_hidden)
        self.dropout = float(dropout)
        self.encoder_kwargs = dict(encoder_kwargs or {})
        self.encoder = encoder or ResNet1D(**self.encoder_kwargs)
        emb = self.encoder.out_dim

        def head(out):
            return nn.Sequential(
                nn.Linear(emb, head_hidden), nn.ReLU(inplace=True),
                nn.Dropout(dropout), nn.Linear(head_hidden, out))

        # Regression heads emit 2 channels per cell: mean and log-variance.
        self.shift_head = head(2 * self.n_groups)
        self.jmag_head = head(2 * self.n_pairs)
        self.jpres_head = head(self.n_pairs)
        self.deg_head = head(self.n_groups * self.n_deg)

    def build_kwargs(self):
        """Everything needed to reconstruct this module (saved in the checkpoint)."""
        return dict(n_groups=self.n_groups, n_deg_classes=self.n_deg,
                    head_hidden=self.head_hidden, dropout=self.dropout,
                    encoder_kwargs=self.encoder_kwargs)

    def forward(self, spectrum):           # (B, P)
        z = self.encoder(spectrum)
        B = spectrum.shape[0]

        shift = self.shift_head(z).view(B, self.n_groups, 2)
        jmag = self.jmag_head(z).view(B, self.n_pairs, 2)

        def clamp_logvar(t):
            return t.float().clamp(LOGVAR_MIN, LOGVAR_MAX)

        return {
            "shifts": shift[..., 0],                       # (B, G)
            "shift_logvar": clamp_logvar(shift[..., 1]),   # (B, G)
            "j_mag": jmag[..., 0],                          # (B, n_pairs)
            "jmag_logvar": clamp_logvar(jmag[..., 1]),      # (B, n_pairs)
            "j_presence": self.jpres_head(z),               # (B, n_pairs) logits
            "deg_logits": self.deg_head(z).view(B, self.n_groups, self.n_deg),
        }


def build_model(cfg=None, *, n_groups=8, n_deg_classes=8, small=False):
    """Convenience constructor used by train.py / gui.py.

    ``small=True`` builds a lightweight encoder for smoke tests; otherwise the
    full-size encoder from DESIGN.md is used.
    """
    enc_kwargs = (dict(stem_channels=16, stage_channels=(16, 32, 64, 64),
                       blocks_per_stage=(1, 1, 1, 1), stem_kernel=9, stem_stride=4)
                  if small else {})
    head_hidden = 128 if small else 512
    return SpinHanceModel(n_groups=n_groups, n_deg_classes=n_deg_classes,
                          head_hidden=head_hidden, encoder_kwargs=enc_kwargs)


if __name__ == "__main__":            # shape smoke test (needs torch)
    m = SpinHanceModel()
    out = m(torch.randn(4, 16384))
    print({k: tuple(v.shape) for k, v in out.items()})
    print("params (M):", round(sum(p.numel() for p in m.parameters()) / 1e6, 2))
