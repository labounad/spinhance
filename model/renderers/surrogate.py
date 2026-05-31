"""
model.renderers.surrogate
=========================
Differentiable neural matrix -> spectrum renderer (Branch 5). A cheap, bounded,
fully-differentiable stand-in for the exact quantum renderer, trained to imitate
it so it can serve as a spectral-consistency loss for the spectrum->matrix model
(Branch 6) without the exact engine's eigendecomposition cost.

Design (chosen with the user):
  * Output = sticks -> analytic broadening. The network predicts a line list
    (centers, amps); the spectrum is produced by the SAME differentiable Lorentzian
    broadening the exact renderer uses (``_broaden_fft_batch``). Peaks are sharp
    for free; the net only learns the hard part (where lines are + how strong),
    i.e. the second-order coupling the eigendecomposition captures.
  * Encoder = edge-aware set transformer over the 8 spin-group tokens. Couplings
    bias attention (keyed on J and J/Δν), so strongly-coupled groups exchange
    information — that cross-group intensity borrowing IS second-order coupling.
  * Field-conditioned: field strength feeds the tokens and the coupling bias
    (J/Δν, the second-order parameter, scales with field) and the broadening width,
    so one model renders both 90 and 600 MHz.

Physical inductive biases baked in:
  * each stick sits near its group's shift (offset predicted in Hz — multiplet
    splittings are field-independent in Hz — then converted to ppm via /field);
  * per-group stick areas sum to the group's degeneracy (integration cue).

Forward signature (field is a scalar per call so the shared-kernel broadening is
valid; train one field per batch):

    spectrum = SurrogateRenderer()(shifts, couplings, degeneracy, field_mhz)
    shifts (B,G) ppm · couplings (B,G,G) Hz · degeneracy (B,G) protons · field float
    -> (B, points), each normalized to unit integral.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.renderers._torch_exact import _broaden_fft_batch
from model.renderers.registry import RENDERERS
from model.schemas.constants import N_GROUPS, N_POINTS, PPM_FROM, PPM_TO


class _EdgeAwareLayer(nn.Module):
    """One transformer block whose attention logits get an additive per-pair bias
    derived from the couplings (via an external (B,H,G,G) bias tensor)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, dim))

    def forward(self, x, attn_bias):                 # x (B,G,D); attn_bias (B*H,G,G)
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_bias, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


@RENDERERS.register("surrogate")
class SurrogateRenderer(nn.Module):
    def __init__(self, n_groups: int = N_GROUPS, dim: int = 128, depth: int = 4,
                 heads: int = 4, sticks_per_group: int = 48, mlp_ratio: float = 2.0,
                 dropout: float = 0.0, offset_max_hz: float = 60.0,
                 points: int = N_POINTS, ppm_from: float = PPM_FROM, ppm_to: float = PPM_TO,
                 linewidth_hz: float = 1.0):
        super().__init__()
        self.G = n_groups
        self.heads = heads
        self.m = sticks_per_group
        self.offset_max_hz = offset_max_hz
        self.points = points
        self.ppm_from, self.ppm_to = ppm_from, ppm_to
        self.linewidth_hz = linewidth_hz

        # token features: [shift/12, deg/9, total|J|/100, log10(field)/3]
        self.token_in = nn.Linear(4, dim)
        # coupling-bias MLP: [J, dnu(kHz), J/dnu, log10(field)/3] -> per-head bias
        self.bias_mlp = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, heads))
        self.layers = nn.ModuleList(
            [_EdgeAwareLayer(dim, heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        # per-group stick decoder: -> (offset_raw, amp_logit) x m
        self.dec = nn.Linear(dim, self.m * 2)

    # ── helpers ────────────────────────────────────────────────────────────────

    def _attn_bias(self, shifts, couplings, field):           # -> (B*H, G, G)
        B, G = shifts.shape
        dnu = (shifts[:, :, None] - shifts[:, None, :]).abs() * field        # (B,G,G) Hz
        J = couplings.abs()
        ratio = J / (dnu + 1.0)                                              # second-order param
        lf = torch.log10(torch.tensor(float(field), device=shifts.device)) / 3.0
        lf = lf.expand(B, G, G)
        feats = torch.stack([J, dnu / 1000.0, ratio, lf], dim=-1)            # (B,G,G,4)
        bias = self.bias_mlp(feats)                                          # (B,G,G,H)
        return bias.permute(0, 3, 1, 2).reshape(B * self.heads, G, G)

    # ── forward ──────────────────────────────────────────────────────────────--

    def forward(self, shifts, couplings, degeneracy, field_mhz: float) -> torch.Tensor:
        B, G = shifts.shape
        field = float(field_mhz)
        device, dtype = shifts.device, shifts.dtype
        deg = degeneracy.to(dtype)

        lf = torch.log10(torch.tensor(field, device=device, dtype=dtype)) / 3.0
        tot_j = couplings.abs().sum(-1)                                      # (B,G)
        tok = torch.stack([shifts / 12.0, deg / 9.0, tot_j / 100.0,
                           lf.expand(B, G)], dim=-1)                         # (B,G,4)
        x = self.token_in(tok)
        bias = self._attn_bias(shifts, couplings, field)
        for layer in self.layers:
            x = layer(x, bias)
        x = self.norm(x)

        dec = self.dec(x).reshape(B, G, self.m, 2)
        offsets_hz = torch.tanh(dec[..., 0]) * self.offset_max_hz           # (B,G,m)
        amps = F.softmax(dec[..., 1], dim=-1) * deg[:, :, None]             # per-group area = deg
        centers = shifts[:, :, None] + offsets_hz / field                   # ppm (B,G,m)

        centers = centers.reshape(B, G * self.m)
        amps = amps.reshape(B, G * self.m)

        dx = (self.ppm_to - self.ppm_from) / self.points
        hwhm = (self.linewidth_hz / 2.0) / field
        spec = _broaden_fft_batch(centers, amps, self.points, self.ppm_from,
                                  self.ppm_to, dx, hwhm, device, dtype)      # (B, points)
        spec = spec.clamp_min(0.0)
        return spec / (spec.sum(-1, keepdim=True) * dx + 1e-12)             # unit integral
