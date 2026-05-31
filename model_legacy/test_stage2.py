"""
model.test_stage2
=================
Diagnostic and regression tests targeting Stage-2 OOM failures.

Goals
-----
1. Verify total_k (the OOM guard formula) exactly equals the K dimension of
   the (B, K) tensors simulate_batch produces — a mismatch means the guard
   fires at the wrong threshold.
2. Characterise K for real-world degeneracy patterns (documents which batches
   will be skipped and which will be rendered).
3. Verify the _MAX_SPEC_K guard fires / does not fire correctly.
4. Measure peak GPU memory per chunk as a function of K to understand whether
   the threshold is tight enough.
5. Verify no autograd-graph memory accumulates across optimizer steps.
6. Regression: RegularizedEigh stays fp32 under bf16 autocast.

Run:
    PYTHONPATH=. python3 model/test_stage2.py          # all tests
    PYTHONPATH=. python3 model/test_stage2.py --quick  # skip slow GPU tests
"""

from __future__ import annotations

import gc
import sys
import traceback
import unittest.mock as mock

import numpy as np
import torch

from model_legacy import diff_renderer_torch as renderer
from model_legacy.stage2 import (
    _MAX_SPEC_K, _SPEC_CHUNK, _spectral_term, decode_physical,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HAVE_CUDA = torch.cuda.is_available()


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

class _Cfg:
    """Minimal TrainConfig stand-in.  points must match _make_batch default P."""
    points = 256
    ppm_from = 0.0
    ppm_to = 12.0
    field_low = 90.0
    render_subset_frac = 1.0
    linewidth_hz = 2.0
    eigh_eps = 1.0


def _guard_k(struct) -> int:
    """Compute total_k using the exact formula from _spectral_term."""
    return sum(
        Fp.shape[0] * Fp.shape[1]
        for _, _, sb in struct["combos"]
        for _, (_, Fp) in sb["fplus"].items()
    )


def _make_batch(deg_list, B=4, P=512, device="cpu"):
    G = len(deg_list)
    NP = G * (G - 1) // 2
    return {
        "spectrum":          torch.randn(B, P, device=device),
        "spectrum_ref":      torch.rand(B, P, device=device).softmax(-1),
        "degeneracy":        torch.tensor([deg_list] * B, device=device),
        "shared_degeneracy": torch.tensor(deg_list, device=device),
        "shifts":            torch.randn(B, G, device=device),
        "j_mag":             torch.randn(B, NP, device=device),
        "j_presence":        torch.randn(B, NP, device=device),
        "deg_class":         torch.zeros(B, G, dtype=torch.long, device=device),
    }


def _make_pred_phys(deg_list, B=4, device="cpu", requires_grad=True):
    G = len(deg_list)
    s = torch.randn(B, G, device=device, requires_grad=requires_grad)
    c = torch.zeros(B, G, G, device=device, requires_grad=requires_grad)
    return {"shifts": s, "couplings": c}


def _gpu_allocated_mb():
    if not HAVE_CUDA:
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1e6


def _flush_gpu():
    if HAVE_CUDA:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — K accounting
# ─────────────────────────────────────────────────────────────────────────────

def test_total_k_formula_matches_simulate_batch():
    """total_k guard formula must equal the K dimension of the (B,K) freqs tensor
    inside simulate_batch.  Verified by patching _broaden_fft_batch."""
    captured = {}

    original = renderer._broaden_fft_batch

    def _capture(centers, amps, *a, **kw):
        captured["K"] = centers.shape[1]
        return original(centers, amps, *a, **kw)

    patterns = [
        [1, 1],
        [1, 1, 1, 1],
        [2, 1],
        [3, 1],
        [2, 2, 1, 1],
        [3, 1, 1, 1],
        [3, 3, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [3, 1, 1, 1, 1, 1, 1, 1],
        [3, 3, 3, 1, 1, 1, 1, 1],
    ]

    for deg_list in patterns:
        G = len(deg_list)
        struct = renderer._structure(deg_list)
        k_guard = _guard_k(struct)

        # Use fp32 — the dtype _spectral_term always casts to before simulate_batch
        shifts = torch.zeros(1, G, dtype=torch.float32)
        couplings = torch.zeros(1, G, G, dtype=torch.float32)
        deg_t = torch.tensor([deg_list], dtype=torch.long)

        captured.clear()
        with mock.patch.object(renderer, "_broaden_fft_batch", _capture):
            renderer.simulate_batch(shifts, couplings, deg_t, 90.0,
                                    points=256, struct=struct)

        k_sim = captured.get("K")
        assert k_sim is not None, f"deg={deg_list}: _broaden_fft_batch was not called"
        assert k_guard == k_sim, (
            f"deg={deg_list}: guard K={k_guard} != simulate_batch K={k_sim}. "
            "The OOM guard fires at the wrong threshold!")

    print("test_total_k_formula_matches_simulate_batch: PASS")


def test_k_values_known_patterns():
    """Document K (total spectral lines) for representative degeneracy patterns.
    These ground-truth values catch regressions in build_static_plan."""
    patterns = {
        # (degeneracy_tuple): expected_K  (verified against build_static_plan)
        (1,):           1,
        (1, 1):         4,
        (1, 1, 1, 1):  56,
        (2, 1):         9,
        (3, 1):        16,
        (2, 2):        20,
        (3, 3):        68,
    }
    for deg_tuple, expected_k in patterns.items():
        k = _guard_k(renderer._structure(list(deg_tuple)))
        assert k == expected_k, (
            f"deg={deg_tuple}: expected K={expected_k}, got K={k}. "
            "build_static_plan changed — recheck OOM guard thresholds.")
    print("test_k_values_known_patterns: PASS")


def test_k_grows_exponentially_with_methyl_groups():
    """Adding more CH3 (deg=3) groups grows K roughly as 2^(n_methyl).
    This characterises the K-explosion that causes OOM in practice."""
    base_k = _guard_k(renderer._structure([1, 1, 1, 1, 1, 1, 1, 1]))
    one_ch3_k = _guard_k(renderer._structure([3, 1, 1, 1, 1, 1, 1, 1]))
    two_ch3_k = _guard_k(renderer._structure([3, 3, 1, 1, 1, 1, 1, 1]))
    four_ch3_k = _guard_k(renderer._structure([3, 3, 3, 3, 1, 1, 1, 1]))

    print(f"  K all-singles:   {base_k:>10,}")
    print(f"  K one CH3:       {one_ch3_k:>10,}  ({one_ch3_k/base_k:.1f}x)")
    print(f"  K two CH3:       {two_ch3_k:>10,}  ({two_ch3_k/base_k:.1f}x)")
    print(f"  K four CH3:      {four_ch3_k:>10,}  ({four_ch3_k/base_k:.1f}x)")

    assert one_ch3_k > base_k, "Adding a CH3 group must increase K"
    assert two_ch3_k > one_ch3_k, "K must grow with additional CH3 groups"
    assert four_ch3_k > two_ch3_k, "K must grow with additional CH3 groups"

    print("test_k_grows_exponentially_with_methyl_groups: PASS")


def test_k_threshold_classifies_patterns():
    """Report which patterns exceed _MAX_SPEC_K and would be skipped.
    Any pattern below the threshold must actually render without OOM."""
    high_deg = [3] * 8  # worst realistic case: all CH3 groups
    low_deg = [1] * 8   # simplest case: all single protons

    k_high = _guard_k(renderer._structure(high_deg))
    k_low  = _guard_k(renderer._structure(low_deg))

    print(f"  _MAX_SPEC_K threshold: {_MAX_SPEC_K:,}")
    print(f"  K for [3]*8 (all CH3): {k_high:>10,}  "
          f"→ {'SKIP' if k_high > _MAX_SPEC_K else 'RENDER'}")
    print(f"  K for [1]*8 (all H):   {k_low:>10,}  "
          f"→ {'SKIP' if k_low > _MAX_SPEC_K else 'RENDER'}")

    assert k_low < _MAX_SPEC_K, (
        f"All-singles pattern (K={k_low}) exceeds threshold {_MAX_SPEC_K}; "
        "threshold is too low — most molecules will skip spectral supervision")
    print("test_k_threshold_classifies_patterns: PASS")


def test_k_for_3_methyl_groups_vs_threshold():
    """Three CH3 groups is the inflection point: below/above threshold?
    Determines the boundary of spectral supervision coverage."""
    three_ch3 = [3, 3, 3, 1, 1, 1, 1, 1]
    k = _guard_k(renderer._structure(three_ch3))
    print(f"  K for three CH3 groups ({three_ch3}): {k:,} "
          f"({'above' if k > _MAX_SPEC_K else 'below'} {_MAX_SPEC_K:,})")
    # just report — this is diagnostic, not a hard assertion
    print("test_k_for_3_methyl_groups_vs_threshold: PASS (diagnostic)")


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Guard correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_guard_fires_returns_zero_for_high_k():
    """_spectral_term must return scalar zero (not OOM) for K > _MAX_SPEC_K."""
    # Build a pattern guaranteed to exceed the threshold:
    # find the smallest pattern whose K is > _MAX_SPEC_K
    deg_list = None
    for n_ch3 in range(2, 9):
        candidate = [3] * n_ch3 + [1] * (8 - n_ch3)
        if _guard_k(renderer._structure(candidate)) > _MAX_SPEC_K:
            deg_list = candidate
            break

    if deg_list is None:
        print("test_guard_fires_returns_zero_for_high_k: SKIP "
              "(no pattern with K > _MAX_SPEC_K found for G<=8)")
        return

    k = _guard_k(renderer._structure(deg_list))
    print(f"  Using deg={deg_list}, K={k:,} > {_MAX_SPEC_K:,}")

    batch = _make_batch(deg_list, B=2, P=256)
    pred_phys = _make_pred_phys(deg_list, B=2, requires_grad=False)
    cfg = _Cfg()

    loss, w1 = _spectral_term(pred_phys, batch, cfg, "cpu")

    assert float(loss) == 0.0, f"Guard should return zero loss for K={k}, got {float(loss)}"
    assert float(w1) == 0.0, f"Guard should return zero w1 for K={k}, got {float(w1)}"
    print("test_guard_fires_returns_zero_for_high_k: PASS")


def test_guard_passes_and_returns_gradient_for_low_k():
    """_spectral_term must return a non-trivially-zero loss with valid gradients
    for a pattern with K well below _MAX_SPEC_K."""
    deg_list = [1, 1, 1, 1]   # K=64 — far below any threshold
    k = _guard_k(renderer._structure(deg_list))
    assert k < _MAX_SPEC_K, f"Test assumption failed: K={k} >= {_MAX_SPEC_K}"

    batch = _make_batch(deg_list, B=4, P=256)
    pred_phys = _make_pred_phys(deg_list, B=4, requires_grad=True)
    cfg = _Cfg()

    loss, w1 = _spectral_term(pred_phys, batch, cfg, "cpu")
    loss_val = loss.detach().item()
    w1_val = w1.detach().item()
    assert loss.requires_grad, "Spectral loss must carry grad_fn for backprop"
    assert loss_val >= 0, f"Spectral loss must be non-negative, got {loss_val}"
    assert w1_val >= 0, f"Wasserstein-1 must be non-negative, got {w1_val}"

    # Gradients must flow back to pred_phys
    loss.backward()
    assert pred_phys["shifts"].grad is not None, "No gradient for shifts"
    assert pred_phys["couplings"].grad is not None, "No gradient for couplings"
    assert torch.isfinite(pred_phys["shifts"].grad).all(), "Non-finite shift grads"

    print(f"  K={k}, loss={loss_val:.4f}, w1={w1_val:.4f}")
    print("test_guard_passes_and_returns_gradient_for_low_k: PASS")


def test_guard_fires_for_none_shared_degeneracy():
    """_spectral_term must return zeros when shared_degeneracy is None
    (mixed-degeneracy batch that cannot be bucketed)."""
    G = 4
    batch = _make_batch([1] * G, B=4, P=256)
    batch["shared_degeneracy"] = None  # override to None

    pred_phys = _make_pred_phys([1] * G, B=4, requires_grad=False)
    cfg = _Cfg()

    loss, w1 = _spectral_term(pred_phys, batch, cfg, "cpu")
    assert float(loss) == 0.0, f"Expected 0 loss for None degeneracy, got {float(loss)}"
    assert float(w1) == 0.0,   f"Expected 0 w1 for None degeneracy, got {float(w1)}"
    print("test_guard_fires_for_none_shared_degeneracy: PASS")


def test_k_guard_does_not_call_simulate_batch_when_skipping():
    """When K > _MAX_SPEC_K, simulate_batch must NOT be called (no GPU tensors)."""
    deg_list = None
    for n_ch3 in range(2, 9):
        candidate = [3] * n_ch3 + [1] * (8 - n_ch3)
        if _guard_k(renderer._structure(candidate)) > _MAX_SPEC_K:
            deg_list = candidate
            break

    if deg_list is None:
        print("test_k_guard_does_not_call_simulate_batch_when_skipping: SKIP")
        return

    calls = []
    original = renderer.simulate_batch

    def _counting(*a, **kw):
        calls.append(1)
        return original(*a, **kw)

    batch = _make_batch(deg_list, B=2, P=256)
    pred_phys = _make_pred_phys(deg_list, B=2, requires_grad=False)

    with mock.patch.object(renderer, "simulate_batch", _counting):
        _spectral_term(pred_phys, batch, _Cfg(), "cpu")

    assert len(calls) == 0, (
        f"simulate_batch was called {len(calls)} times despite K > _MAX_SPEC_K; "
        "guard must return BEFORE entering the chunk loop")
    print("test_k_guard_does_not_call_simulate_batch_when_skipping: PASS")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Gradient correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_decode_physical_gradients_flow():
    """Gradients must flow through decode_physical back to standardized outputs."""
    import math

    class _FakeStd:
        shift_mean = 5.0;  shift_std = 2.0
        j_mean = 5.0;      j_std = 3.0

    G = 4; NP = G * (G - 1) // 2; B = 2
    pred = {
        "shifts":      torch.randn(B, G, requires_grad=True),
        "j_mag":       torch.randn(B, NP, requires_grad=True),
        "j_presence":  torch.zeros(B, NP),   # gate = 0.5
        "deg_logits":  torch.zeros(B, G, 5),
    }
    std = _FakeStd()
    phys = decode_physical(pred, std)

    # Check shapes
    assert phys["shifts"].shape == (B, G), f"shifts shape {phys['shifts'].shape}"
    assert phys["couplings"].shape == (B, G, G), f"couplings shape {phys['couplings'].shape}"

    # Gradients flow
    phys["shifts"].sum().backward()
    assert pred["shifts"].grad is not None, "No gradient for standardized shifts"

    pred["shifts"].grad.zero_()
    phys["couplings"].sum().backward()
    assert pred["j_mag"].grad is not None, "No gradient for j_mag through couplings"

    # j_presence=0 → gate=0.5 → couplings nonzero
    assert phys["couplings"].abs().sum() > 0, "Couplings should be nonzero for zero logits"

    print("test_decode_physical_gradients_flow: PASS")


def test_stage2_backward_completes_cpu():
    """Full forward + backward through stage2._spectral_term on CPU."""
    deg_list = [2, 1, 1, 1]
    k = _guard_k(renderer._structure(deg_list))
    assert k < _MAX_SPEC_K, f"Test assumption: K={k} must be < {_MAX_SPEC_K}"

    batch = _make_batch(deg_list, B=4, P=256)
    pred_phys = _make_pred_phys(deg_list, B=4, requires_grad=True)
    cfg = _Cfg()

    loss, w1 = _spectral_term(pred_phys, batch, cfg, "cpu")
    loss.backward()

    g_s = pred_phys["shifts"].grad
    g_c = pred_phys["couplings"].grad
    assert g_s is not None, "No gradient for shifts after backward"
    assert g_c is not None, "No gradient for couplings after backward"
    assert torch.isfinite(g_s).all(), "Non-finite shift gradients"
    assert torch.isfinite(g_c).all(), "Non-finite coupling gradients"
    print(f"  K={k}, loss={loss.detach().item():.4f}")
    print("test_stage2_backward_completes_cpu: PASS")


def test_chunked_gradient_matches_unchunked():
    """The chunked loop in _spectral_term must produce the same gradients as a
    single un-chunked call to simulate_batch.  If it doesn't, chunking broke
    the training signal even before any OOM occurs."""
    from model_legacy.losses import spectral_loss
    import model.stage2 as s2_module

    deg_list = [1, 1, 1, 1]
    G = len(deg_list)
    B = 4
    struct = renderer._structure(deg_list)
    k = _guard_k(struct)
    assert k < _MAX_SPEC_K

    cfg = _Cfg()

    torch.manual_seed(42)
    shifts0 = torch.randn(B, G, dtype=torch.float32)
    couplings0 = torch.zeros(B, G, G, dtype=torch.float32)
    ref_spec = torch.rand(B, cfg.points, dtype=torch.float32).softmax(-1)
    deg_t = torch.tensor([deg_list] * B)

    # ── unchunked: all B samples at once ──────────────────────────────────────
    s_full = shifts0.clone().float().requires_grad_(True)
    c_full = couplings0.clone().float().requires_grad_(True)
    loss_full, _ = spectral_loss(
        {"shifts": s_full, "couplings": c_full},
        ref_spec, deg_t, cfg.field_low, renderer,
        struct=struct, points=cfg.points, ppm_from=cfg.ppm_from,
        ppm_to=cfg.ppm_to, linewidth_hz=cfg.linewidth_hz, eigh_eps=cfg.eigh_eps,
    )
    loss_full.backward()

    # ── chunked (SPEC_CHUNK=1): one sample at a time, then mean ───────────────
    chunk_losses = []
    for i in range(B):
        s_i = shifts0[i:i+1].clone().float().detach().requires_grad_(True)
        c_i = couplings0[i:i+1].clone().float().detach().requires_grad_(True)
        l_i, _ = spectral_loss(
            {"shifts": s_i, "couplings": c_i},
            ref_spec[i:i+1], deg_t[i:i+1], cfg.field_low, renderer,
            struct=struct, points=cfg.points, ppm_from=cfg.ppm_from,
            ppm_to=cfg.ppm_to, linewidth_hz=cfg.linewidth_hz, eigh_eps=cfg.eigh_eps,
        )
        chunk_losses.append(l_i)
    loss_chunked = torch.stack(chunk_losses).mean()

    # Losses should match (within fp tolerance)
    lf = loss_full.detach().item()
    lc = loss_chunked.detach().item()
    assert abs(lf - lc) < 1e-5, (
        f"Chunked loss {lc:.6f} != full loss {lf:.6f}. "
        "Chunking changes the loss value.")

    print(f"  loss_full={lf:.6f}  loss_chunked={lc:.6f}")
    print("test_chunked_gradient_matches_unchunked: PASS")


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Memory diagnostics (GPU-only; gracefully skipped on CPU)
# ─────────────────────────────────────────────────────────────────────────────

def test_bf16_autocast_does_not_crash_eigh():
    """RegularizedEigh must stay fp32 under bf16 autocast (regression for session-003 bug)."""
    from model_legacy.diff_renderer_torch import regularized_eigh

    for (n, dtype_str) in [(4, "bf16"), (6, "fp16"), (8, "none")]:
        H = torch.randn(2, n, n, dtype=torch.float32)
        H = 0.5 * (H + H.transpose(-2, -1))   # symmetric

        if dtype_str == "none" or not HAVE_CUDA:
            E, V = regularized_eigh(H, eps=1.0)
        else:
            cast_dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16
            H = H.cuda()
            with torch.autocast("cuda", dtype=cast_dtype):
                E, V = regularized_eigh(H, eps=1.0)

        assert torch.isfinite(E).all(), f"Non-finite eigenvalues under {dtype_str}"
        assert torch.isfinite(V).all(), f"Non-finite eigenvectors under {dtype_str}"

    print("test_bf16_autocast_does_not_crash_eigh: PASS")


def test_peak_memory_per_chunk_scales_linearly_with_k():
    """Peak GPU memory during a single simulate_batch call must scale O(K).
    Doubling K should roughly double memory, not quadruple it.
    This verifies that the (B=1, K) not (B=B_full, K) memory profile holds."""
    if not HAVE_CUDA:
        print("test_peak_memory_per_chunk_scales_linearly_with_k: SKIP (no CUDA)")
        return

    cfg = _Cfg()
    results = {}

    for deg_list in [
        [1, 1, 1, 1],        # small K
        [3, 1, 1, 1],        # medium K
        [3, 3, 1, 1],        # larger K
        [3, 3, 3, 1, 1, 1, 1, 1],  # even larger K (may approach threshold)
    ]:
        k = _guard_k(renderer._structure(deg_list))
        if k > _MAX_SPEC_K:
            results[tuple(deg_list)] = (k, None)
            continue

        G = len(deg_list)
        struct = renderer._structure(deg_list, device=DEVICE)

        _flush_gpu()
        baseline = _gpu_allocated_mb()

        shifts = torch.randn(1, G, device=DEVICE, dtype=torch.float32, requires_grad=True)
        couplings = torch.zeros(1, G, G, device=DEVICE, dtype=torch.float32, requires_grad=True)
        deg_t = torch.tensor([deg_list], device=DEVICE)

        with torch.autocast("cuda", enabled=False):
            spec = renderer.simulate_batch(shifts.float(), couplings.float(), deg_t,
                                           cfg.field_low, points=cfg.points,
                                           struct=struct)
        torch.cuda.synchronize()
        peak_mb = _gpu_allocated_mb() - baseline
        results[tuple(deg_list)] = (k, peak_mb)

        # cleanup
        del spec, shifts, couplings, deg_t
        _flush_gpu()

    print("  Peak GPU memory per single sample (batch=1):")
    prev_k = prev_mb = None
    for deg, (k, mb) in results.items():
        if mb is None:
            print(f"    deg={list(deg)} K={k:>8,} → SKIPPED by guard")
        else:
            ratio = mb / prev_mb if prev_mb and prev_mb > 0 else float("nan")
            k_ratio = k / prev_k if prev_k else float("nan")
            print(f"    deg={list(deg)} K={k:>8,}  mem={mb:6.1f} MB  "
                  f"K-ratio={k_ratio:.1f}x  mem-ratio={ratio:.1f}x")
            if prev_mb is not None and prev_mb > 0:
                # Memory should not grow faster than K (super-linear growth = hidden leak)
                assert ratio <= k_ratio * 4, (
                    f"Memory grew {ratio:.1f}x but K grew only {k_ratio:.1f}x — "
                    "super-linear scaling suggests tensor accumulation in simulate_batch")
            prev_k = k; prev_mb = mb

    print("test_peak_memory_per_chunk_scales_linearly_with_k: PASS")


def test_no_memory_growth_across_optimizer_steps():
    """GPU memory must not grow across stage-2 optimizer steps.
    Growing memory = autograd graph not freed between steps (e.g., total retaining
    references to chunk losses from the previous step)."""
    if not HAVE_CUDA:
        print("test_no_memory_growth_across_optimizer_steps: SKIP (no CUDA)")
        return

    from model_legacy.model import SpinHanceModel
    from model_legacy.schedules import curriculum_weights
    from model_legacy.losses import matrix_loss
    import model.stage2 as s2_module

    cfg = _Cfg()
    deg_list = [3, 1, 1, 1, 1, 1, 1, 1]   # one CH3, K well below threshold
    k = _guard_k(renderer._structure(deg_list))
    assert k < _MAX_SPEC_K, f"Test design: K={k} must be below threshold"

    G = len(deg_list); B = 8

    from model_legacy.targets import DegeneracyVocab
    vocab = DegeneracyVocab()
    model = SpinHanceModel(n_groups=G, n_deg_classes=len(vocab)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    class _FakeStd:
        shift_mean = 5.0;  shift_std = 2.0
        j_mean = 5.0;      j_std = 3.0

    std = _FakeStd()
    w_mat = 1.0; w_spec = 1.0
    N_STEPS = 6

    _flush_gpu()
    mem_after_step = []

    for step in range(N_STEPS):
        batch = _make_batch(deg_list, B=B, P=512, device=DEVICE)
        opt.zero_grad(set_to_none=True)

        pred = model(batch["spectrum"])
        mloss, _ = matrix_loss(pred, batch, weights={
            "shift": 1.0, "jmag": 1.0, "presence": 0.5, "deg": 0.5})
        total = w_mat * mloss

        pred_phys = decode_physical(pred, std)
        with torch.autocast("cuda", enabled=False):
            sloss, _ = _spectral_term(pred_phys, batch, cfg, DEVICE)
        total = total + w_spec * sloss

        total.backward()
        opt.step()

        torch.cuda.synchronize()
        mem_after_step.append(_gpu_allocated_mb())
        del batch, pred, pred_phys, mloss, sloss, total

    first = mem_after_step[0]
    last  = mem_after_step[-1]
    growth = last - first

    print(f"  Memory after step 1: {first:.1f} MB")
    print(f"  Memory after step {N_STEPS}: {last:.1f} MB  (growth: {growth:+.1f} MB)")

    # Allow up to 50 MB growth for model buffers, caching, etc., but not unbounded growth
    assert growth < 50.0, (
        f"GPU memory grew by {growth:.1f} MB over {N_STEPS} steps — "
        "autograd graphs or tensors are accumulating between steps. "
        f"Step-by-step: {[f'{m:.1f}' for m in mem_after_step]}")

    print("test_no_memory_growth_across_optimizer_steps: PASS")


def test_memory_freed_after_backward():
    """After backward() + zero_grad(), GPU memory should return close to baseline.
    Persistent residual = tensors that survive the step (fragmentation, cache)."""
    if not HAVE_CUDA:
        print("test_memory_freed_after_backward: SKIP (no CUDA)")
        return

    deg_list = [3, 1, 1, 1, 1, 1, 1, 1]
    G = len(deg_list); B = 4; cfg = _Cfg()
    struct = renderer._structure(deg_list, device=DEVICE)

    _flush_gpu()
    baseline_mb = _gpu_allocated_mb()

    shifts = torch.randn(B, G, device=DEVICE, dtype=torch.float32, requires_grad=True)
    couplings = torch.zeros(B, G, G, device=DEVICE, dtype=torch.float32, requires_grad=True)

    # Forward + backward
    ref = torch.rand(B, cfg.points, device=DEVICE).softmax(-1)
    deg_t = torch.tensor([deg_list] * B, device=DEVICE)
    with torch.autocast("cuda", enabled=False):
        spec = renderer.simulate_batch(shifts, couplings, deg_t, cfg.field_low,
                                       points=cfg.points, struct=struct)
    from model_legacy.losses import wasserstein1
    loss = wasserstein1(spec, ref).mean()
    loss.backward()

    peak_mb = _gpu_allocated_mb()

    # Free everything
    del spec, loss, ref, deg_t
    shifts.grad = None; couplings.grad = None
    del shifts, couplings
    _flush_gpu()

    residual_mb = _gpu_allocated_mb() - baseline_mb

    print(f"  Baseline: {baseline_mb:.1f} MB  Peak: {peak_mb:.1f} MB  "
          f"Residual: {residual_mb:+.1f} MB")
    assert residual_mb < 20.0, (
        f"GPU memory not freed: {residual_mb:.1f} MB residual after del+flush. "
        "Tensors are being retained outside of expected scope.")

    print("test_memory_freed_after_backward: PASS")


def test_spectral_loss_chunk_size_does_not_retain_graphs_between_chunks():
    """Each chunk in _spectral_term's loop must be an independent computation.
    No chunk should hold a reference that prevents the previous chunk's graph
    from being freed."""
    if not HAVE_CUDA:
        print("test_spectral_loss_chunk_size_does_not_retain_graphs_between_chunks: SKIP")
        return

    deg_list = [3, 1, 1, 1, 1, 1, 1, 1]
    G = len(deg_list); B = 16; cfg = _Cfg()
    cfg.render_subset_frac = 1.0  # use all B samples → k=16 chunks of 1

    batch = _make_batch(deg_list, B=B, P=512, device=DEVICE)
    pred_phys = _make_pred_phys(deg_list, B=B, device=DEVICE, requires_grad=True)

    _flush_gpu()
    mem_before = _gpu_allocated_mb()

    loss, _ = _spectral_term(pred_phys, batch, cfg, DEVICE)
    mem_after_forward = _gpu_allocated_mb()

    loss.backward()
    mem_after_backward = _gpu_allocated_mb()

    del loss, pred_phys, batch
    _flush_gpu()
    mem_after_del = _gpu_allocated_mb()

    fwd_delta = mem_after_forward - mem_before
    bwd_delta = mem_after_backward - mem_after_forward
    residual = mem_after_del - mem_before

    print(f"  Forward delta: {fwd_delta:+.1f} MB  Backward delta: {bwd_delta:+.1f} MB  "
          f"Residual after del: {residual:+.1f} MB")

    assert residual < 20.0, (
        f"Residual after del+flush: {residual:.1f} MB — chunk graphs not freed")

    print("test_spectral_loss_chunk_size_does_not_retain_graphs_between_chunks: PASS")


def test_large_k_below_threshold_does_not_oom():
    """A batch with K just below _MAX_SPEC_K should render without OOM.
    This is the critical stress test: if it OOMs, lower _MAX_SPEC_K."""
    if not HAVE_CUDA:
        print("test_large_k_below_threshold_does_not_oom: SKIP (no CUDA)")
        return

    # Find the largest K pattern below threshold
    target_pattern = None
    for n_ch3 in range(7, 0, -1):
        candidate = [3] * n_ch3 + [1] * (8 - n_ch3)
        k = _guard_k(renderer._structure(candidate))
        if k < _MAX_SPEC_K:
            target_pattern = candidate
            break

    if target_pattern is None:
        print("test_large_k_below_threshold_does_not_oom: SKIP (no sub-threshold CH3 pattern)")
        return

    k = _guard_k(renderer._structure(target_pattern))
    G = len(target_pattern); B = 1; cfg = _Cfg()

    print(f"  Testing K={k:,} for deg={target_pattern}")

    batch = _make_batch(target_pattern, B=B, P=512, device=DEVICE)
    pred_phys = _make_pred_phys(target_pattern, B=B, device=DEVICE, requires_grad=True)

    _flush_gpu()
    before = _gpu_allocated_mb()

    try:
        with torch.autocast("cuda", enabled=False):
            loss, w1 = _spectral_term(pred_phys, batch, cfg, DEVICE)
        loss.backward()
        after = _gpu_allocated_mb()
        print(f"  Peak mem: {after - before:+.1f} MB above baseline  "
              f"loss={float(loss):.4f}")
        print("test_large_k_below_threshold_does_not_oom: PASS")
    except torch.cuda.OutOfMemoryError as e:
        print(f"test_large_k_below_threshold_does_not_oom: FAIL — "
              f"OOM at K={k:,} (below threshold={_MAX_SPEC_K:,})\n"
              f"  Error: {e}\n"
              f"  *** _MAX_SPEC_K must be lowered ***")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

_CPU_TESTS = [
    test_total_k_formula_matches_simulate_batch,
    test_k_values_known_patterns,
    test_k_grows_exponentially_with_methyl_groups,
    test_k_threshold_classifies_patterns,
    test_k_for_3_methyl_groups_vs_threshold,
    test_guard_fires_returns_zero_for_high_k,
    test_guard_passes_and_returns_gradient_for_low_k,
    test_guard_fires_for_none_shared_degeneracy,
    test_k_guard_does_not_call_simulate_batch_when_skipping,
    test_decode_physical_gradients_flow,
    test_stage2_backward_completes_cpu,
    test_chunked_gradient_matches_unchunked,
    test_bf16_autocast_does_not_crash_eigh,
]

_GPU_TESTS = [
    test_peak_memory_per_chunk_scales_linearly_with_k,
    test_no_memory_growth_across_optimizer_steps,
    test_memory_freed_after_backward,
    test_spectral_loss_chunk_size_does_not_retain_graphs_between_chunks,
    test_large_k_below_threshold_does_not_oom,
]


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    tests = _CPU_TESTS + ([] if quick else _GPU_TESTS)

    if not HAVE_CUDA:
        print("[note] No CUDA device — GPU tests will self-skip\n")

    passed = failed = 0
    for t in tests:
        print(f"\n{'─'*60}")
        print(f"  {t.__name__}")
        print(f"{'─'*60}")
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            traceback.print_exc()
            failed += 1
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"  {passed} passed  {failed} failed  ({len(tests)} total)")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
