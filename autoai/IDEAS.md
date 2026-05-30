# SpinHance ML Ideas

The orchestrator reads this file at the start of every training cycle.
Add your ideas below — describe the approach, not the implementation.
Opus will decide what to try and in what order.

---

## Project context

**Goal:** Train a neural network that maps a 16384-point normalized ¹H NMR spectrum
(simulated at 90 MHz) to the underlying spin-system parameters:
- 8×8 symmetric J-coupling matrix (Hz, off-diagonal) + chemical shifts (ppm, diagonal)
- 8-element degeneracy vector (protons per spin group)

Combined representation: an 8×9 matrix where column 9 is degeneracy.

**Key challenge:** Spin-group labels are arbitrary — the same molecule has 8! = 40320
equivalent matrix representations. The loss must be permutation-invariant.

**Data location:** `data/processed/` — spectra in `spectra/90MHz/`, matrices in `matrices/`

---

## Ideas

<!-- Sam, Yiming, Lucas — add your ideas below. Each idea should be a ## section. -->

### Template
**Approach:** (what architecture / loss / trick to try)
**Motivation:** (why you think it'll help)
**References:** (optional)
