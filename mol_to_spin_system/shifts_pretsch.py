from __future__ import annotations

"""Pure-Python ¹H chemical-shift engine based on Pretsch (2009).

"Structure Determination of Organic Compounds — Tables of Spectral Data",
M. Badertscher, P. Bühlmann, E. Pretsch, Springer 2009, Chapter 5 (¹H NMR).

This is "Tier 3" of the enrichment campaign and is intended to *eventually*
replace the Java nmrshiftdb2/HOSE predictor (``mol_to_spin_system/shifts.py``)
with a dependency-free additive-increment estimator. It is NOT yet wired into
the generation pipeline — a human reviews the validation gate first.

Public entry point
------------------
``predict_shifts_pretsch(mol) -> dict[int, float]``
    key  = RDKit atom index of a protium H (¹H; isotope 0 or 1 only)
    value= predicted δ in ppm

Design
------
The estimator runs several *paths*, each transcribed verbatim from a specific
book table (page citations inline). Each path knows which protons it can claim;
a proton claimed by no path receives a coarse class default and is recorded as
"uncovered". Use :func:`predict_shifts_pretsch_verbose` to obtain, per H, both
the value and the path that produced it (for coverage accounting).

Conventions
-----------
* Increment tables are stored as plain dicts (substituent-key -> increments) so
  they are auditable against the book.
* Aromatic position relative to an H: ortho = ring-distance 1, meta = 2,
  para = 3 (benzene). Heteroaromatic positions come from
  :func:`mol_to_spin_system.heteroaromatic._canonical_positions`.
* Protium only: H atoms with ``GetIsotope() not in (0, 1)`` are skipped.
"""

from rdkit import Chem

from mol_to_spin_system.heteroaromatic import (
    _canonical_positions,
    _NAME,
)

__all__ = [
    "predict_shifts_pretsch",
    "predict_shifts_pretsch_verbose",
]

# ───────────────────────────────────────────────────────────────────────────
# Path 1 — Monosubstituted-benzene aromatic protons
#   Pretsch p178-179 (§5.5, "Effect of Substituents on the ¹H Chemical Shifts
#   of Monosubstituted Benzenes").  δ = 7.34 + Σ Z_i over every ring substituent
#   i, using Z_ortho / Z_meta / Z_para according to the ring distance between the
#   substituted carbon and the carbon bearing the proton (1 / 2 / 3 bonds around
#   the ring).  Base 7.34 is benzene δ minus nothing (benzene itself = 7.26;
#   the table's base of 7.34 is the textbook value used with these increments).
#
# Keys are canonicalised substituent descriptors produced by :func:`_aryl_subst`.
# Values are (Z_ortho, Z_meta, Z_para).  VERBATIM from p178 (Z2/Z3/Z4) and the
# p179 continuation.  Entries flagged "# uncertain" had a marginal render.
# ───────────────────────────────────────────────────────────────────────────
AROM_BASE = 7.34  # p178, δ_Hi = 7.34 + Σ Z_i

# p178 — carbon substituents
AROM_INCR: dict[str, tuple[float, float, float]] = {
    "CH3":            (-0.17, -0.09, -0.17),  # –CH3
    "CH2CH3":         (-0.14, -0.06, -0.17),  # –CH2CH3
    "CH(CH3)2":       (-0.13, -0.08, -0.18),  # –CH(CH3)2
    "C(CH3)3":        ( 0.05, -0.05, -0.18),  # –C(CH3)3  (p178: 0.05/-0.05/-0.18; starter said -0.04 meta)
    "CF3":            ( 0.19, -0.07,  0.00),  # –CF3  (uncertain meta sign small)
    "CCl3":           ( 0.55, -0.07, -0.09),  # –CCl3
    "CH2OH":          (-0.07, -0.07, -0.07),  # –CH2OH
    "CH2Cl":          ( 0.08, -0.02, -0.09),  # –CH2Cl  (uncertain)
    "CH=CHPh":        ( 0.16,  0.06, -0.15),  # –CH=CH–phenyl (trans)
    "C#CH":           ( 0.16, -0.01, -0.01),  # –C≡CH
    "C#CPh":          ( 0.20, -0.04, -0.07),  # –C≡C–phenyl
    "phenyl":         ( 0.22,  0.06, -0.04),  # –phenyl
    "2-pyridyl":      ( 0.73,  0.09,  0.02),  # –2-pyridyl (uncertain)
    "CH=CH2":         ( 0.08, -0.02, -0.09),  # –CH=CH2  (starter; vinyl)
    # p178 — halogens (X)
    "F":              (-0.31, -0.03, -0.21),  # –F
    "Cl":             (-0.01, -0.06, -0.12),  # –Cl  (p178: -0.01/-0.06/-0.12)
    "Br":             ( 0.15, -0.12, -0.06),  # –Br
    "I":              ( 0.36, -0.24, -0.02),  # –I  (p178: 0.36/-0.24/-0.02)
    # p178 — oxygen (O)
    "OH":             (-0.51, -0.10, -0.41),  # –OH  (uncertain ortho ~ -0.51)
    "OCH3":           (-0.44, -0.05, -0.40),  # –OCH3
    "OCH2CH=CH2":     (-0.45, -0.13, -0.43),  # –OCH2CH=CH2
    "Ophenyl":        (-0.33, -0.02, -0.25),  # –O–phenyl
    "OCOCH3":         (-0.26,  0.03, -0.12),  # –OCO–CH3 (acetoxy)
    "OSO2CH3":        (-0.05,  0.07, -0.01),  # –O–SO2CH3  (uncertain)
    # p178 — nitrogen (N)
    "NH2":            (-0.67, -0.20, -0.59),  # –NH2  (p178 verbatim)
    "NHCH3":          (-0.73, -0.16, -0.64),  # –NHCH3
    "N(CH3)2":        (-0.60, -0.10, -0.62),  # –N(CH3)2  (p178: -0.60/-0.10/-0.62)
    "N(phenyl)2":     (-0.26, -0.10, -0.34),  # –N(phenyl)2
    "N+(CH3)3":       ( 0.72,  0.40,  0.34),  # –N⁺(CH3)3 I⁻
    "NHCHO":          ( 0.25,  0.03, -0.13),  # –NHCHO (trans to O)  (uncertain)
    "N=CHPh":         (-0.20,  0.21, -0.05),  # –N=CH–phenyl (cis to O)  (uncertain)
    "NHCOCH3":        ( 0.15, -0.02, -0.23),  # –NHCOCH3  (p178: 0.15/-0.02/-0.23)
    "NHCSNH2":        ( 0.14,  0.07, -0.14),  # –NHCSNH2  (uncertain)
    # p179 continuation — more N
    "NHNH2":          (-0.60, -0.08, -0.55),  # –NHNH2
    "N=N-phenyl":     ( 0.67,  0.20,  0.20),  # –N=N–phenyl
    "NO":             ( 0.55,  0.29,  0.35),  # –NO
    "NO2":            ( 0.93,  0.26,  0.39),  # –NO2  (p179 verbatim)
    "NCS":            ( 0.32,  0.14,  0.28),  # –NCS  (isothiocyanate)
    # p179 — sulfur (S)
    "SH":             (-0.08, -0.16, -0.22),  # –SH
    "SCH3":           (-0.08, -0.10, -0.24),  # –SCH3
    "Sphenyl":        (-0.06, -0.20, -0.35),  # –S–phenyl  (uncertain)
    "SSphenyl":       ( 0.13, -0.05, -0.10),  # –S–S–phenyl  (uncertain)
    "SO2CH=CH2":      ( 0.28,  0.15,  0.15),  # –S(O)2–CH=CH2
    "SOCH3":          ( 0.29,  0.09,  0.13),  # –S(O)–CH3
    "SO2CH3":         ( 0.70,  0.37,  0.41),  # –S(O)2CH3  (p179)
    "SOCH3b":         ( 0.60,  0.26,  0.28),  # –S(O)–CH3 (alt)  (uncertain)
    "SO2Cl":          ( 0.68,  0.27,  0.37),  # –S(O)2Cl
    "SO2NH2":         ( 0.51,  0.28,  0.24),  # –S(O)2NH2  (sulfonamide)
    # p179 — carbonyl (O=C)
    "CHO":            ( 0.54,  0.19,  0.29),  # –CHO  (p179)
    "COCH3":          ( 0.62,  0.12,  0.22),  # –COCH3 (acetyl)
    "COCH2CH3":       ( 0.61,  0.11,  0.21),  # –COCH2CH3
    "COphenyl":       ( 0.56,  0.12,  0.23),  # –CO–phenyl (benzoyl)
    "CO-2-pyridyl":   ( 0.86,  0.11,  0.20),  # –CO–(2-pyridyl)  (uncertain)
    "COOH":           ( 0.70,  0.09,  0.21),  # –COOH  (p179)
    "COOCH3":         ( 0.73,  0.11,  0.20),  # –COOCH3 (ester)
    "COOCH(CH3)2":    ( 0.73,  0.11,  0.20),  # –COOCH(CH3)2
    "COOphenyl":      ( 0.87,  0.18,  0.30),  # –COO–phenyl
    "CONH2":          ( 0.48,  0.11,  0.19),  # –CONH2 (amide)
    "COF":            ( 0.71,  0.21,  0.38),  # –COF
    "COCl":           ( 0.77,  0.15,  0.35),  # –COCl
    "COBr":           ( 0.70,  0.15,  0.32),  # –COBr
    "CH=N-phenyl":    ( 0.64,  0.24,  0.24),  # –CH=N–phenyl
    "C#N":            ( 0.32,  0.14,  0.28),  # –C≡N (nitrile)  (uncertain — shares p179 NCS row)
    # p179 — metals / misc (M) — rare in drug-like sets, included for completeness
    "Si(CH3)3":       ( 0.19,  0.00,  0.00),  # –Si(CH3)3
}


# ───────────────────────────────────────────────────────────────────────────
# Path 2 — Substituted-alkane CH / CH2 / CH3
#   Pretsch p160 (§5.3, "Estimation of ¹H Chemical Shifts of Substituted
#   Alkanes").  Shoolery-type additivity:
#       δ(CH3 R)      = 0.86 + Z_α
#       δ(CH2 R R')   = 1.37 + Σ Z_α(i) + Σ Z_β(j)
#       δ(CH  R R' R")= 1.50 + Σ Z_α(i) + Σ Z_β(j)
#   Z_α = increment for a substituent on the SAME carbon as the proton.
#   Z_β = increment for a substituent on an ADJACENT carbon.
#   The table gives (Z_α, Z_β) per CH3 / CH2 / CH column; the CH column reuses
#   the CH2 β where its own β is blank (book footnote).  VERBATIM from p160.
#
# Values transcribed as: key -> ((CH3_a, CH3_b), (CH2_a, CH2_b), (CH_a, CH_b)).
# A None means the cell was blank in the book (fall back to the CH2 value).
# ───────────────────────────────────────────────────────────────────────────
ALK_BASE = {1: 0.86, 2: 1.37, 3: 1.50}  # base δ by # heavy substituents (CH3/CH2/CH)

# (Z_alpha, Z_beta) columns for CH3, CH2, CH
ALK_INCR: dict[str, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = {
    # C  — carbon substituents (p160)
    "C":        ((0.00, 0.05), (0.00, -0.06), (0.17, -0.01)),  # –C< (alkyl)
    "C=C":      ((0.85, 0.20), (0.63, 0.00), (0.68, 0.03)),    # –C=C<
    "C#C":      ((0.94, 0.32), (0.70, 0.13), (1.04, None)),    # –C≡C<
    "phenyl":   ((1.51, 0.38), (1.22, 0.29), (1.28, 0.38)),    # –phenyl
    # X — halogens (p160)
    "F":        ((3.41, 0.41), (2.76, 0.16), (1.83, 0.27)),    # –F
    "Cl":       ((2.20, 0.63), (2.05, 0.24), (1.84, 0.18)),    # –Cl
    "Br":       ((1.83, 0.83), (1.97, 0.46), (2.44, 0.41)),    # –Br
    "I":        ((1.30, 1.02), (1.80, 0.53), (2.46, 0.45)),    # –I
    # O — oxygen (p160)
    "OH":       ((2.53, 0.25), (2.20, 0.15), (1.73, 0.08)),    # –OH
    "OC":       ((2.38, 0.25), (2.04, 0.13), (1.85, 0.32)),    # –O–C< (ether)
    "OC=C":     ((2.64, 0.36), (2.63, 0.13), (2.20, 0.50)),    # –O–C=C<  (uncertain)
    "Ophenyl":  ((2.87, 0.47), (2.61, 0.38), (2.20, 0.50)),    # –O–phenyl
    "OCO":      ((2.81, 0.44), (2.83, 0.24), (2.47, 0.59)),    # –O–CO– (acyloxy)
    # N — nitrogen (p160)
    "N":        ((1.61, 0.14), (1.32, 0.22), (1.23, 0.23)),    # –N< (amine)
    "N+":       ((2.44, 0.39), (1.91, 0.40), (1.78, 0.56)),    # –N⁺< (uncertain)
    "NCO":      ((1.88, 0.34), (1.63, 0.22), (2.10, 0.62)),    # –N–CO– (amide N)
    "NO2":      ((3.43, 0.65), (3.08, 0.58), (2.31, None)),    # –NO2
    "C#N":      ((1.12, 0.45), (1.08, 0.33), (1.00, None)),    # –C≡N (nitrile)
    # S — sulfur (p160)
    "S":        ((1.14, 0.45), (1.23, 0.26), (1.06, 0.31)),    # –S–
    "SCO":      ((1.41, 0.37), (1.54, 0.63), (1.31, 0.19)),    # –S–CO–
    "SO":       ((1.64, 0.36), (1.24, 0.30), (1.25, None)),    # –S(O)–
    "SO2":      ((1.98, 0.42), (2.08, 0.52), (1.50, 0.40)),    # –S(O)2–
    "NCS":      ((1.75, 0.66), (1.62, None), (1.64, None)),    # –NCS
    # O=C — carbonyl on the same/adjacent carbon (p160)
    "CHO":      ((1.34, 0.21), (1.07, 0.29), (0.86, 0.22)),    # –CHO
    "CO":       ((1.23, 0.20), (1.12, 0.24), None),            # –CO– (ketone)
    "COOH":     ((1.22, 0.23), (0.90, 0.23), (0.87, 0.32)),    # –COOH
    "COO":      ((1.15, 0.28), (0.92, 0.35), (0.83, 0.63)),    # –COO– (ester C)
    "CON":      ((1.16, 0.28), (0.85, 0.24), (0.94, 0.30)),    # –CO–N< (amide C)
    "COCl":     ((1.94, 0.22), (1.51, 0.25), None),            # –COCl
}


# ───────────────────────────────────────────────────────────────────────────
# Path 4 — Alkene / vinyl protons
#   Pretsch p168-169 (§5.2, "Estimation of ¹H Chemical Shifts of Substituted
#   Ethylenes").  δ = 5.25 + Z_gem + Z_cis + Z_trans, where each ring/chain
#   substituent on the C=C contributes the increment matching its geometric
#   relationship (geminal / cis / trans) to the proton in question.  Base 5.25
#   = ethylene.  VERBATIM from p168 (carbon/halogen/oxygen/nitrogen) and the
#   p169 continuation (sulfur/carbonyl).  (Z_gem, Z_cis, Z_trans).
# ───────────────────────────────────────────────────────────────────────────
ALKENE_BASE = 5.25

ALKENE_INCR: dict[str, tuple[float, float, float]] = {
    "H":          ( 0.00,  0.00,  0.00),  # –H
    # C
    "alkyl":      ( 0.45, -0.22, -0.28),  # –alkyl
    "C":          ( 0.45, -0.22, -0.28),  # alias
    "CH2O":       ( 0.64, -0.01, -0.02),  # –CH2O–
    "CH2N":       ( 0.58, -0.10, -0.08),  # –CH2N<  (uncertain)
    "CH2S":       ( 0.71, -0.13, -0.22),  # –CH2S–
    "CH2CO":      ( 0.69, -0.08, -0.06),  # –CH2CO–  (uncertain)
    "CH2-aromatic": ( 1.05, -0.29, -0.32),  # –CH2–aromatic
    "CHF":        ( 0.70,  0.11, -0.04),  # –CHF2 etc  (uncertain)
    "CF3":        ( 0.66,  0.32,  0.21),  # –CF3
    "CH2CN":      ( 0.69, -0.08, -0.06),  # –CH2CN  (uncertain — shares p168 row)
    "C=C":        ( 1.00, -0.09, -0.23),  # –C=C< conjugated
    "C=Cconj":    ( 1.24,  0.02,  0.05),  # –C=C< conjugated (alt)
    "C#C":        ( 0.47,  0.38,  0.12),  # –C≡C–
    "aromatic":   ( 1.38,  0.36, -0.07),  # –aromatic
    "aromatic-fixed": ( 1.60,  None, -0.05),  # –aromatic, fixed
    "aromatic-o": ( 1.65,  0.19,  0.09),  # –aromatic, o-substituted
    # X
    "F":          ( 1.54, -0.40, -1.02),  # –F
    "Cl":         ( 1.08,  0.18,  0.13),  # –Cl
    "Br":         ( 1.07,  0.45,  0.55),  # –Br
    "I":          ( 1.14,  0.81,  0.88),  # –I
    # O
    "OC":         ( 1.22, -1.07, -1.21),  # –O–C< (sp3 ether)
    "OCsp2":      ( 1.21, -0.60, -1.00),  # –O–C< (sp2)
    "OCO":        ( 2.11, -0.35, -0.64),  # –O–CO–
    # N
    "N":          ( 0.80, -1.26, -1.21),  # –NR2 (R: H, C< sp3)
    "Nsp2":       ( 1.17, -0.53, -0.99),  # –NR< (C= sp2)
    "NCO":        ( 2.08, -0.57, -0.72),  # –N–CO–
    "N=N-phenyl": ( 2.39,  1.11,  0.67),  # –N=N–phenyl
    "NO2":        ( 1.87,  1.30,  0.62),  # –NO2
    "C#N":        ( 0.27,  0.75,  0.55),  # –C≡N
    # S (p169)
    "S":          ( 1.11, -0.29, -0.13),  # –S–
    "SO":         ( 1.27,  0.67,  0.41),  # –S(O)–
    "SO2":        ( 1.55,  1.16,  0.93),  # –S(O)2–
    "SCO":        ( 1.41,  0.06,  0.02),  # –S–CO–
    "SCN":        ( 0.94,  0.45,  0.41),  # –SCN
    "SF":         ( 1.68,  0.61,  0.49),  # –SF
    # O=C (p169)
    "CHO":        ( 1.02,  0.95,  1.17),  # –CHO
    "CO":         ( 1.10,  1.12,  0.87),  # –CO–
    "COconj":     ( 1.06,  0.91,  0.74),  # –CO– conjugated
    "COOH":       ( 0.97,  1.41,  0.71),  # –COOH
    "COOHconj":   ( 0.80,  0.98,  0.32),  # –COOH conjugated
    "COO":        ( 0.80,  1.18,  0.55),  # –COO–
    "COOconj":    ( 0.78,  1.01,  0.46),  # –COOR conjugated
    "CONH":       ( 1.37,  0.98,  0.46),  # –CONR2
    "COCl":       ( 1.11,  1.46,  1.01),  # –COCl
}


# ───────────────────────────────────────────────────────────────────────────
# Path 3 — Heteroaromatic ring protons (base shifts)
#   Pretsch §5.6 (p184-194).  Per-ring base δ by canonical ring position
#   (heteroatom = position 1, IUPAC lowest-locant numbering — same scheme as
#   mol_to_spin_system.heteroaromatic._canonical_positions / _NAME).  These are
#   the *parent* (unsubstituted) ring shifts; substituent corrections from the
#   per-ring Z-tables (p180/186-188) are NOT applied (known gap — see module
#   docstring), so substituted hetero rings get base+flag.
#   VERBATIM δ from the structure drawings on the cited pages.
# ───────────────────────────────────────────────────────────────────────────
HETARO_BASE: dict[str, dict[int, float]] = {
    # p184/187 furan (O=1): H2 7.42, H3 6.38, H4 6.38, H5 7.42
    "furan":      {2: 7.42, 3: 6.38, 4: 6.38, 5: 7.42},
    # p184/188 thiophene (S=1): H2 7.31, H3 7.09
    "thiophene":  {2: 7.31, 3: 7.09, 4: 7.09, 5: 7.31},
    # p184/187 pyrrole (NH=1): H2 6.71, H3 6.23  (NH ~8 broad, not predicted here)
    "pyrrole":    {2: 6.71, 3: 6.23, 4: 6.23, 5: 6.71},
    # p185 pyridine (N=1, CDCl3): H2 8.59, H3 7.25, H4 7.62
    "pyridine":   {2: 8.59, 3: 7.25, 4: 7.62, 5: 7.25, 6: 8.59},
    # p185 pyridazine (1,2-diazine; N1,N2): H3 9.22, H4 7.40 (a 9.22, b 7.40, c 7.32)
    "pyridazine": {3: 9.22, 4: 7.40, 5: 7.40, 6: 9.22},
    # p185 pyrimidine (1,3-diazine; N1,N3): H2 9.27, H4 8.78, H5 7.38
    "pyrimidine": {2: 9.27, 4: 8.78, 5: 7.38, 6: 8.78},
    # p185 pyrazine (1,4-diazine): all four H equivalent at 8.63
    "pyrazine":   {2: 8.63, 3: 8.63, 5: 8.63, 6: 8.63},
    # ── azoles (p184), verified against the rendered page (Pretsch verbatim) ──
    # 1,3-oxazole (O1,C2,N3,C4,C5): H2 7.90, H4 7.15 (next to N3), H5 7.68 (next to O1)
    "oxazole":    {2: 7.90, 4: 7.15, 5: 7.68},
    # 1,2-oxazole / isoxazole (O1,N2,C3,C4,C5): H3 8.49, H4 6.38, H5 8.31
    "isoxazole":  {3: 8.49, 4: 6.38, 5: 8.31},
    # 1,3-thiazole (S1,C2,N3,C4,C5): H2 8.88, H4 7.98 (next to N3), H5 7.41 (next to S1)
    "thiazole":   {2: 8.88, 4: 7.98, 5: 7.41},
    # 1,3-diazole / imidazole (N1,C2,N3,C4,C5): H2 7.74, H4/H5 7.13 (tautomer-averaged)
    "imidazole":  {2: 7.74, 4: 7.13, 5: 7.13},
    # 1,2-diazole / pyrazole (N1,N2,C3,C4,C5): H3 7.74, H4 6.10, H5 7.74
    "pyrazole":   {3: 7.74, 4: 6.10, 5: 7.74},
}


# ───────────────────────────────────────────────────────────────────────────
# Fused (condensed) heteroaromatics — SMARTS-anchored base shifts.
#   Pretsch p191-194 (§5.6.2).  Because RDKit canonical-position numbering is
#   not trivially defined for fused systems, we match the parent ring with a
#   SMARTS that pins atom order, then map matched ring carbons (those bearing
#   exactly one protium H) to the book δ.  Substituent effects are ignored
#   (base+flag).  Only the most abundant drug-like scaffolds are covered.
#
# Format: (name, SMARTS, {smarts_query_atom_idx: δ}).  Only query atoms that are
# carbons bearing a single ring H need a δ entry.  VERBATIM δ from p191-194.
# ───────────────────────────────────────────────────────────────────────────
FUSED_RINGS: list[tuple[str, str, dict[int, float]]] = [
    # Benzofuran (p191).  SMARTS atoms: 0=C2,1=C3,2=C3a,3=C4,4=C5,5=C6,6=C7,7=C7a,8=O1
    ("benzofuran", "c1cc2ccccc2o1",
     {0: 7.54, 1: 6.69, 3: 7.55, 4: 7.20, 5: 7.25, 6: 7.47}),
    # Benzothiophene (p191). 0=C2,1=C3,2=C3a,3=C4,4=C5,5=C6,6=C7,7=C7a,8=S1
    ("benzothiophene", "c1cc2ccccc2s1",
     {0: 7.42, 1: 7.33, 3: 7.82, 4: 7.36, 5: 7.33, 6: 7.88}),
    # Indole (p191). 0=C2,1=C3,2=C3a,3=C4,4=C5,5=C6,6=C7,7=C7a,8=N1
    ("indole", "c1cc2ccccc2[nH]1",
     {0: 7.05, 1: 6.52, 3: 7.64, 4: 7.12, 5: 7.18, 6: 7.27}),
    # Benzoxazole (p191). 0=C2,1=N3,2=C3a,3=C4,4=C5,5=C6,6=C7,7=C7a,8=O1
    ("benzoxazole", "c1nc2ccccc2o1",
     {0: 8.10, 3: 7.79, 4: 7.41, 5: 7.34, 6: 7.58}),
    # Benzothiazole (p191). 0=C2,1=N3,2=C3a,3=C4,4=C5,5=C6,6=C7,7=C7a,8=S1
    ("benzothiazole", "c1nc2ccccc2s1",
     {0: 8.97, 3: 7.94, 4: 7.46, 5: 7.51, 6: 8.14}),
    # Benzimidazole (p191). 0=C2,1=N3,2=C3a,3=C4,4=C5,5=C6,6=C7,7=C7a,8=N1
    ("benzimidazole", "c1nc2ccccc2[nH]1",
     {0: 8.08, 3: 7.70, 4: 7.26, 5: 7.26, 6: 7.70}),
    # Quinoline (p193, verified): H2 8.92, H3 7.39, H4 8.12, H5 7.82, H6 7.55,
    # H7 7.72, H8 8.15.  0=C2,1=C3,2=C4,3=C4a,4=C5,5=C6,6=C7,7=C8,8=C8a,9=N1
    ("quinoline", "c1ccc2ccccc2n1",
     {0: 8.92, 1: 7.39, 2: 8.12, 4: 7.82, 5: 7.55, 6: 7.72, 7: 8.15}),
    # Isoquinoline (p193): H1 9.22, H3 8.50 (both read on the page); benzo
    # standard. 0=C1,1=N2,2=C3,3=C4,4=C4a,5=C5,6=C6,7=C7,8=C8,9=C8a
    ("isoquinoline", "c1cc2ccccc2cn1",
     {0: 9.22, 2: 8.50, 3: 7.64, 5: 7.82, 6: 7.64, 7: 7.55, 8: 7.97}),
]


# ───────────────────────────────────────────────────────────────────────────
# Path 5 — Special / abundant functional-group singletons (small lookup).
#   Pretsch general ¹H ranges (Ch.5 overview).  These are assigned by SMARTS on
#   the H's parent atom and override the additive paths.
# ───────────────────────────────────────────────────────────────────────────
# (name, SMARTS, δ, h_query_atom): the H's are taken from the SMARTS atom at
# index `h_query_atom` in each match (the atom that actually carries the proton).
SPECIAL_GROUPS: list[tuple[str, str, float, int]] = [
    ("aldehyde_CHO", "[CX3H1](=O)[#6]", 9.7, 0),    # R–CHO   (H on the carbonyl C)
    ("formate_CHO",  "[CX3H1](=O)O",   8.05, 0),    # HCOO–R
    ("formamide_CHO","[CX3H1](=O)N",   8.1,  0),    # HCON<
    ("carboxylic_OH","[CX3](=O)[OX2H1]", 11.5, 2),  # –COOH (the acidic H on the OH oxygen)
    ("alcohol_OH",   "[OX2H1][CX4]",   2.0,  0),    # aliphatic –OH (broad, solvent dependent)
    ("phenol_OH",    "[OX2H1]c",       5.5,  0),    # Ar–OH
    ("amine_NH",     "[NX3;H1,H2][#6]", 1.5, 0),    # aliphatic amine N–H (broad)
    ("amide_NH",     "[NX3;H1,H2][CX3]=O", 7.0, 0), # amide N–H (broad)
]

# Coarse class defaults for the fallback path (uncovered protons).
CLASS_DEFAULTS = {
    "aromatic_CH": 7.30,
    "sp3_CH":      1.4,
    "sp2_CH":      5.8,
    "heteroatom_H": 3.0,
    "other":       2.0,
}

# Naphthalene base shifts (Pretsch p182): the four alpha positions (1,4,5,8 —
# peri to a ring-fusion carbon) = 7.84, the four beta positions (2,3,6,7) = 7.48.
NAPHTHALENE_ALPHA = 7.84
NAPHTHALENE_BETA = 7.48


def _fused_carbocyclic_aromatic_shift(mol: Chem.Mol, c_idx: int) -> float:
    """Naphthalene-type base δ for an H on a *fused* all-carbon aromatic 6-ring:
    7.84 if the carbon is adjacent (peri) to a ring-fusion carbon (alpha, e.g.
    naphthalene H1/4/5/8), else 7.48 (beta). Pretsch p182. Substituent
    increments are NOT applied (approximate — flagged), but this is far closer
    than the generic 7.30 aromatic fallback and generalises to the benzo rings
    of arbitrary fused systems not covered by a named scaffold."""
    ri = mol.GetRingInfo()
    a = mol.GetAtomWithIdx(c_idx)
    is_alpha = any(nb.GetIsAromatic() and nb.GetAtomicNum() == 6
                   and ri.NumAtomRings(nb.GetIdx()) > 1
                   for nb in a.GetNeighbors())
    return NAPHTHALENE_ALPHA if is_alpha else NAPHTHALENE_BETA


# ─── helpers ────────────────────────────────────────────────────────────────

def _is_protium(atom: Chem.Atom) -> bool:
    return atom.GetAtomicNum() == 1 and atom.GetIsotope() in (0, 1)


def _heavy_neighbor(h_atom: Chem.Atom) -> Chem.Atom | None:
    """The single heavy atom an H is bonded to (None if odd valence)."""
    heavy = [n for n in h_atom.GetNeighbors() if n.GetAtomicNum() > 1]
    return heavy[0] if len(heavy) == 1 else None


def _aryl_subst(mol: Chem.Mol, ring_carbon: Chem.Atom, ring: set[int]) -> str | None:
    """Classify the (single) non-ring substituent on a benzene ring carbon into
    an :data:`AROM_INCR` key.  Returns None if it is just an H (no substituent)
    or cannot be classified."""
    subs = [n for n in ring_carbon.GetNeighbors()
            if n.GetIdx() not in ring and n.GetAtomicNum() > 1]
    if not subs:
        return None
    s = subs[0]
    z = s.GetAtomicNum()

    if z == 9:
        return "F"
    if z == 17:
        return "Cl"
    if z == 35:
        return "Br"
    if z == 53:
        return "I"
    if z == 8:
        # –OH / –OCH3 / –O-phenyl / –OCO– …
        onbrs = [n for n in s.GetNeighbors() if n.GetIdx() != ring_carbon.GetIdx()]
        if not onbrs:
            return "OH"
        o2 = onbrs[0]
        if o2.GetAtomicNum() == 1:
            return "OH"
        if o2.GetIsAromatic():
            return "Ophenyl"
        if o2.GetAtomicNum() == 6:
            # ester O (–O–C(=O)–) ?
            if any(b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(o2).GetAtomicNum() == 8
                   for b in o2.GetBonds()):
                return "OCOCH3"
            return "OCH3"
        return "OCH3"
    if z == 16:
        # sulfur: thioether / sulfonyl / sulfonamide
        ndbl_o = sum(1 for b in s.GetBonds()
                     if b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(s).GetAtomicNum() == 8)
        if ndbl_o >= 2:
            if any(n.GetAtomicNum() == 7 for n in s.GetNeighbors()):
                return "SO2NH2"
            if any(n.GetAtomicNum() == 17 for n in s.GetNeighbors()):
                return "SO2Cl"
            return "SO2CH3"
        if ndbl_o == 1:
            return "SOCH3"
        if any(n.GetAtomicNum() == 1 for n in s.GetNeighbors()):
            return "SH"
        return "SCH3"
    if z == 7:
        ndbl = sum(1 for b in s.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)
        # nitro?
        o_nbrs = [n for n in s.GetNeighbors() if n.GetAtomicNum() == 8]
        if len(o_nbrs) == 2:
            return "NO2"
        # amide N (–NHC(=O)–)?
        for n in s.GetNeighbors():
            if n.GetAtomicNum() == 6 and any(
                b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(n).GetAtomicNum() == 8
                for b in n.GetBonds()
            ):
                return "NHCOCH3"
        nH = s.GetTotalNumHs()
        nC = sum(1 for n in s.GetNeighbors()
                 if n.GetAtomicNum() == 6 and n.GetIdx() != ring_carbon.GetIdx())
        if nH >= 2:
            return "NH2"
        if nC >= 2:
            return "N(CH3)2"
        return "NHCH3"
    if z == 6:
        # carbon substituent: distinguish carbonyl/nitrile/aromatic/alkyl
        if s.GetIsAromatic():
            return "phenyl"
        # nitrile?
        if any(b.GetBondType() == Chem.BondType.TRIPLE and b.GetOtherAtom(s).GetAtomicNum() == 7
               for b in s.GetBonds()):
            return "C#N"
        # alkyne?
        if any(b.GetBondType() == Chem.BondType.TRIPLE for b in s.GetBonds()):
            return "C#CH"
        # carbonyl carbon directly on ring?
        dbl_o = [b for b in s.GetBonds()
                 if b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(s).GetAtomicNum() == 8]
        if dbl_o:
            # CHO / COOH / COOR / CONH2 / COCl / ketone
            others = [n for n in s.GetNeighbors() if n.GetIdx() != ring_carbon.GetIdx()
                      and not (n.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(s.GetIdx(), n.GetIdx()).GetBondType() == Chem.BondType.DOUBLE)]
            if s.GetTotalNumHs() == 1:
                return "CHO"
            for n in others:
                if n.GetAtomicNum() == 8:
                    if n.GetTotalNumHs() == 1:
                        return "COOH"
                    return "COOCH3"
                if n.GetAtomicNum() == 7:
                    return "CONH2"
                if n.GetAtomicNum() == 17:
                    return "COCl"
            return "COCH3"
        # vinyl?
        if any(b.GetBondType() == Chem.BondType.DOUBLE for b in s.GetBonds()):
            return "CH=CH2"
        # CF3 / CCl3?
        f = sum(1 for n in s.GetNeighbors() if n.GetAtomicNum() == 9)
        cl = sum(1 for n in s.GetNeighbors() if n.GetAtomicNum() == 17)
        if f == 3:
            return "CF3"
        if cl == 3:
            return "CCl3"
        # CH2OH / CH2Cl?
        if any(n.GetAtomicNum() == 8 for n in s.GetNeighbors()):
            return "CH2OH"
        if any(n.GetAtomicNum() == 17 for n in s.GetNeighbors()):
            return "CH2Cl"
        # plain alkyl: count carbons on the substituent carbon
        cs = sum(1 for n in s.GetNeighbors()
                 if n.GetAtomicNum() == 6 and n.GetIdx() != ring_carbon.GetIdx())
        nh = s.GetTotalNumHs()
        if nh == 3:
            return "CH3"
        if cs == 1 and nh == 2:
            return "CH2CH3"
        if cs >= 2 and nh == 1:
            return "CH(CH3)2"
        if cs >= 3:
            return "C(CH3)3"
        return "CH3"  # default alkyl
    return None


def _alkane_subst(mol: Chem.Mol, carbon: Chem.Atom, exclude: int) -> str | None:
    """Classify a heavy neighbour of an sp3 carbon into an :data:`ALK_INCR`
    key.  `exclude` is an atom index to skip (e.g. the carbon we came from when
    walking to a β substituent).  Returns None if unclassifiable."""
    z = carbon.GetAtomicNum()
    # `carbon` here is actually the substituent atom on the proton-bearing C.
    if z == 9:
        return "F"
    if z == 17:
        return "Cl"
    if z == 35:
        return "Br"
    if z == 53:
        return "I"
    if z == 8:
        nbrs = [n for n in carbon.GetNeighbors() if n.GetIdx() != exclude]
        for n in nbrs:
            if n.GetAtomicNum() == 6 and any(
                b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(n).GetAtomicNum() == 8
                for b in n.GetBonds()):
                return "OCO"
            if n.GetIsAromatic():
                return "Ophenyl"
        if carbon.GetTotalNumHs() >= 1 and not nbrs:
            return "OH"
        # ester vs ether vs OH
        if all(n.GetAtomicNum() == 1 for n in nbrs):
            return "OH"
        return "OC"
    if z == 16:
        ndbl_o = sum(1 for b in carbon.GetBonds()
                     if b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(carbon).GetAtomicNum() == 8)
        if ndbl_o >= 2:
            return "SO2"
        if ndbl_o == 1:
            return "SO"
        return "S"
    if z == 7:
        o_nbrs = [n for n in carbon.GetNeighbors() if n.GetAtomicNum() == 8]
        if len(o_nbrs) == 2:
            return "NO2"
        for n in carbon.GetNeighbors():
            if n.GetAtomicNum() == 6 and any(
                b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(n).GetAtomicNum() == 8
                for b in n.GetBonds()):
                return "NCO"
        return "N"
    if z == 6:
        if carbon.GetIsAromatic():
            return "phenyl"
        if any(b.GetBondType() == Chem.BondType.TRIPLE and b.GetOtherAtom(carbon).GetAtomicNum() == 7
               for b in carbon.GetBonds()):
            return "C#N"
        if any(b.GetBondType() == Chem.BondType.TRIPLE for b in carbon.GetBonds()):
            return "C#C"
        dbl_o = [b for b in carbon.GetBonds()
                 if b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(carbon).GetAtomicNum() == 8]
        if dbl_o:
            if carbon.GetTotalNumHs() == 1:
                return "CHO"
            for n in carbon.GetNeighbors():
                if n.GetIdx() == exclude:
                    continue
                if n.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(carbon.GetIdx(), n.GetIdx()).GetBondType() != Chem.BondType.DOUBLE:
                    if n.GetTotalNumHs() == 1:
                        return "COOH"
                    return "COO"
                if n.GetAtomicNum() == 7:
                    return "CON"
                if n.GetAtomicNum() == 17:
                    return "COCl"
            return "CO"
        if any(b.GetBondType() == Chem.BondType.DOUBLE for b in carbon.GetBonds()):
            return "C=C"
        return "C"
    return None


def _ring_distance(mol: Chem.Mol, ring: list[int], i: int, j: int) -> int:
    """Shortest path length (in bonds) between ring atoms i and j, going around
    the ring only.  ortho=1, meta=2, para=3 for benzene."""
    n = len(ring)
    pos = {a: k for k, a in enumerate(ring)}
    a, b = pos[i], pos[j]
    d = abs(a - b)
    return min(d, n - d)


# ─── path implementations ────────────────────────────────────────────────────

def _benzene_shift(mol: Chem.Mol, h_idx: int, c_idx: int, ring: list[int]) -> float | None:
    """δ for a benzene ring proton via p178-179 increments, or None if a ring
    substituent could not be classified (so the caller can flag)."""
    delta = AROM_BASE
    rset = set(ring)
    ok = True
    for other in ring:
        if other == c_idx:
            continue
        oc = mol.GetAtomWithIdx(other)
        key = _aryl_subst(mol, oc, rset)
        if key is None:
            continue  # bare CH, contributes nothing
        incr = AROM_INCR.get(key)
        if incr is None:
            ok = False
            continue
        d = _ring_distance(mol, ring, c_idx, other)
        if d == 1:
            delta += incr[0]
        elif d == 2:
            delta += incr[1]
        elif d == 3:
            delta += incr[2]
    return delta if ok else None


def _alkane_shift(mol: Chem.Mol, h_idx: int, c_idx: int) -> float | None:
    """δ for an sp3 C–H via p160 additivity, or None if unclassifiable."""
    c = mol.GetAtomWithIdx(c_idx)
    heavy = [n for n in c.GetNeighbors() if n.GetAtomicNum() > 1]
    n_heavy = len(heavy)
    if n_heavy not in (1, 2, 3):
        return None
    col = n_heavy  # 1->CH3 col, 2->CH2, 3->CH
    col_idx = {1: 0, 2: 1, 3: 2}[col]
    delta = ALK_BASE[col]
    ok = True
    # α substituents: heavy atoms on this carbon
    for sub in heavy:
        key = _alkane_subst(mol, sub, c_idx)
        if key is None:
            ok = False
            continue
        cols = ALK_INCR.get(key)
        if cols is None:
            ok = False
            continue
        cell = cols[col_idx] or cols[1]   # blank CH column -> CH2 column (book footnote)
        za = cell[0]
        if za is None:
            za = cols[1][0]  # fall back to CH2 alpha
        delta += za
    # β substituents: a substituent counts as β only when it sits on a carbon
    # that is itself a *plain alkyl* α-substituent ("–C–").  If the α group is a
    # functional group in its own right (phenyl, C=C, C=O, –OC<, …) its α value
    # already encodes the whole group, so we must NOT walk into it again — doing
    # so double-counts (e.g. toluene's phenyl ring carbons leaking back as β).
    for sub in heavy:
        if sub.GetAtomicNum() != 6:
            continue
        if _alkane_subst(mol, sub, c_idx) != "C":
            continue  # α group is functional, already fully counted
        for b in sub.GetNeighbors():
            if b.GetIdx() == c_idx or b.GetAtomicNum() <= 1:
                continue
            key = _alkane_subst(mol, b, sub.GetIdx())
            if key is None or key == "C":
                continue  # plain C-C chain extension is not a β substituent
            cols = ALK_INCR.get(key)
            if cols is None:
                continue
            cell = cols[col_idx] or cols[1]
            zb = cell[1]
            if zb is None:
                zb = cols[1][1]
            if zb is not None:
                delta += zb
    return delta if ok else None


def _alkene_shift(mol: Chem.Mol, h_idx: int, c_idx: int) -> float | None:
    """δ for a vinylic C–H via p168-169 increments.  Identifies the C=C, then
    classifies each substituent on the two olefinic carbons by gem/cis/trans
    relative to this proton (using the existing 2-D/3-D geometry)."""
    c = mol.GetAtomWithIdx(c_idx)
    # find the double-bond partner
    partner = None
    for b in c.GetBonds():
        if b.GetBondType() == Chem.BondType.DOUBLE:
            o = b.GetOtherAtom(c)
            if o.GetAtomicNum() == 6:
                partner = o
                break
    if partner is None:
        return None
    delta = ALKENE_BASE
    ok = True

    # geminal substituents: on c itself (besides this H and the partner)
    for n in c.GetNeighbors():
        if n.GetIdx() == h_idx or n.GetIdx() == partner.GetIdx() or n.GetAtomicNum() <= 1:
            continue
        key = _alkene_classify(mol, n, c_idx)
        if key is None:
            ok = False
            continue
        delta += ALKENE_INCR[key][0]

    # substituents on the partner carbon: cis / trans by geometry
    other_subs = [n for n in partner.GetNeighbors()
                  if n.GetIdx() != c_idx and n.GetAtomicNum() > 1]
    for n in other_subs:
        key = _alkene_classify(mol, n, partner.GetIdx())
        if key is None:
            ok = False
            continue
        rel = _cis_trans(mol, h_idx, c_idx, partner.GetIdx(), n.GetIdx())
        idx = 1 if rel == "cis" else 2  # cis -> Zcis, trans -> Ztrans
        val = ALKENE_INCR[key][idx]
        if val is None:
            val = 0.0
        delta += val
    return delta if ok else None


def _alkene_classify(mol: Chem.Mol, sub: Chem.Atom, from_idx: int) -> str | None:
    z = sub.GetAtomicNum()
    if z == 9:
        return "F"
    if z == 17:
        return "Cl"
    if z == 35:
        return "Br"
    if z == 53:
        return "I"
    if z == 8:
        for n in sub.GetNeighbors():
            if n.GetIdx() == from_idx:
                continue
            if n.GetAtomicNum() == 6 and any(
                b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(n).GetAtomicNum() == 8
                for b in n.GetBonds()):
                return "OCO"
        return "OC"
    if z == 7:
        o_nbrs = [n for n in sub.GetNeighbors() if n.GetAtomicNum() == 8]
        if len(o_nbrs) == 2:
            return "NO2"
        for n in sub.GetNeighbors():
            if n.GetAtomicNum() == 6 and any(
                b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(n).GetAtomicNum() == 8
                for b in n.GetBonds()):
                return "NCO"
        return "N"
    if z == 16:
        return "S"
    if z == 6:
        if sub.GetIsAromatic():
            return "aromatic"
        if any(b.GetBondType() == Chem.BondType.TRIPLE and b.GetOtherAtom(sub).GetAtomicNum() == 7
               for b in sub.GetBonds()):
            return "C#N"
        dbl_o = [b for b in sub.GetBonds()
                 if b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(sub).GetAtomicNum() == 8]
        if dbl_o:
            if sub.GetTotalNumHs() == 1:
                return "CHO"
            for n in sub.GetNeighbors():
                if n.GetIdx() == from_idx:
                    continue
                if n.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(sub.GetIdx(), n.GetIdx()).GetBondType() != Chem.BondType.DOUBLE:
                    if n.GetTotalNumHs() == 1:
                        return "COOH"
                    return "COO"
                if n.GetAtomicNum() == 7:
                    return "CONH"
            return "CO"
        if any(b.GetBondType() == Chem.BondType.DOUBLE for b in sub.GetBonds()):
            return "C=C"
        return "alkyl"
    return None


def _cis_trans(mol: Chem.Mol, h_idx: int, c_idx: int, partner_idx: int, sub_idx: int) -> str:
    """Determine whether `sub` (on the partner carbon) is cis or trans to the
    proton across the C=C, using the 3-D conformer dihedral H–c=partner–sub.
    |dihedral| near 0 -> cis (same side); near 180 -> trans."""
    if mol.GetNumConformers() == 0:
        return "cis"  # no geometry -> assume cis (documented assumption)
    from rdkit.Chem import rdMolTransforms
    conf = mol.GetConformer()
    try:
        dih = rdMolTransforms.GetDihedralDeg(conf, h_idx, c_idx, partner_idx, sub_idx)
    except Exception:
        return "cis"
    return "cis" if abs(dih) < 90 else "trans"


def _hetero_shift(mol: Chem.Mol, h_idx: int, c_idx: int) -> tuple[float, bool] | None:
    """δ for a monocyclic heteroaromatic ring proton.  Returns (delta, flagged)
    where `flagged` marks rings whose base values were uncertain or substituted.
    Returns None if the ring is not a supported monocyclic heteroaromatic."""
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if c_idx not in ring:
            continue
        if not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        if any(ri.NumAtomRings(i) > 1 for i in ring):  # fused -> handled elsewhere
            continue
        size = len(ring)
        if size not in (5, 6):
            continue
        pos = _canonical_positions(mol, ring)
        if pos is None:
            continue
        het = tuple(sorted((pos[i], mol.GetAtomWithIdx(i).GetAtomicNum())
                           for i in ring if mol.GetAtomWithIdx(i).GetAtomicNum() != 6))
        name = _NAME.get((size, het))
        if name is None or name not in HETARO_BASE:
            continue
        base = HETARO_BASE[name].get(pos[c_idx])
        if base is None:
            continue
        uncertain_rings = {"oxazole", "isoxazole", "imidazole", "pyrazole"}
        # flag if substituted (ring carbon bears a non-H heavy substituent) too
        substituted = any(
            n.GetIdx() not in ring and n.GetAtomicNum() > 1
            for n in mol.GetAtomWithIdx(c_idx).GetNeighbors())
        return base, (name in uncertain_rings or substituted)
    return None


def _fused_shifts(mol: Chem.Mol) -> dict[int, tuple[float, bool]]:
    """δ for protons on supported fused heteroaromatic scaffolds (base+flag).
    Returns {h_idx: (delta, flagged=True)}."""
    out: dict[int, tuple[float, bool]] = {}
    for name, smarts, dmap in FUSED_RINGS:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        for match in mol.GetSubstructMatches(patt):
            for q_idx, atom_idx in enumerate(match):
                delta = dmap.get(q_idx)
                if delta is None:
                    continue
                a = mol.GetAtomWithIdx(atom_idx)
                hs = [n.GetIdx() for n in a.GetNeighbors() if _is_protium(n)]
                if len(hs) == 1:
                    out.setdefault(hs[0], (delta, True))
    return out


def _special_shifts(mol: Chem.Mol) -> dict[int, float]:
    """δ for special functional-group H's (aldehyde/COOH/OH/NH …) by SMARTS."""
    out: dict[int, float] = {}
    for _name, smarts, delta, h_q in SPECIAL_GROUPS:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        for match in mol.GetSubstructMatches(patt):
            head = mol.GetAtomWithIdx(match[h_q])
            for n in head.GetNeighbors():
                if _is_protium(n):
                    out.setdefault(n.GetIdx(), delta)
    return out


def _fallback(mol: Chem.Mol, h_idx: int, c_idx: int) -> float:
    a = mol.GetAtomWithIdx(c_idx)
    if a.GetAtomicNum() != 6:
        return CLASS_DEFAULTS["heteroatom_H"]
    if a.GetIsAromatic():
        return CLASS_DEFAULTS["aromatic_CH"]
    if any(b.GetBondType() == Chem.BondType.DOUBLE for b in a.GetBonds()):
        return CLASS_DEFAULTS["sp2_CH"]
    if a.GetHybridization() == Chem.HybridizationType.SP3:
        return CLASS_DEFAULTS["sp3_CH"]
    return CLASS_DEFAULTS["other"]


# ─── public API ──────────────────────────────────────────────────────────────

def predict_shifts_pretsch_verbose(mol: Chem.Mol) -> dict[int, tuple[float, str]]:
    """Predict ¹H shifts and report the originating path per proton.

    Returns ``{h_atom_idx: (delta_ppm, path_label)}`` where ``path_label`` is one
    of ``"special"``, ``"benzene"``, ``"hetero"``, ``"hetero?"`` (flagged),
    ``"fused?"``, ``"alkene"``, ``"alkane"``, or ``"fallback:<class>"``.
    Only protium H's appear.  Path precedence: special > fused/hetero > benzene
    > alkene > alkane > fallback.
    """
    result: dict[int, tuple[float, str]] = {}

    specials = _special_shifts(mol)
    fused = _fused_shifts(mol)

    benzene = Chem.MolFromSmarts("c1ccccc1")
    benzene_atoms: set[int] = set()
    benzene_rings: list[list[int]] = []
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if len(ring) == 6 and all(
            mol.GetAtomWithIdx(i).GetAtomicNum() == 6 and mol.GetAtomWithIdx(i).GetIsAromatic()
            for i in ring) and not any(ri.NumAtomRings(i) > 1 for i in ring):
            benzene_rings.append(list(ring))
            benzene_atoms.update(ring)

    # fused all-carbon aromatic 6-rings (naphthalene-type / benzo rings of fused
    # systems) not covered by a named scaffold -> naphthalene alpha/beta base δ.
    fused_carbo_atoms: set[int] = set()
    for ring in ri.AtomRings():
        if len(ring) == 6 and all(
            mol.GetAtomWithIdx(i).GetAtomicNum() == 6 and mol.GetAtomWithIdx(i).GetIsAromatic()
            for i in ring) and any(ri.NumAtomRings(i) > 1 for i in ring):
            fused_carbo_atoms.update(ring)

    for atom in mol.GetAtoms():
        if not _is_protium(atom):
            continue
        h_idx = atom.GetIdx()
        c = _heavy_neighbor(atom)
        if c is None:
            continue
        c_idx = c.GetIdx()

        # 1) special groups
        if h_idx in specials:
            result[h_idx] = (round(specials[h_idx], 3), "special")
            continue
        # 2) fused heteroaromatic
        if h_idx in fused:
            d, flagged = fused[h_idx]
            result[h_idx] = (round(d, 3), "fused?")
            continue
        # 3) monocyclic heteroaromatic
        het = _hetero_shift(mol, h_idx, c_idx)
        if het is not None:
            d, flagged = het
            result[h_idx] = (round(d, 3), "hetero?" if flagged else "hetero")
            continue
        # 4) benzene
        if c_idx in benzene_atoms and c.GetTotalNumHs() >= 0:
            ring = next(r for r in benzene_rings if c_idx in r)
            d = _benzene_shift(mol, h_idx, c_idx, ring)
            if d is not None:
                result[h_idx] = (round(d, 3), "benzene")
                continue
        # 4b) fused carbocyclic aromatic (naphthalene-type / fused benzo ring)
        if c_idx in fused_carbo_atoms:
            d = _fused_carbocyclic_aromatic_shift(mol, c_idx)
            result[h_idx] = (round(d, 3), "fused_arom?")
            continue
        # 5) alkene (sp2 carbon, C=C)
        if c.GetAtomicNum() == 6 and not c.GetIsAromatic() and \
           any(b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(c).GetAtomicNum() == 6
               for b in c.GetBonds()):
            d = _alkene_shift(mol, h_idx, c_idx)
            if d is not None:
                result[h_idx] = (round(d, 3), "alkene")
                continue
        # 6) alkane
        if c.GetAtomicNum() == 6 and c.GetHybridization() == Chem.HybridizationType.SP3:
            d = _alkane_shift(mol, h_idx, c_idx)
            if d is not None:
                result[h_idx] = (round(d, 3), "alkane")
                continue
        # 7) fallback
        d = _fallback(mol, h_idx, c_idx)
        cls = _fallback_class(mol, c_idx)
        result[h_idx] = (round(d, 3), f"fallback:{cls}")
    return result


def _fallback_class(mol: Chem.Mol, c_idx: int) -> str:
    a = mol.GetAtomWithIdx(c_idx)
    if a.GetAtomicNum() != 6:
        return "heteroatom_H"
    if a.GetIsAromatic():
        return "aromatic_CH"
    if any(b.GetBondType() == Chem.BondType.DOUBLE for b in a.GetBonds()):
        return "sp2_CH"
    if a.GetHybridization() == Chem.HybridizationType.SP3:
        return "sp3_CH"
    return "other"


def predict_shifts_pretsch(mol: Chem.Mol) -> dict[int, float]:
    """Predict ¹H chemical shifts (ppm) for every protium H in `mol`.

    Returns ``{h_atom_idx: delta_ppm}`` — a drop-in shape for the existing
    ``predict_shifts`` mapping (minus the min/max).  Uses additive increments
    transcribed from Pretsch (2009); see module docstring for paths and gaps.
    """
    return {idx: val for idx, (val, _path) in predict_shifts_pretsch_verbose(mol).items()}
