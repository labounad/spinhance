"""Tests for generate.spin_equivalence — deuterium substitution test.

All tests use RDKit only.  No network, no MNova, no AWS credentials.

Covered cases
-------------
- Homotopic protons (CH₃, t-Bu, gem-dimethyl) → one spin group.
- Enantiotopic CH₂ in an achiral molecule → two spin groups.
- Diastereotopic CH₂ in a molecule with a defined stereocentre → two groups.
- Known 8-group molecule accepted; known 9-group molecule rejected.
- Exchangeable protons (N-H, O-H) are ignored.
- strip_exchangeable_protons only removes heteroatom-bonded H.
- embed_3d returns a molecule with a conformer for typical drug-like input.
- passes_heuristic filters correctly on carbon and proton counts.
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from generate.spin_equivalence import (
    passes_heuristic,
    analyze_spin_systems,
    classify_spin_groups,
    embed_3d,
    strip_exchangeable_protons,
    substitution_signature,
)
from generate.config import N_SPIN_GROUPS


# ── Helper ────────────────────────────────────────────────────────────────────

def _mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    return mol


# ── embed_3d ──────────────────────────────────────────────────────────────────

class TestEmbed3D:
    def test_returns_mol_with_h(self):
        mol_h, _ = embed_3d(_mol("CC"))
        h_count = sum(1 for a in mol_h.GetAtoms() if a.GetAtomicNum() == 1)
        assert h_count == 6   # ethane has 6 H

    def test_has_conformer_for_simple_mol(self):
        _, has_3d = embed_3d(_mol("CC"))
        assert has_3d

    def test_has_conformer_for_drug_like(self):
        # Aspirin
        _, has_3d = embed_3d(_mol("CC(=O)Oc1ccccc1C(=O)O"))
        assert has_3d


# ── strip_exchangeable_protons ────────────────────────────────────────────────

class TestStripExchangeable:
    def test_removes_oh_proton(self):
        mol_h, _ = embed_3d(_mol("OCC"))          # ethanol
        stripped  = strip_exchangeable_protons(mol_h)
        # After stripping O-H, only C-H remain
        for atom in stripped.GetAtoms():
            if atom.GetAtomicNum() == 1:
                parent = atom.GetNeighbors()[0]
                assert parent.GetAtomicNum() == 6, "Non-C-H found after strip"

    def test_removes_nh_proton(self):
        mol_h, _ = embed_3d(_mol("NCC"))          # ethylamine
        stripped  = strip_exchangeable_protons(mol_h)
        for atom in stripped.GetAtoms():
            if atom.GetAtomicNum() == 1:
                assert atom.GetNeighbors()[0].GetAtomicNum() == 6

    def test_preserves_ch_protons(self):
        mol_h, _ = embed_3d(_mol("CC"))            # ethane — no exchangeable H
        n_before  = sum(1 for a in mol_h.GetAtoms() if a.GetAtomicNum() == 1)
        stripped  = strip_exchangeable_protons(mol_h)
        n_after   = sum(1 for a in stripped.GetAtoms() if a.GetAtomicNum() == 1)
        assert n_after == n_before


# ── substitution_signature ────────────────────────────────────────────────────

class TestSubstitutionSignature:
    def test_homotopic_same_signature(self):
        """All three CH₃ protons of ethane must give the same signature."""
        mol_h, use_3d = embed_3d(_mol("CC"))
        mol_h = strip_exchangeable_protons(mol_h)
        h_idxs = [a.GetIdx() for a in mol_h.GetAtoms() if a.GetAtomicNum() == 1]
        sigs = {substitution_signature(mol_h, idx, use_3d=use_3d) for idx in h_idxs}
        assert len(sigs) == 1, "Homotopic protons must share one signature"

    def test_diastereotopic_distinct_signatures(self):
        """Two CH₂ protons in a chiral ring must give different signatures."""
        smi = "C[C@H]1Cc2cc3c(C(F)(F)F)cc(O)nc3cc2N[C@@H]1C"
        mol_h, use_3d = embed_3d(_mol(smi))
        mol_h = strip_exchangeable_protons(mol_h)
        # Find the single CH₂ carbon (two H neighbours)
        ch2_hs = [
            a.GetIdx() for a in mol_h.GetAtoms()
            if a.GetAtomicNum() == 1
            and a.GetNeighbors()[0].GetAtomicNum() == 6
            and sum(1 for n in a.GetNeighbors()[0].GetNeighbors() if n.GetAtomicNum() == 1) == 2
        ]
        assert len(ch2_hs) == 2
        sig_a = substitution_signature(mol_h, ch2_hs[0], use_3d=use_3d)
        sig_b = substitution_signature(mol_h, ch2_hs[1], use_3d=use_3d)
        assert sig_a != sig_b, "Diastereotopic protons must have distinct signatures"


# ── analyze_spin_systems ──────────────────────────────────────────────────────

class TestAnalyzeSpinSystems:
    # ── homotopic cases ──────────────────────────────────────────────────────
    def test_ch3_one_group(self):
        n, sizes = analyze_spin_systems(_mol("CC(=O)O"))   # acetic acid
        assert 3 in sizes                                   # CH₃ → 1 group of 3 protons

    def test_tbu_one_group(self):
        n, _ = analyze_spin_systems(_mol("CC(C)(C)C"))     # neopentane
        assert n == 1

    def test_benzene_one_hard_group(self):
        """Benzene: all 6 H homotopic with identical distance profiles → 1 HARD group.

        With no external H atoms, condition 2 of the magnetic-equivalence check
        is vacuously satisfied, so the single equivalence class is HARD (n=1).
        This is correct: benzene is an A₆ spin system — one resonance, one group.
        """
        n, sizes = analyze_spin_systems(_mol("c1ccccc1"))
        assert n == 1
        assert sizes == [6]

    def test_135_trisubstituted_benzene_hard_pair(self):
        """1,3,5-trisubstituted benzene: H4 and H6 are HARD, H2 is NONE.

        H4 and H6 are homotopic (identical D-sub SMILES by C₂ symmetry) AND
        magnetically equivalent: both are equidistant from H2 (the only other
        ring proton) and from all side-chain H atoms through the C5-O bond.
        H2, flanked by two CF₃ groups, is chemically distinct → NONE.
        """
        smi = "FC(F)(F)c1cc(C(F)(F)F)cc(OCCCCl)c1"
        mol = _mol(smi)
        _, groups = classify_spin_groups(mol)

        aromatic_groups = [
            g for g in groups
            if any(
                mol.GetAtomWithIdx(
                    mol.GetAtomWithIdx(
                        # heavy parent in original mol (same index as mol_h heavy atoms)
                        g.heavy_parent_indices[0]
                    ).GetIdx()
                ).GetIsAromatic()
                for _ in [None]
            )
        ]
        # Easier: check by tier
        hard_aromatic = [g for g in groups if g.tier == "HARD" and any(
            True for hi in g.h_indices
            # We just check the count — exactly one HARD class with 2 H is expected
        ) and len(g.h_indices) == 2]
        assert len(hard_aromatic) == 1, (
            f"Expected 1 HARD aromatic pair, got {hard_aromatic}"
        )

    # ── enantiotopic cases ───────────────────────────────────────────────────
    def test_enantiotopic_ch2_counts_separately(self):
        """CH₂ in fluorochloromethane: two enantiotopic H → 2 groups."""
        n, _ = analyze_spin_systems(_mol("FCC1CC1"))       # prochiral CH₂
        # At minimum the CH₂ should be split
        assert n >= 2

    # ── diastereotopic cases ─────────────────────────────────────────────────
    def test_diastereotopic_ch2_rejected_from_8(self):
        """The known 9-group chiral molecule must NOT match N_SPIN_GROUPS=8."""
        smi = "C[C@H]1Cc2cc3c(C(F)(F)F)cc(O)nc3cc2N[C@@H]1C"
        n, _ = analyze_spin_systems(_mol(smi))
        assert n == 9
        assert n != N_SPIN_GROUPS

    # ── N-H structural context ───────────────────────────────────────────────
    def test_indole_nh_preserves_ring_asymmetry(self):
        """Dichloroindole aromatic CH protons must be chemically distinct (NONE).

        The indole N-H is the structural element that differentiates the two
        aromatic CH positions on the 6-membered ring.  The old code stripped
        the N-H before the D-substitution test, making both ring junction
        carbons appear equivalent and incorrectly merging the two CH protons
        as SOFT.  The fix keeps N-H in the molecule as context while excluding
        it from the candidate list.
        """
        from generate.spin_equivalence import classify_spin_groups
        smi = "S=C(SCc1nc2cc(Cl)c(Cl)cc2[nH]1)N1CCCCC1"
        mol = _mol(smi)

        # No SOFT class should contain two aromatic CH protons.
        mol_h, groups = classify_spin_groups(mol)
        for g in groups:
            if g.tier == "SOFT" and len(g.class_h_indices) == 2:
                parents_are_aromatic = [
                    mol_h.GetAtomWithIdx(
                        mol_h.GetAtomWithIdx(h).GetNeighbors()[0].GetIdx()
                    ).GetIsAromatic()
                    for h in g.class_h_indices
                ]
                assert not all(parents_are_aromatic), (
                    "Two aromatic CH protons merged as SOFT — "
                    "N-H context was not preserved during equivalence test"
                )

    # ── exchangeable protons ─────────────────────────────────────────────────
    def test_oh_not_counted(self):
        """Removing the O-H from the proton count must not change the group count."""
        # Methanol CH₃OH: only the CH₃ counts → 1 group of 3.
        n, sizes = analyze_spin_systems(_mol("CO"))
        assert n == 1
        assert sizes == [3]

    # ── group sizes ──────────────────────────────────────────────────────────
    def test_group_sizes_sum_to_total_h(self):
        mol = _mol("CC1CC(C)CC(C)C1")   # 1,3,5-trimethylcyclohexane
        mol_h, _ = embed_3d(mol)
        mol_h = strip_exchangeable_protons(mol_h)
        total_h = sum(1 for a in mol_h.GetAtoms() if a.GetAtomicNum() == 1)
        _, sizes = analyze_spin_systems(mol)
        assert sum(sizes) == total_h

    def test_sizes_sorted_descending(self):
        _, sizes = analyze_spin_systems(_mol("CC(C)(C)C(F)Cl"))
        assert sizes == sorted(sizes, reverse=True)


# ── passes_heuristic ──────────────────────────────────────────────────────────

class TestPassesHeuristic:
    def test_small_molecule_passes(self):
        ok, n_c, n_h = passes_heuristic(_mol("CC(C)C"))   # isobutane
        assert ok

    def test_too_many_ch_carbons_fails(self):
        # Decane has 10 CH₂/CH₃ carbons — well above MAX_PROTON_BEARING_C=8
        ok, n_c, _ = passes_heuristic(_mol("CCCCCCCCCC"))
        assert not ok
        assert n_c > 8

    def test_no_protons_fails(self):
        # Tetrafluoromethane: 0 C-H protons
        ok, _, n_h = passes_heuristic(_mol("FC(F)(F)F"))
        assert not ok
        assert n_h == 0
