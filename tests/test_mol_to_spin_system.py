import math
import shutil

import pytest

from mol_to_spin_system import shifts as shifts_mod
from mol_to_spin_system.aromatic import aromatic_couplings
from mol_to_spin_system.coupling import all_couplings
from mol_to_spin_system.geminal import _geminal_2j, geminal_couplings
from mol_to_spin_system.groups import degeneracies, proton_groups
from mol_to_spin_system.long_range import long_range_couplings
from mol_to_spin_system.olefinic import olefinic_couplings
from mol_to_spin_system.shifts import make_test_mol_3d
from mol_to_spin_system.vicinal import karplus, vicinal_couplings


# --- couplings (no Java needed) ---------------------------------------------

def _first_multi_h_carbon(smi):
    mol = make_test_mol_3d(smi)
    return next(
        a for a in mol.GetAtoms()
        if a.GetAtomicNum() == 6
        and sum(n.GetAtomicNum() == 1 for n in a.GetNeighbors()) >= 2
    )


def test_geminal_model_reference_values():
    # The additive 2J model reproduces Pretsch literature values (validated on
    # the canonical methyl probes).  These are intrinsic geminal 2J — used by
    # geminal_couplings only for true methylenes, never emitted for methyls.
    assert round(_geminal_2j(_first_multi_h_carbon("Cc1ccccc1")), 1) == -14.3  # toluene
    assert round(_geminal_2j(_first_multi_h_carbon("CC(C)=O")), 1) == -14.9    # acetone
    assert round(_geminal_2j(_first_multi_h_carbon("CC#N")), 1) == -16.9       # CH3CN


def test_geminal_multi_substituent_saturation():
    # B6: two electronegative / two pi substituents on the same carbon undershoot
    # under a purely linear additive model.  The saturating correction lands them
    # on the Pretsch anchors (CH2Cl2 -7.5, malononitrile -20.3) without disturbing
    # the single-substituent / single-pi cases above.
    assert round(_geminal_2j(_first_multi_h_carbon("ClCCl")), 1) == -7.5     # CH2Cl2
    assert round(_geminal_2j(_first_multi_h_carbon("N#CCC#N")), 1) == -20.3  # malononitrile


def test_geminal_olefinic_base():
    # B10: terminal =CH2 geminal 2J is +2.5 Hz (ethylene, Pretsch p.165), not +2.0.
    assert sorted(set(geminal_couplings(make_test_mol_3d("C=C")).values())) == [2.5]


def test_geminal_only_methylenes():
    # Methyl protons are magnetically equivalent (free rotation) → no
    # observable mutual coupling; geminal_couplings must skip CH3 (3 H).
    assert geminal_couplings(make_test_mol_3d("Cc1ccccc1")) == {}  # toluene CH3
    assert geminal_couplings(make_test_mol_3d("CC#N")) == {}       # CH3CN
    # True methylenes (2 H) still get exactly one geminal pair.
    assert set(geminal_couplings(make_test_mol_3d("ClCCl")).values())   # CH2Cl2
    assert len(geminal_couplings(make_test_mol_3d("N#CCC#N"))) == 1     # malononitrile CH2


def test_karplus_extremes():
    assert math.isclose(karplus(180), 9.2, abs_tol=0.05)  # anti
    assert math.isclose(karplus(0), 8.2, abs_tol=0.05)    # cis
    assert karplus(90) < 0                                 # ~ -0.3


def test_vicinal_rotatable():
    assert set(vicinal_couplings(make_test_mol_3d("CC")).values()) == {7.3}   # ethane
    assert set(vicinal_couplings(make_test_mol_3d("CCO")).values()) == {6.9}  # ethanol (Pretsch p.162)


def test_vicinal_sp2_sp2_diene():
    # B5: the =CH-CH= single bond between two olefinic sp2 carbons (1,3-butadiene
    # C2-C3) is NOT a freely-rotating sp3-sp3 bond; it takes the s-trans diene
    # 3J (~10.4 Hz, Pretsch p.166), not the ethane base (7.3 Hz).
    assert sorted(set(vicinal_couplings(make_test_mol_3d("C=CC=C")).values())) == [10.4]
    # sp3-sp3 vicinal couplings remain unchanged by the hybridization gate.
    assert set(vicinal_couplings(make_test_mol_3d("CC")).values()) == {7.3}   # ethane
    assert set(vicinal_couplings(make_test_mol_3d("CCC")).values()) == {7.3}  # propane


def test_vicinal_ring_karplus_range():
    js = list(vicinal_couplings(make_test_mol_3d("C1CCCCC1")).values())  # cyclohexane
    assert min(js) < 3.0 and max(js) > 8.0  # gauche..anti spread


def test_olefinic_cis_trans():
    # Unsubstituted ethylene: cis 11.6, trans 19.1 (Pretsch p.166).
    assert sorted(set(olefinic_couplings(make_test_mol_3d("C=C")).values())) == [11.6, 19.1]


def test_vicinal_substituent_anchors():
    # Pretsch vicinal substituent table (Tables of Spectral Data, 2009, p.162).
    def jset(smi):
        return set(vicinal_couplings(make_test_mol_3d(smi)).values())

    assert jset("CC") == {7.3}            # ethane (base)
    assert jset("CCF") == {6.9}           # CH3CH2F   6.9
    assert jset("CC(F)F") == {4.5}        # CH3CHF2   4.5
    assert jset("CCCl") == {7.2}          # CH3CH2Cl  7.2
    assert jset("CC(Cl)Cl") == {6.1}      # CH3CHCl2  6.1
    assert jset("CCO") == {6.9}           # CH3CH2OH  6.9


def test_olefinic_substituent_dependent():
    # Pretsch monosubstituted-ethylene couplings (p.166-167); cis from dihedral.
    def jset(smi):
        return sorted(set(olefinic_couplings(make_test_mol_3d(smi)).values()))

    assert jset("C=CC") == [10.0, 16.8]    # propene: alkyl
    assert jset("C=CF") == [4.7, 12.8]     # vinyl fluoride
    assert jset("C=CCl") == [7.5, 14.5]    # vinyl chloride
    assert jset("C=CBr") == [7.1, 14.9]    # vinyl bromide
    assert jset("C=CO") == [6.4, 14.0]     # vinyl alcohol (O)
    assert jset("C=CC=O") == [10.7, 17.6]  # acrolein (carbonyl C)
    assert jset("C=CC#N") == [11.3, 17.8]  # acrylonitrile (nitrile)


def test_couplings_skip_deuterium():
    # Only protium is emitted; deuterated positions produce no coupling entries.
    assert vicinal_couplings(make_test_mol_3d("[2H]C([2H])([2H])C([2H])([2H])[2H]")) == {}  # ethane-d6
    assert olefinic_couplings(make_test_mol_3d("[2H]C([2H])=C([2H])[2H]")) == {}            # ethylene-d4


def test_aromatic_ortho_meta_para():
    assert set(aromatic_couplings(make_test_mol_3d("c1ccccc1")).values()) == {7.5, 1.5, 0.7}


def test_heteroaromatic_ring_specific_couplings():
    from mol_to_spin_system.heteroaromatic import heteroaromatic_couplings
    from mol_to_spin_system.coupling import all_couplings
    # furan: J23=1.8, J34=3.4, J24=0.9, J25=1.5 (Pretsch). NOT benzene 7.5/1.5/0.7.
    assert sorted(set(heteroaromatic_couplings(make_test_mol_3d("c1ccoc1")).values())) == [0.9, 1.5, 1.8, 3.4]
    assert sorted(set(heteroaromatic_couplings(make_test_mol_3d("c1ccsc1")).values())) == [1.0, 2.8, 3.5, 4.8]  # thiophene
    assert sorted(set(heteroaromatic_couplings(make_test_mol_3d("c1cc[nH]c1")).values())) == [1.3, 2.1, 2.6, 3.5]  # pyrrole
    # pyridine: J23=6.0, J34=7.6, J24=1.9, J25=0.9, J26=0.4, J35=1.6
    assert sorted(set(heteroaromatic_couplings(make_test_mol_3d("c1ccncc1")).values())) == [0.4, 0.9, 1.6, 1.9, 6.0, 7.6]
    # diazines (canonical IUPAC numbering): pyridazine J34=4.9/J45=8.4/J35=2.0/J36=3.5
    assert sorted(set(heteroaromatic_couplings(make_test_mol_3d("c1ccnnc1")).values())) == [2.0, 3.5, 4.9, 8.4]
    assert sorted(set(heteroaromatic_couplings(make_test_mol_3d("c1cncnc1")).values())) == [1.5, 2.5, 5.0]  # pyrimidine
    # azoles
    assert sorted(set(heteroaromatic_couplings(make_test_mol_3d("c1ccno1")).values())) == [0.3, 1.7, 1.8]  # isoxazole
    assert sorted(set(heteroaromatic_couplings(make_test_mol_3d("c1cscn1")).values())) == [1.9, 3.2]        # thiazole
    assert sorted(set(heteroaromatic_couplings(make_test_mol_3d("c1cc[nH]n1")).values())) == [2.1]          # pyrazole
    # the combined estimator overrides benzene values on the pyridine ring (no 7.5)
    assert 7.5 not in set(all_couplings(make_test_mol_3d("c1ccncc1")).values())
    # carbocyclic benzene is untouched (still 7.5/1.5/0.7)
    assert heteroaromatic_couplings(make_test_mol_3d("c1ccccc1")) == {}
    # benzo-fused 5-ring H2-H3: indole 3.1, benzofuran 2.5, benzothiophene 5.5
    assert 3.1 in set(heteroaromatic_couplings(make_test_mol_3d("c1ccc2[nH]ccc2c1")).values())
    assert 2.5 in set(heteroaromatic_couplings(make_test_mol_3d("c1ccc2occc2c1")).values())
    assert 5.5 in set(heteroaromatic_couplings(make_test_mol_3d("c1ccc2sccc2c1")).values())
    # benzimidazole (no intra-5-ring H pair) and naphthalene (carbocyclic) get nothing here
    assert heteroaromatic_couplings(make_test_mol_3d("c1ccc2[nH]cnc2c1")) == {}
    assert heteroaromatic_couplings(make_test_mol_3d("c1ccc2ccccc2c1")) == {}


def test_long_range_allylic():
    assert set(long_range_couplings(make_test_mol_3d("C=CC")).values()) == {-1.3}  # propene
    assert long_range_couplings(make_test_mol_3d("CCC")) == {}                     # propane


def test_all_couplings_no_overlap():
    mol = make_test_mol_3d("C=CC")
    merged = all_couplings(mol)
    individual = sum(
        len(fn(mol))
        for fn in (
            geminal_couplings,
            vicinal_couplings,
            olefinic_couplings,
            aromatic_couplings,
            long_range_couplings,
        )
    )
    assert len(merged) == individual  # no key collisions, nothing dropped


def test_proton_groups():
    groups, _ = proton_groups(make_test_mol_3d("CCO"))
    assert sorted(degeneracies(groups)) == [2, 3]
    groups, _ = proton_groups(make_test_mol_3d("c1ccccc1"))
    assert degeneracies(groups) == [6]


def test_proton_groups_excludes_deuterium():
    # Deuterium (2H) is NMR-invisible: benzene-d6 has no proton groups, and a
    # partially deuterated toluene counts only the CH3 (the ring D is ignored).
    groups, _ = proton_groups(make_test_mol_3d("[2H]c1c([2H])c([2H])c([2H])c([2H])c1[2H]"))
    assert groups == []
    groups, _ = proton_groups(make_test_mol_3d("Cc1c([2H])c([2H])c([2H])c([2H])c1[2H]"))
    assert degeneracies(groups) == [3]  # only the methyl


def test_proton_groups_split_diastereotopic():
    """Diastereotopic protons are chemically INEQUIVALENT and must NOT be merged
    (regression: the old CanonicalRankAtoms grouping collapsed them). The geminal
    pair must also become a real inter-group coupling."""
    from mol_to_spin_system.matrix import build_spin_system

    # 3,3-disubstituted azetidine (CH3 != OH on C3 -> the ring CH2 protons are
    # diastereotopic): expect CH3(3) + two CH2 groups (2,2) + 3 aromatic(1,1,1),
    # NOT a single deg-4 CH2 group.
    sys_az = build_spin_system(make_test_mol_3d("CC1(CN(C1)C2=C(C=CC=N2)C(F)(F)F)O"))
    assert sorted(sys_az.degeneracy.tolist()) == [1, 1, 1, 2, 2, 3]
    # the two deg-2 CH2 groups must carry a geminal 2J (~ -8..-15 Hz), not be a singlet
    deg2 = [i for i, d in enumerate(sys_az.degeneracy) if d == 2]
    assert len(deg2) == 2 and sys_az.matrix[deg2[0], deg2[1]] < -5.0

    # CH2's flanking a defined stereocentre are diastereotopic -> every CH2 proton split
    sys_ch = build_spin_system(make_test_mol_3d("C1=CC(=C(C=C1Br)[C@@H](CCN)N)O"))
    assert sys_ch.degeneracy.tolist() == [1] * 8


def test_equiv_orbit_topicity_for_overdispersion():
    """The over-dispersion shares a shift draw only WITHIN a topicity orbit, so the
    orbit key must separate diastereotopic protons (independent shifts) while uniting
    enantiotopic ones. Regression for the equiv_orbit bug: it used CanonicalRankAtoms
    (constitutional), collapsing same-carbon diastereotopic protons into one orbit and
    forcing Delta-delta = 0."""
    from mol_to_spin_system.groups import _substitution_key

    mol = make_test_mol_3d("CC1(CN(C1)C2=C(C=CC=N2)C(F)(F)F)O")  # 3,3-disub azetidine
    # the two H's on a single ring CH2 carbon are diastereotopic -> distinct keys
    ch2_carbons = [a.GetIdx() for a in mol.GetAtoms()
                   if a.GetAtomicNum() == 6 and not a.GetIsAromatic()
                   and sum(n.GetAtomicNum() == 1 for n in a.GetNeighbors()) == 2]
    assert ch2_carbons
    found_diastereotopic = False
    for c in ch2_carbons:
        hs = [n.GetIdx() for n in mol.GetAtomWithIdx(c).GetNeighbors() if n.GetAtomicNum() == 1]
        if len(hs) == 2 and _substitution_key(mol, hs[0]) != _substitution_key(mol, hs[1]):
            found_diastereotopic = True
    assert found_diastereotopic, "same-carbon diastereotopic protons must get distinct orbit keys"


def test_proton_groups_keep_enantiotopic():
    """Enantiotopic / homotopic protons ARE equivalent in an achiral solvent and must
    stay merged (guard against the diastereotopic fix over-splitting)."""
    # ethanol CH2 is enantiotopic -> stays deg 2
    assert sorted(degeneracies(proton_groups(make_test_mol_3d("CCO"))[0])) == [2, 3]
    # isopropanol's two methyls are enantiotopic -> one deg-6 group, not two deg-3
    assert sorted(degeneracies(proton_groups(make_test_mol_3d("CC(O)C"))[0])) == [1, 6]


# --- end-to-end (needs Java + nmrshiftdb predictor) -------------------------

def _predictor_available() -> bool:
    return (
        shutil.which("java") is not None
        and (shifts_mod._SNAPSHOTS / "predictorh.jar").exists()
    )


requires_predictor = pytest.mark.skipif(
    not _predictor_available(), reason="nmrshiftdb predictor / java not available"
)


def test_matrix_ethanol():
    # build_spin_system is now Java-free (Pretsch shift engine), so this runs
    # without the nmrshiftdb predictor.
    from mol_to_spin_system.matrix import build_spin_system

    system = build_spin_system(make_test_mol_3d("CCO"))
    assert system.n_groups == 2
    assert system.degeneracy.tolist() == [3, 2]
    assert system.matrix[0, 1] == system.matrix[1, 0]  # symmetric
    assert system.matrix[0, 1] > 5.0                    # ~6.9 Hz CH3-CH2 (vicinal)
    assert 0.0 < system.matrix[0, 0] < 10.0             # plausible 1H shift
    assert system.pack().shape == (8, 9)


@requires_predictor
def test_shifts_ethanol_methyl():
    from mol_to_spin_system.shifts import predict_shifts

    means = [v["mean"] for v in predict_shifts(make_test_mol_3d("CCO"), nucleus="H").values()]
    assert any(abs(m - 1.2) < 0.5 for m in means)  # methyl near 1.2 ppm


# --- Audit-2: per-group symmetry orbit (soft-equivalence signal) -------------

def test_entry_to_spin_system_emits_symmetry_equiv_orbit():
    """entry_to_spin_system must tag each spin group with its canonical symmetry orbit
    (Chem.CanonicalRankAtoms, breakTies=False) so chemically-equivalent-but-distinct groups
    share an id. This is the Audit-2 linchpin: the original bug keyed equivalence off a
    (shift, range) proxy, forcing accidental collisions to merge. Ethyl benzoate's
    monosubstituted ring gives three symmetry-equivalent group pairs among its 8 groups."""
    import os
    import tempfile
    from collections import Counter

    from generate.xyz_writer import molecule_to_xyz
    from mol_to_spin_system.xyz import entry_to_spin_system, iter_xyz_entries

    block = molecule_to_xyz("CCOC(=O)c1ccccc1", chembl_id="TEST", inchikey="TESTKEY")
    assert block, "expected an 8-group labelled XYZ block"
    with tempfile.NamedTemporaryFile("w", suffix=".xyz", delete=False) as f:
        f.write(block)
        path = f.name
    try:
        comment, atoms = next(iter_xyz_entries(path))
    finally:
        os.unlink(path)

    d = entry_to_spin_system(comment, atoms).to_dict()
    orb = d.get("equiv_orbit")
    assert orb is not None and len(orb) == len(d["labels"]) == 8   # one orbit id per group
    assert all(isinstance(o, int) for o in orb)                    # serialized as ints
    # The symmetric ring yields soft-equiv pairs: orbit-class sizes are robust to RDKit's
    # internal rank numbers (a chemical fact), unlike the raw ids. Three pairs + two singletons.
    assert sorted(Counter(orb).values()) == [1, 1, 2, 2, 2]
