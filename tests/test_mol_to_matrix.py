import math
import shutil

import pytest

from mol_to_matrix import shifts as shifts_mod
from mol_to_matrix.aromatic import aromatic_couplings
from mol_to_matrix.coupling import all_couplings
from mol_to_matrix.geminal import geminal_couplings
from mol_to_matrix.groups import degeneracies, proton_groups
from mol_to_matrix.long_range import long_range_couplings
from mol_to_matrix.olefinic import olefinic_couplings
from mol_to_matrix.shifts import make_test_mol_3d
from mol_to_matrix.vicinal import karplus, vicinal_couplings


# --- couplings (no Java needed) ---------------------------------------------

def test_geminal_reference_values():
    assert -14.3 in set(geminal_couplings(make_test_mol_3d("Cc1ccccc1")).values())  # toluene
    assert -14.9 in set(geminal_couplings(make_test_mol_3d("CC(C)=O")).values())   # acetone
    assert -16.9 in set(geminal_couplings(make_test_mol_3d("CC#N")).values())      # CH3CN


def test_karplus_extremes():
    assert math.isclose(karplus(180), 9.2, abs_tol=0.05)  # anti
    assert math.isclose(karplus(0), 8.2, abs_tol=0.05)    # cis
    assert karplus(90) < 0                                 # ~ -0.3


def test_vicinal_rotatable():
    assert set(vicinal_couplings(make_test_mol_3d("CC")).values()) == {7.3}   # ethane
    assert set(vicinal_couplings(make_test_mol_3d("CCO")).values()) == {6.8}  # ethanol


def test_vicinal_ring_karplus_range():
    js = list(vicinal_couplings(make_test_mol_3d("C1CCCCC1")).values())  # cyclohexane
    assert min(js) < 3.0 and max(js) > 8.0  # gauche..anti spread


def test_olefinic_cis_trans():
    assert sorted(set(olefinic_couplings(make_test_mol_3d("C=C")).values())) == [11.0, 17.0]


def test_aromatic_ortho_meta_para():
    assert set(aromatic_couplings(make_test_mol_3d("c1ccccc1")).values()) == {7.5, 1.5, 0.7}


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


# --- end-to-end (needs Java + nmrshiftdb predictor) -------------------------

def _predictor_available() -> bool:
    return (
        shutil.which("java") is not None
        and (shifts_mod._SNAPSHOTS / "predictorh.jar").exists()
    )


requires_predictor = pytest.mark.skipif(
    not _predictor_available(), reason="nmrshiftdb predictor / java not available"
)


@requires_predictor
def test_matrix_ethanol():
    from mol_to_matrix.matrix import build_spin_system

    system = build_spin_system(make_test_mol_3d("CCO"))
    assert system.n_groups == 2
    assert system.degeneracy.tolist() == [3, 2]
    assert system.matrix[0, 1] == system.matrix[1, 0]  # symmetric
    assert system.matrix[0, 1] > 5.0                    # ~6.8 Hz CH3-CH2
    assert 0.0 < system.matrix[0, 0] < 10.0             # plausible 1H shift
    assert system.pack().shape == (8, 9)


@requires_predictor
def test_shifts_ethanol_methyl():
    from mol_to_matrix.shifts import predict_shifts

    means = [v["mean"] for v in predict_shifts(make_test_mol_3d("CCO"), nucleus="H").values()]
    assert any(abs(m - 1.2) < 0.5 for m in means)  # methyl near 1.2 ppm
