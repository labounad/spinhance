from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem

from mol_to_spin_system.coupling import all_couplings
from mol_to_spin_system.groups import degeneracies, proton_groups
from mol_to_spin_system.shifts import DEFAULT_SOLVENT, predict_shifts


@dataclass
class SpinSystem:
    """A molecule's proton spin system.

    matrix: symmetric NxN with chemical shifts (ppm) on the diagonal and
            inter-group couplings (Hz) off-diagonal.
    degeneracy: protons per spin group.
    groups: H atom indices making up each group.
    """

    matrix: np.ndarray
    degeneracy: np.ndarray
    groups: list[list[int]]

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    def pack(self, n: int = 8) -> np.ndarray:
        """Pack into the n x (n+1) block: n x n matrix + degeneracy column."""
        if self.n_groups > n:
            raise ValueError(f"{self.n_groups} spin groups exceeds capacity {n}")
        out = np.zeros((n, n + 1))
        k = self.n_groups
        out[:k, :k] = self.matrix
        out[:k, n] = self.degeneracy
        return out


def build_spin_system(mol: Chem.Mol, solvent: str = DEFAULT_SOLVENT) -> SpinSystem:
    """Assemble the proton spin-system matrix for a 3D-embedded molecule."""
    groups, group_of_atom = proton_groups(mol)
    n = len(groups)

    shifts = predict_shifts(mol, nucleus="H", solvent=solvent)
    coups = all_couplings(mol)

    matrix = np.zeros((n, n))

    # diagonal: group-averaged chemical shift
    for gi, atoms in enumerate(groups):
        vals = [shifts[a]["mean"] for a in atoms if a in shifts]
        matrix[gi, gi] = round(float(np.mean(vals)), 2) if vals else 0.0

    # off-diagonal: inter-group coupling, averaged over contributing atom pairs
    sums: dict[tuple[int, int], float] = {}
    counts: dict[tuple[int, int], int] = {}
    for (i, j), jval in coups.items():
        gi, gj = group_of_atom.get(i), group_of_atom.get(j)
        if gi is None or gj is None or gi == gj:
            continue
        key = (min(gi, gj), max(gi, gj))
        sums[key] = sums.get(key, 0.0) + jval
        counts[key] = counts.get(key, 0) + 1
    for (gi, gj), total in sums.items():
        avg = round(total / counts[(gi, gj)], 1)
        matrix[gi, gj] = matrix[gj, gi] = avg

    return SpinSystem(matrix, np.array(degeneracies(groups)), groups)


def save_spin_system(system: SpinSystem, stem: str | Path) -> tuple[Path, Path]:
    """Write the spin system as <stem>.npy (packed 8x9) and <stem>.json."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    npy_path = stem.with_suffix(".npy")
    json_path = stem.with_suffix(".json")

    np.save(npy_path, system.pack())
    json_path.write_text(
        json.dumps(
            {
                "matrix": system.matrix.tolist(),
                "degeneracy": system.degeneracy.tolist(),
                "groups": system.groups,
            },
            indent=2,
        )
    )
    return npy_path, json_path


if __name__ == "__main__":
    from mol_to_spin_system.shifts import make_test_mol_3d

    mol = make_test_mol_3d("CCO")
    system = build_spin_system(mol)
    print("ethanol spin system:")
    print(f"  groups: {system.groups}")
    print(f"  degeneracy: {system.degeneracy.tolist()}")
    print("  matrix (diag=shift ppm, off-diag=J Hz):")
    print(system.matrix)
