from __future__ import annotations

import gzip
import json
import multiprocessing as mp
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from mol_to_matrix.coupling import all_couplings
from mol_to_matrix.shifts import DEFAULT_SOLVENT, predict_shifts

# atom record from a parsed XYZ block: (symbol, x, y, z, group_label, tier_class)
Atom = tuple[str, float, float, float, "str | None", "str | None"]

_MAX_BOND_LEN = 2.5  # A; longer bonded distance => atom-index mismatch


@dataclass
class LabeledSpinSystem:
    """Spin-system data for one labelled molecule, keyed by group label.

    labels: spin-group labels (A, B, ...), sorted.
    shifts: (n_groups, 2) array of [chemical shift (ppm), number of H].
    couplings: list of (label_i, label_j, J_Hz) for distinct group pairs.
    meta: the molecule's JSON comment (smiles, chembl_id, ...).
    """

    labels: list[str]
    shifts: np.ndarray
    couplings: list[tuple[str, str, float]]
    meta: dict

    def to_dict(self) -> dict:
        """JSON-serializable record for this molecule."""
        return {
            "chembl_id": self.meta.get("chembl_id"),
            "smiles": self.meta.get("smiles"),
            "inchikey": self.meta.get("inchikey"),
            "labels": self.labels,
            "spin_groups": [[round(float(d), 2), int(n)] for d, n in self.shifts],
            "couplings": [[gi, gj, jval] for gi, gj, jval in self.couplings],
        }


def iter_xyz_entries(path: str | Path):
    """Yield (comment_dict, atoms) per molecule from a multi-XYZ(.gz) file."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        while True:
            count_line = fh.readline()
            if not count_line:
                break
            if not count_line.strip():
                continue
            n = int(count_line.strip())
            comment = json.loads(fh.readline())
            atoms: list[Atom] = []
            for _ in range(n):
                parts = fh.readline().split()
                atoms.append(
                    (
                        parts[0],
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                        parts[4] if len(parts) > 4 else None,
                        parts[5] if len(parts) > 5 else None,
                    )
                )
            yield comment, atoms


def _build_mol(comment: dict, atoms: list[Atom]) -> Chem.Mol:
    """RDKit mol from the SMILES with the XYZ geometry attached by atom index."""
    smiles = comment["smiles"]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"could not parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    if mol.GetNumAtoms() != len(atoms):
        raise ValueError(f"atom count mismatch: mol {mol.GetNumAtoms()} vs xyz {len(atoms)}")

    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (sym, x, y, z, _g, _t) in enumerate(atoms):
        if mol.GetAtomWithIdx(i).GetSymbol() != sym:
            raise ValueError(
                f"atom order mismatch at {i}: mol {mol.GetAtomWithIdx(i).GetSymbol()} vs xyz {sym}"
            )
        conf.SetAtomPosition(i, Point3D(x, y, z))
    conf.Set3D(True)
    mol.AddConformer(conf, assignId=True)

    # sanity check: a wrong index mapping shows up as an over-long bond
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if conf.GetAtomPosition(a).Distance(conf.GetAtomPosition(b)) > _MAX_BOND_LEN:
            raise ValueError(f"bond {a}-{b} too long; XYZ atom order likely mismatched")
    return mol


def _class_key(tier: str, atom_idx: int):
    """Chemical-equivalence class: shared int for H{n}/S{n}, unique for N."""
    m = re.fullmatch(r"[HS](\d+)", tier)
    if m:
        return ("class", int(m.group(1)))
    return ("singleton", atom_idx)  # N (or unrecognized) -> its own class


def entry_to_spin_system(
    comment: dict,
    atoms: list[Atom],
    solvent: str = DEFAULT_SOLVENT,
) -> LabeledSpinSystem:
    """Convert one labelled XYZ entry into per-spin-group shifts + couplings.

    - Protons sharing a `tier` class number get one averaged chemical shift.
    - Each `group_label` is one spin group; its shift is the (tier-averaged)
      value and its degeneracy is the proton count.
    - Coupling between two groups is averaged over their contributing H pairs.
    """
    mol = _build_mol(comment, atoms)

    # The predictor's 3D stereo-aware HOSE path can crash on some stereocentres
    # (needs 2D coords); fall back to the non-3D prediction in that case.
    try:
        raw_shifts = predict_shifts(mol, "H", solvent, use_3d=True)
    except RuntimeError:
        raw_shifts = predict_shifts(mol, "H", solvent, use_3d=False)
    per_atom_shift = {i: v["mean"] for i, v in raw_shifts.items()}
    couplings = all_couplings(mol)

    # labelled (non-exchangeable) protons only
    group_of = {i: a[4] for i, a in enumerate(atoms) if a[4] is not None}
    class_of = {i: _class_key(atoms[i][5], i) for i in group_of}

    # tier averaging: one shift per chemical-equivalence class
    class_atoms: dict[tuple, list[int]] = {}
    for i, c in class_of.items():
        class_atoms.setdefault(c, []).append(i)
    class_shift = {
        c: float(np.mean([per_atom_shift[i] for i in idxs if i in per_atom_shift]))
        for c, idxs in class_atoms.items()
    }

    # spin groups, sorted Excel-style (A..Z, AA..)
    group_atoms: dict[str, list[int]] = {}
    for i, g in group_of.items():
        group_atoms.setdefault(g, []).append(i)
    labels = sorted(group_atoms, key=lambda s: (len(s), s))

    shifts = np.zeros((len(labels), 2))
    for row, g in enumerate(labels):
        members = group_atoms[g]
        delta = float(np.mean([class_shift[class_of[i]] for i in members]))
        shifts[row] = [round(delta, 2), len(members)]

    # group averaging for couplings (distinct groups only)
    sums: dict[tuple[str, str], float] = {}
    counts: dict[tuple[str, str], int] = {}
    for (i, j), jval in couplings.items():
        gi, gj = group_of.get(i), group_of.get(j)
        if gi is None or gj is None or gi == gj:
            continue
        key = (gi, gj) if (len(gi), gi) <= (len(gj), gj) else (gj, gi)
        sums[key] = sums.get(key, 0.0) + jval
        counts[key] = counts.get(key, 0) + 1
    coupling_list = [
        (gi, gj, round(sums[(gi, gj)] / counts[(gi, gj)], 1))
        for (gi, gj) in sorted(sums, key=lambda k: ((len(k[0]), k[0]), (len(k[1]), k[1])))
    ]

    return LabeledSpinSystem(labels, shifts, coupling_list, comment)


def _convert_one(args: tuple[dict, list, str]) -> dict | None:
    """Worker: convert one (comment, atoms, solvent) entry, None on failure."""
    comment, atoms, solvent = args
    try:
        return entry_to_spin_system(comment, atoms, solvent).to_dict()
    except Exception:  # noqa: BLE001 - skip bad entries, keep going
        return None


def convert_file(
    in_path: str | Path,
    out_path: str | Path,
    solvent: str = DEFAULT_SOLVENT,
    limit: int | None = None,
    workers: int = 1,
) -> dict[str, int]:
    """Convert a multi-XYZ file into one big JSON array on disk.

    Streams entries so neither file is held fully in memory. Entries that fail
    to convert are skipped. `limit` caps the number of entries read; `workers`
    sets the number of parallel processes (each runs its own predictor JVM).
    Output order follows the input. Returns {ok, skipped}.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_skipped = 0

    def _entries():
        for count, (comment, atoms) in enumerate(iter_xyz_entries(in_path)):
            if limit is not None and count >= limit:
                break
            yield (comment, atoms, solvent)

    with open(out_path, "w") as out:
        out.write("[\n")
        if workers > 1:
            with mp.Pool(workers) as pool:
                records = pool.imap(_convert_one, _entries(), chunksize=1)
                for record in records:
                    if record is None:
                        n_skipped += 1
                        continue
                    out.write((",\n" if n_ok else "") + json.dumps(record))
                    n_ok += 1
        else:
            for args in _entries():
                record = _convert_one(args)
                if record is None:
                    n_skipped += 1
                    continue
                out.write((",\n" if n_ok else "") + json.dumps(record))
                n_ok += 1
        out.write("\n]\n")
    return {"ok": n_ok, "skipped": n_skipped}


if __name__ == "__main__":
    path = Path(__file__).resolve().parents[1] / "generate" / "data" / "8spin.xyz"
    comment, atoms = next(iter_xyz_entries(path))
    system = entry_to_spin_system(comment, atoms)
    print(json.dumps(system.to_dict(), indent=2))
