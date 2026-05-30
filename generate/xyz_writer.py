"""generate/xyz_writer.py — write spin-annotated XYZ files from screened molecules.

Converts every molecule in ``8spin.csv`` to an entry in a single gzip-compressed
multi-XYZ file (``8spin.xyz.gz``).  The multi-XYZ format concatenates individual
XYZ blocks back-to-back; standard tools (ASE, OpenBabel, MDAnalysis) can read it
transparently.

Compression rationale
---------------------
100,000 molecules × ~25 atoms × ~45 chars/line ≈ 100 MB uncompressed.
Floating-point coordinate text compresses at roughly 8–10×, yielding a
~10–12 MB single file with no external dependencies (Python's ``gzip`` stdlib).
A directory of 100,000 individual files would cause filesystem overhead on
NTFS/WSL2 and is impractical for downstream tooling.

File format (one block per molecule)
-------------------------------------
::

    <n_atoms>
    {"smiles":"...","inchikey":"...","chembl_id":"...","inchi":"..."}
    C   x.xxxxxx   y.yyyyyy   z.zzzzzz
    H   x.xxxxxx   y.yyyyyy   z.zzzzzz   A  H1
    H   x.xxxxxx   y.yyyyyy   z.zzzzzz   A  H1
    H   x.xxxxxx   y.yyyyyy   z.zzzzzz   B  S2
    H   x.xxxxxx   y.yyyyyy   z.zzzzzz   C  S2
    H   x.xxxxxx   y.yyyyyy   z.zzzzzz   D  N

Annotation columns (H atoms only)
----------------------------------
``{group_letter}``
    Excel-style spin-group label (A, B, C …).  Each spin group gets one letter.

``{tier}{class_number}``
    Tier character:

    * ``H`` — HARD: homotopic and magnetically equivalent (e.g. methyl rotor).
      All N protons in the group share one label and one class number.
    * ``S`` — SOFT: chemically equivalent but magnetically inequivalent
      (e.g. enantiotopic methylene, AA′BB′ aromatic pair).  All SOFT protons
      sharing the same averaged chemical shift carry the **same class number**,
      making their shift relationship explicit.
    * ``N`` — NONE: chemically distinct singleton.  No class number appended.

    HARD and SOFT groups with the same class number share one averaged δ in the
    NMR simulation.

Exchangeable protons (N-H, O-H, S-H) are included in the 3-D geometry but
carry no annotation — they are invisible in solution-state ¹H NMR.

Running
-------
::

    python generate/cli.py xyz                          # 8spin.csv → 8spin.xyz.gz
    python generate/cli.py xyz --input  /path/to/8spin.csv \\
                                --output /path/to/8spin.xyz.gz
"""

from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_REPO_ROOT     = Path(__file__).resolve().parent.parent
DEFAULT_INPUT  = _REPO_ROOT / "generate" / "data" / "8spin.csv"
DEFAULT_OUTPUT = _REPO_ROOT / "generate" / "data" / "8spin.xyz.gz"


# ── Per-molecule XYZ block ────────────────────────────────────────────────────

def molecule_to_xyz(
    smiles: str,
    *,
    chembl_id: str = "",
    inchikey:  str = "",
) -> str | None:
    """Return an annotated XYZ block for *smiles*, or ``None`` on failure.

    Parameters
    ----------
    smiles:
        Canonical SMILES of the molecule.
    chembl_id:
        ChEMBL identifier embedded in the comment line.
    inchikey:
        Standard InChIKey embedded in the comment line.

    Returns
    -------
    str
        Complete XYZ block (including trailing newline), ready to be appended
        to a multi-XYZ file.
    None
        Returned when SMILES parsing fails, 3-D embedding fails, or RDKit raises
        an unexpected error.  The caller should count and skip these.
    """
    from rdkit import Chem, RDLogger            # noqa: PLC0415
    from rdkit.Chem.inchi import MolToInchi     # noqa: PLC0415
    RDLogger.DisableLog("rdApp.*")
    from generate.spin_equivalence import classify_spin_groups  # noqa: PLC0415

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        mol_h, groups = classify_spin_groups(mol)
    except Exception:
        return None

    if mol_h.GetNumConformers() == 0:
        return None

    # ── Map each H atom index → its SpinGroup ─────────────────────────────────
    h_to_group: dict[int, object] = {}
    for g in groups:
        for h_idx in g.h_indices:
            h_to_group[h_idx] = g

    # ── Assign sequential class numbers to chemical-equivalence classes ────────
    # HARD and SOFT groups with the same class_h_indices share one class number,
    # reflecting their shared averaged chemical shift.
    # NONE singletons receive no class number.
    class_num: dict[tuple[int, ...], int] = {}
    counter = 1
    for g in groups:
        if g.tier in ("HARD", "SOFT") and g.class_h_indices not in class_num:
            class_num[g.class_h_indices] = counter
            counter += 1

    # ── InChI ─────────────────────────────────────────────────────────────────
    try:
        inchi = MolToInchi(mol) or ""
    except Exception:
        inchi = ""

    # ── Build block ───────────────────────────────────────────────────────────
    comment = json.dumps(
        {"smiles": smiles, "inchikey": inchikey,
         "chembl_id": chembl_id, "inchi": inchi},
        separators=(",", ":"),
    )

    conf    = mol_h.GetConformer()
    n_atoms = mol_h.GetNumAtoms()
    lines:  list[str] = [str(n_atoms), comment]

    for atom in mol_h.GetAtoms():
        idx = atom.GetIdx()
        sym = atom.GetSymbol()
        p   = conf.GetAtomPosition(idx)

        if atom.GetAtomicNum() == 1 and idx in h_to_group:
            g = h_to_group[idx]
            tier_char = g.tier[0]          # first char: H, S, or N

            if g.tier == "NONE":
                annotation = f"{g.label} {tier_char}"
            else:
                n = class_num[g.class_h_indices]
                annotation = f"{g.label} {tier_char}{n}"

            lines.append(
                f"{sym:<2s}  {p.x:12.6f}  {p.y:12.6f}  {p.z:12.6f}  {annotation}"
            )
        else:
            lines.append(
                f"{sym:<2s}  {p.x:12.6f}  {p.y:12.6f}  {p.z:12.6f}"
            )

    return "\n".join(lines) + "\n"


# ── Batch writer ──────────────────────────────────────────────────────────────

def write_xyz_gz(
    input_path:  Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    *,
    verbose: bool = True,
) -> tuple[int, int]:
    """Stream molecules from *input_path* into a gzip multi-XYZ at *output_path*.

    Each molecule is processed and flushed to disk immediately — memory use is
    O(1 molecule) regardless of dataset size.

    Parameters
    ----------
    input_path:
        CSV produced by ``python generate/cli.py run``.
        Required columns: ``chembl_id``, ``smiles``, ``inchikey``.
    output_path:
        Destination ``.xyz.gz`` file.  Parent directories are created if absent.
        Any existing file is overwritten.
    verbose:
        Show a tqdm progress bar and summary when ``True``.

    Returns
    -------
    total : int
        Molecules read from *input_path*.
    written : int
        Molecules successfully written to *output_path*.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        try:
            from tqdm import tqdm as _tqdm  # noqa: PLC0415
        except ImportError:
            _tqdm = None
    else:
        _tqdm = None

    csv.field_size_limit(10 * 1024 * 1024)
    with open(input_path, newline="") as f:
        rows = list(csv.DictReader(f))

    total   = 0
    written = 0
    failed  = 0

    iterator = (
        _tqdm(rows, desc="Writing 8spin.xyz.gz", unit=" mol") if _tqdm else rows
    )

    with gzip.open(output_path, "wt", encoding="utf-8", compresslevel=6) as gz:
        for row in iterator:
            total += 1
            smiles    = row.get("smiles", "")
            chembl_id = row.get("chembl_id", f"mol_{total}")
            inchikey  = row.get("inchikey", "")

            xyz = molecule_to_xyz(smiles, chembl_id=chembl_id, inchikey=inchikey)

            if xyz is None:
                failed += 1
                continue

            gz.write(xyz)
            written += 1

    if verbose:
        size_mb = output_path.stat().st_size / 1e6
        print(f"\nRead    : {total:>8,}")
        print(f"Written : {written:>8,}  ({100 * written / max(total, 1):.1f}%)")
        print(f"Failed  : {failed:>8,}")
        print(f"Output  : {output_path}  ({size_mb:.1f} MB compressed)")

    return total, written
