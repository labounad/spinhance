"""generate/dedup.py — drop duplicate molecules from a screened dataset.

PubChem (and other sources) deposit the same compound under multiple CIDs, so a
merged screen can contain several rows that share an InChIKey.  This collapses
them to **one row per InChIKey** (first occurrence wins) and filters the
companion multi-XYZ file to the surviving IDs, keeping the CSV and XYZ in sync.

Usage
-----
::

    python -m generate.dedup IN.csv OUT.csv [IN.xyz.gz OUT.xyz.gz]
    python generate/cli.py dedup IN.csv OUT.csv --in-xyz IN.xyz.gz --out-xyz OUT.xyz.gz
"""

from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

csv.field_size_limit(10 * 1024 * 1024)

#: CSV column holding the dedup key (chembl_id, smiles, inchikey, ...).
INCHIKEY_COL = 2
ID_COL = 0


def _filter_xyz(in_xyz: Path, out_xyz: Path, kept_ids: set[str]) -> int:
    """Copy only the XYZ blocks whose comment ``chembl_id`` is in *kept_ids*."""
    n = 0
    with (
        gzip.open(in_xyz, "rt") as fin,
        gzip.open(out_xyz, "wt", encoding="utf-8", compresslevel=6) as fout,
    ):
        while True:
            head = fin.readline()
            if not head:
                break
            na = int(head)
            comment = fin.readline()
            body = [fin.readline() for _ in range(na)]
            if json.loads(comment)["chembl_id"] in kept_ids:
                fout.write(head)
                fout.write(comment)
                fout.writelines(body)
                n += 1
    return n


def dedup_dataset(
    in_csv: str | Path,
    out_csv: str | Path,
    *,
    in_xyz: str | Path | None = None,
    out_xyz: str | Path | None = None,
    key_col: int = INCHIKEY_COL,
) -> tuple[int, int, int]:
    """Collapse rows sharing column *key_col* to their first occurrence.

    Parameters
    ----------
    in_csv, out_csv:
        Source and destination CSV (header preserved).
    in_xyz, out_xyz:
        Optional companion multi-XYZ; when both are given, only blocks whose
        ID survived the CSV dedup are written, so the pair stays consistent.
    key_col:
        Column to dedup on (default :data:`INCHIKEY_COL`, the InChIKey).

    Returns
    -------
    (kept, dropped, xyz_written)
    """
    in_csv, out_csv = Path(in_csv), Path(out_csv)
    seen: set[str] = set()
    kept_ids: set[str] = set()
    kept = dropped = 0

    with (
        open(in_csv, newline="") as f,
        open(out_csv, "w", newline="") as out,
    ):
        reader = csv.reader(f)
        writer = csv.writer(out)
        header = next(reader, None)
        if header is not None:
            writer.writerow(header)
        for row in reader:
            key = row[key_col]
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            kept_ids.add(row[ID_COL])
            writer.writerow(row)
            kept += 1

    xyz_written = 0
    if in_xyz and out_xyz:
        xyz_written = _filter_xyz(Path(in_xyz), Path(out_xyz), kept_ids)
    return kept, dropped, xyz_written


def main() -> None:
    args = sys.argv[1:]
    if len(args) not in (2, 4):
        sys.exit(
            "usage: python -m generate.dedup IN.csv OUT.csv "
            "[IN.xyz.gz OUT.xyz.gz]"
        )
    in_csv, out_csv = args[0], args[1]
    in_xyz, out_xyz = (args[2], args[3]) if len(args) == 4 else (None, None)
    kept, dropped, n_xyz = dedup_dataset(
        in_csv, out_csv, in_xyz=in_xyz, out_xyz=out_xyz
    )
    print(f"kept {kept:,} unique  (dropped {dropped:,} duplicate InChIKeys) -> {out_csv}")
    if out_xyz:
        print(f"wrote {n_xyz:,} XYZ blocks -> {out_xyz}")


if __name__ == "__main__":
    main()
