"""generate/buckets.py — split a categorising scan into per-count datasets.

A range scan (``run --min-groups 1 --max-groups 26``) produces one combined CSV
and one combined multi-XYZ tagged by spin-group count.  This partitions them
into per-count files::

    <out_dir>/<prefix>_1spin.csv   <out_dir>/<prefix>_1spin.xyz.gz
    <out_dir>/<prefix>_2spin.csv   <out_dir>/<prefix>_2spin.xyz.gz
    ...

so each spin-group count is a standalone, ready-to-use dataset.

The CSV is split on its ``n_groups`` column.  The XYZ is split by counting the
distinct spin-group **labels** in each block (each block's label set has exactly
``n_groups`` members), so no CSV lookup or large in-memory map is needed.

Usage
-----
::

    python -m generate.buckets COMBINED.csv OUT_DIR PREFIX [COMBINED.xyz.gz]
    python generate/cli.py split COMBINED.csv OUT_DIR --prefix pubchem --xyz COMBINED.xyz.gz
"""

from __future__ import annotations

import csv
import gzip
import sys
from collections import Counter
from pathlib import Path

csv.field_size_limit(10 * 1024 * 1024)

N_GROUPS_COL = 3


def _block_n_groups(comment_atoms: list[str]) -> int:
    """Number of distinct spin-group labels in one block's atom lines.

    Annotated ¹H lines look like ``H  x y z  <label> <tier>`` (6+ tokens);
    heavy atoms and exchangeable H have only 4.  The label is token index 4.
    """
    labels = set()
    for line in comment_atoms:
        parts = line.split()
        if len(parts) >= 6 and parts[0] == "H":
            labels.add(parts[4])
    return len(labels)


def split_csv(in_csv: Path, out_dir: Path, prefix: str) -> Counter:
    """Split *in_csv* into ``<prefix>_<n>spin.csv`` by the ``n_groups`` column."""
    out_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[int, csv.writer] = {}
    handles: dict[int, object] = {}
    counts: Counter = Counter()
    with open(in_csv, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            n = int(row[N_GROUPS_COL])
            if n not in writers:
                fh = open(out_dir / f"{prefix}_{n}spin.csv", "w", newline="")
                handles[n] = fh
                writers[n] = csv.writer(fh)
                if header is not None:
                    writers[n].writerow(header)
            writers[n].writerow(row)
            counts[n] += 1
    for fh in handles.values():
        fh.close()
    return counts


def split_xyz(in_xyz: Path, out_dir: Path, prefix: str) -> Counter:
    """Split *in_xyz* into ``<prefix>_<n>spin.xyz.gz`` by per-block label count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    handles: dict[int, object] = {}
    counts: Counter = Counter()
    with gzip.open(in_xyz, "rt") as fin:
        while True:
            head = fin.readline()
            if not head:
                break
            na = int(head)
            comment = fin.readline()
            body = [fin.readline() for _ in range(na)]
            n = _block_n_groups(body)
            if n not in handles:
                handles[n] = gzip.open(
                    out_dir / f"{prefix}_{n}spin.xyz.gz", "wt",
                    encoding="utf-8", compresslevel=6,
                )
            fh = handles[n]
            fh.write(head)
            fh.write(comment)
            fh.writelines(body)
            counts[n] += 1
    for fh in handles.values():
        fh.close()
    return counts


def split_dataset(
    in_csv: str | Path,
    out_dir: str | Path,
    prefix: str,
    in_xyz: str | Path | None = None,
) -> tuple[Counter, Counter]:
    """Split a combined scan into per-count CSV (+ XYZ) files.

    Returns ``(csv_counts, xyz_counts)`` keyed by spin-group count.
    """
    out_dir = Path(out_dir)
    csv_counts = split_csv(Path(in_csv), out_dir, prefix)
    xyz_counts = split_xyz(Path(in_xyz), out_dir, prefix) if in_xyz else Counter()
    return csv_counts, xyz_counts


def main() -> None:
    args = sys.argv[1:]
    if len(args) not in (3, 4):
        sys.exit(
            "usage: python -m generate.buckets COMBINED.csv OUT_DIR PREFIX "
            "[COMBINED.xyz.gz]"
        )
    in_csv, out_dir, prefix = args[0], args[1], args[2]
    in_xyz = args[3] if len(args) == 4 else None
    csv_counts, xyz_counts = split_dataset(in_csv, out_dir, prefix, in_xyz)
    total = sum(csv_counts.values())
    print(f"split {total:,} rows into {len(csv_counts)} buckets -> {out_dir}/{prefix}_<n>spin.csv")
    for n in sorted(csv_counts):
        x = f", {xyz_counts.get(n, 0):,} xyz" if xyz_counts else ""
        print(f"  {n:>2}spin: {csv_counts[n]:>10,} rows{x}")


if __name__ == "__main__":
    main()
