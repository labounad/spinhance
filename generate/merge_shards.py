"""generate/merge_shards.py — combine Slurm-array shard outputs.

A sharded ``run`` (``--num-shards N``) writes one CSV and one gzip multi-XYZ
per array task::

    <shard_dir>/part_0.csv      <shard_dir>/part_0.xyz.gz
    <shard_dir>/part_1.csv      <shard_dir>/part_1.xyz.gz
    ...

This merges them back into the single-file outputs the rest of the pipeline
expects:

* **CSV** — the header is written once, then every shard's data rows are
  appended in numeric shard order.
* **XYZ** — gzip members concatenate losslessly, so the per-shard ``.xyz.gz``
  files are streamed back-to-back into one ``.gz``; ``gzip.open`` (and ASE,
  OpenBabel, …) read the result as a single multi-XYZ file.

Usage
-----
::

    python -m generate.merge_shards <shard_dir> <out.csv> [<out.xyz.gz>]
    python generate/cli.py merge <shard_dir>          # via the CLI
"""

from __future__ import annotations

import csv
import re
import shutil
import sys
from pathlib import Path

csv.field_size_limit(10 * 1024 * 1024)


def _shard_key(path: Path) -> int:
    """Numeric shard index parsed from ``part_<n>...`` for correct ordering."""
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else 0


def merge_csv(shard_dir: Path, out_csv: Path) -> int:
    """Concatenate ``part_*.csv`` into *out_csv*; return data rows written."""
    files = sorted(Path(shard_dir).glob("part_*.csv"), key=_shard_key)
    rows = 0
    with open(out_csv, "w", newline="") as out:
        writer = csv.writer(out)
        header_written = False
        for f in files:
            with open(f, newline="") as fin:
                reader = csv.reader(fin)
                header = next(reader, None)
                if header is None:
                    continue
                if not header_written:
                    writer.writerow(header)
                    header_written = True
                for row in reader:
                    writer.writerow(row)
                    rows += 1
    return rows


def merge_xyz(shard_dir: Path, out_xyz: Path) -> int:
    """Concatenate ``part_*.xyz.gz`` (gzip members) into *out_xyz*.

    Returns the number of shard files merged.  Concatenated gzip streams are a
    valid gzip file, so no decompression is needed.
    """
    files = sorted(Path(shard_dir).glob("part_*.xyz.gz"), key=_shard_key)
    with open(out_xyz, "wb") as out:
        for f in files:
            with open(f, "rb") as fin:
                shutil.copyfileobj(fin, out)
    return len(files)


def merge_shards(
    shard_dir: str | Path,
    out_csv: str | Path,
    out_xyz: str | Path | None = None,
) -> tuple[int, int]:
    """Merge a shard directory into *out_csv* (+ *out_xyz* if given).

    Returns ``(csv_rows, xyz_shards_merged)``.
    """
    shard_dir = Path(shard_dir)
    rows = merge_csv(shard_dir, Path(out_csv))
    n_xyz = merge_xyz(shard_dir, Path(out_xyz)) if out_xyz else 0
    return rows, n_xyz


def main() -> None:
    args = sys.argv[1:]
    if len(args) not in (2, 3):
        sys.exit(
            "usage: python -m generate.merge_shards "
            "<shard_dir> <out.csv> [<out.xyz.gz>]"
        )
    shard_dir, out_csv = args[0], args[1]
    out_xyz = args[2] if len(args) == 3 else None
    rows, n_xyz = merge_shards(shard_dir, out_csv, out_xyz)
    print(f"merged {rows:,} rows -> {out_csv}")
    if out_xyz:
        print(f"merged {n_xyz} shard XYZ files -> {out_xyz}")


if __name__ == "__main__":
    main()
