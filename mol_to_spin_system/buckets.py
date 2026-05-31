"""mol_to_spin_system/buckets.py — merge Task 2 shards into per-count datasets.

A sharded conversion (``mol_to_spin_system.xyz --num-shards``) of a multi-spin
input writes one ``shard_<n>.json`` array per task, each holding records with a
mix of spin-group counts.  This merges them and routes every record to a
per-count, gzip-compressed output::

    <out_dir>/spin_systems_<prefix>_1spin.json.gz
    <out_dir>/spin_systems_<prefix>_2spin.json.gz
    ...

A record's spin-group count is ``len(record["labels"])``.  Shards are read one
at a time (each is small) and streamed straight to the bucket handles, so the
full merged dataset is never held in memory.

Usage
-----
::

    python -m mol_to_spin_system.buckets <shard_dir> <out_dir> <prefix>
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path


def _shard_key(path: Path) -> int:
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else 0


def merge_split_by_count(
    shard_dir: str | Path,
    out_dir: str | Path,
    prefix: str,
) -> Counter:
    """Merge ``shard_*.json`` into ``spin_systems_<prefix>_<n>spin.json.gz``.

    Returns a Counter of records written per spin-group count.
    """
    shard_dir, out_dir = Path(shard_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(shard_dir.glob("shard_*.json"), key=_shard_key)

    handles: dict[int, object] = {}
    written: dict[int, int] = {}
    counts: Counter = Counter()

    for f in files:
        for rec in json.loads(f.read_text()):
            n = len(rec["labels"])
            if n not in handles:
                fh = gzip.open(
                    out_dir / f"spin_systems_{prefix}_{n}spin.json.gz",
                    "wt", encoding="utf-8", compresslevel=6,
                )
                fh.write("[\n")
                handles[n] = fh
                written[n] = 0
            fh = handles[n]
            fh.write((",\n" if written[n] else "") + json.dumps(rec))
            written[n] += 1
            counts[n] += 1

    for n, fh in handles.items():
        fh.write("\n]\n")
        fh.close()
    return counts


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit("usage: python -m mol_to_spin_system.buckets "
                 "<shard_dir> <out_dir> <prefix>")
    shard_dir, out_dir, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
    counts = merge_split_by_count(shard_dir, out_dir, prefix)
    total = sum(counts.values())
    print(f"merged {total:,} records into {len(counts)} per-count files "
          f"-> {out_dir}/spin_systems_{prefix}_<n>spin.json.gz")
    for n in sorted(counts):
        print(f"  {n:>2}spin: {counts[n]:>10,}")


if __name__ == "__main__":
    main()
