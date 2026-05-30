from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def merge_shards(out_path: str | Path, shard_dir: str | Path) -> int:
    """Merge shard_<n>.json arrays (in numeric order) into one JSON array.

    Loads one shard at a time and streams records to the output. Returns the
    total number of records written.
    """
    files = sorted(
        Path(shard_dir).glob("shard_*.json"),
        key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)),
    )
    n = 0
    with open(out_path, "w") as out:
        out.write("[\n")
        for f in files:
            for record in json.loads(f.read_text()):
                out.write((",\n" if n else "") + json.dumps(record))
                n += 1
        out.write("\n]\n")
    return n


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python -m mol_to_spin_system.merge_shards OUT.json SHARD_DIR")
    out, shard_dir = sys.argv[1], sys.argv[2]
    print(f"merged {merge_shards(out, shard_dir)} records -> {out}")
