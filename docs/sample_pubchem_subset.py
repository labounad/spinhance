#!/usr/bin/env python3
"""
sample_pubchem_subset.py
========================
Reservoir-sample a random N-molecule subset from the large PubChem spin-system
dataset (mol_to_spin_system/data/spin_systems_pubchem.json.tar.gz, ~2M+ records,
~1 GB uncompressed) and write it as the website's hero pool.

The source JSON is one record per line, so we stream it line-by-line and keep
only N records in memory (O(N), never the whole file). The tarball is read as a
stream — it is never extracted to disk.

Output: docs/data/spin_systems_pubchem.json  (the pool build_field_sweep.py reads)
"""
from __future__ import annotations

import json
import random
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC_TAR = REPO / "mol_to_spin_system" / "data" / "spin_systems_pubchem.json.tar.gz"
MEMBER = "spin_systems_pubchem.json"
OUT = REPO / "docs" / "data" / "spin_systems_pubchem.json"

N_SAMPLE = 1000
SEED = 0  # fixed for reproducibility; bump to draw a different subset


def reservoir_sample(stream, k: int, rng: random.Random) -> list[str]:
    """Reservoir-sample k JSON-record lines from a text stream of one-per-line records."""
    reservoir: list[str] = []
    n = 0
    for raw in stream:
        line = raw.decode("utf-8").strip() if isinstance(raw, (bytes, bytearray)) else raw.strip()
        if not line or line[0] != "{":      # skip the opening "[" and closing "]"
            continue
        rec = line.rstrip(",")               # records are comma-separated, one per line
        n += 1
        if len(reservoir) < k:
            reservoir.append(rec)
        else:
            j = rng.randint(0, n - 1)
            if j < k:
                reservoir[j] = rec
    return reservoir, n


def main() -> None:
    if not SRC_TAR.exists():
        raise SystemExit(f"source not found: {SRC_TAR}")
    rng = random.Random(SEED)
    print(f"streaming {SRC_TAR.name} (member {MEMBER})…")
    with tarfile.open(SRC_TAR, "r:gz") as tf:
        member = tf.getmember(MEMBER)
        f = tf.extractfile(member)
        if f is None:
            raise SystemExit(f"could not open {MEMBER} inside the tarball")
        sampled, total = reservoir_sample(f, N_SAMPLE, rng)

    records = [json.loads(t) for t in sampled]
    OUT.write_text(json.dumps(records, separators=(",", ":")))
    print(f"scanned {total:,} molecules -> sampled {len(records)} (seed {SEED})")
    print(f"wrote {OUT}  ({OUT.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
