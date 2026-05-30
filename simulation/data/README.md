# simulation/data — generated spectra

Simulated ¹H spectra for the Task 4 model, produced by the `pyspin` engine from
Task 2's spin-system graphs.

## Provenance

- **Source:** `mol_to_spin_system/data/spin_systems.json` (1072 molecules, preliminary set)
- **Engine:** `simulation/cli.py run --engine python` (pure-Python, exact;
  local-cluster approximation for any fragment > 12 coupled spins)
- **Fields:** 90 MHz (low, model input) and 600 MHz (high, reference)
- **Grid:** 16384 points over 0–12 ppm, each spectrum normalised to ∫ = 1

## Layout

```
spectra/
├── 90MHz/   mol_000000.npy … mol_001071.npy   (+ ppm_axis.npy)
├── 600MHz/  mol_000000.npy … mol_001071.npy   (+ ppm_axis.npy)
└── index.csv                                   # mol_<i> → chembl_id
```

`mol_<i>` indexes the i-th record of `spin_systems.json` (see `index.csv` for the
ChEMBL id). Each `.npy` is a float32 array of 16384 intensities; `ppm_axis.npy`
is the shared chemical-shift axis.

## Regenerate

The `spectra/` contents are **not committed** (large binaries; see
`.gitignore`). Recreate them with:

```bash
python -m simulation.cli run \
    --graphs mol_to_spin_system/data/spin_systems.json \
    --out_dir simulation/data \
    --fields 90 600 --engine python --workers 8
```

(~seconds for this set; `--workers` scales across cores for larger datasets.)
