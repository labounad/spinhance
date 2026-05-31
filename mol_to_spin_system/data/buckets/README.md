# spin-system buckets (per spin-group count)

Task 2 output bucketed by number of ¹H spin groups, produced from the
`generate/data/buckets/` molecule sets. One gzipped JSON array per count:

```
spin_systems_<source>_<n>spin.json.gz      # n = 1 .. 10
```

Kept separate from the top-level `spin_systems_*.json` (which teammate
training references) so existing workflows are untouched.

## Record schema

```json
{
  "chembl_id": "...", "smiles": "...", "inchikey": "...",
  "labels": ["A","B", ...],
  "spin_groups":   [[shift_ppm, n_H], ...],   // mean predicted shift
  "couplings":     [["A","B", J_Hz], ...],
  "shift_range":   [[min_ppm, max_ppm], ...], // NMRShiftDB spread, per group
  "coupling_types":["aromatic","vicinal", ...]// mechanism, per coupling
}
```

`shift_range` and `coupling_types` are the **intermediate info from the
expensive predictor calc**, stored so spectra can be **randomized for
augmentation without re-running the predictor**:

```python
from mol_to_spin_system.augment import sample_record
aug = sample_record(record)   # N(mean, σ) shifts + N(J, σ_type) couplings
```

σ is derived per group from `shift_range` (clip((max-min)/4, floor, cap)) and
per coupling from its type. See `mol_to_spin_system/augment.py` and visualize
with `python -m mol_to_spin_system.bintest`.

## Counts (ChEMBL)

| n | mols | n | mols |
|---|---|---|---|
| 1 | 1,576 | 6 | 32,775 |
| 2 | 2,883 | 7 | 48,632 |
| 3 | 5,936 | 8 | 69,243 |
| 4 | 11,932 | 9 | 85,253 |
| 5 | 20,378 | 10 | 100,030 |

The 8-spin bucket supersedes the legacy `spin_systems_chembl.json` (64k,
mean-only): recall-improved and augmentation-ready.
