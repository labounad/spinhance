# mol_to_matrix/data

Spin-system data converted from `generate/data/8spin.xyz{,.gz}` by
`mol_to_matrix.xyz.convert_file`.

- `spin_systems.json` — **1072 molecules** from `8spin.xyz` (small uncompressed set).
  Regenerate with `convert_file('generate/data/8spin.xyz', <out>, workers=8)`.
- `spin_systems_60k.json` — **64,476 molecules** from `8spin.xyz.gz` (full dataset).
  Regenerate with `convert_file('generate/data/8spin.xyz.gz', <out>, workers=32)`.

## Format

A single JSON **array**; each element is one molecule:

```json
{
  "chembl_id": "CHEMBL6622",
  "smiles": "O=[N+]([O-])O[C@H]1CO[C@H]2[C@@H]1OC[C@H]2O[N+](=O)[O-]",
  "inchikey": "MOYKHGMNXAOIAT-JGWLITMVSA-N",
  "labels": ["A", "B", "C", "D", "E", "F", "G", "H"],
  "spin_groups": [[4.59, 1], [4.05, 1], "..."],
  "couplings": [["A", "C", 5.7], ["B", "C", -10.8], "..."]
}
```

| Field | Type | Meaning |
|-------|------|---------|
| `chembl_id` | str | ChEMBL compound ID |
| `smiles` | str | canonical SMILES |
| `inchikey` | str | standard InChIKey |
| `labels` | list[str] | spin-group labels (A, B, …), sorted; index-aligned with `spin_groups` |
| `spin_groups` | list[[float, int]] | per group: `[chemical shift (ppm), number of H]` |
| `couplings` | list[[str, str, float]] | `[group_i, group_j, J (Hz)]` between distinct spin groups |

## How values are produced

- **Shifts**: NMRShiftDB HOSE-code predictor on the molecule's 3D geometry
  (falls back to non-3D when the predictor's stereo path fails).
- **Couplings**: geometry-based heuristics (Karplus vicinal, table-based
  geminal/olefinic/aromatic/allylic); see `mol_to_matrix/README.md`.
- **Averaging**: protons in the same spin group (`labels`) are averaged for
  both shift and J; protons sharing a tier class number (chemically equivalent
  across groups) are averaged for shift only.

Shifts are rounded to 2 dp, couplings to 1 dp. Coupling sign is retained
(geminal is negative). No magnitude threshold is applied.
