# molecule buckets (per spin-group count)

Screened molecules bucketed by number of ¹H spin groups, from a single
categorising scan (`run --min-groups 1 --max-groups N`) → `merge --dedup` →
`split`. Two gzipped files per count:

```
<source>_<n>spin.csv.gz     # chembl_id, smiles, inchikey, n_groups, group_sizes
<source>_<n>spin.xyz.gz     # 3-D structures, spin-group annotated (see XYZ_FORMAT.md)
```

One scan buckets every molecule by its count instead of running one screen per
count (~Nx less embedding). The wider `carbons ≤ N` heuristic also improves
per-bucket recall vs the old exact-count screen (e.g. ChEMBL 8-spin: 69,324 vs
the legacy 64,476).

Kept separate from `generate/data/chembl_8spin.*` / `pubchem_8spin.*` so the
existing 8-spin workflows are untouched.

Downstream: `mol_to_spin_system` converts the `.xyz.gz` here into the
per-count spin-system datasets under `mol_to_spin_system/data/buckets/`.

## Counts (ChEMBL, csv rows)

| n | mols | n | mols |
|---|---|---|---|
| 1 | 1,578 | 6 | 32,792 |
| 2 | 2,884 | 7 | 48,660 |
| 3 | 5,936 | 8 | 69,324 |
| 4 | 11,935 | 9 | 85,313 |
| 5 | 20,384 | 10 | 100,092 |
