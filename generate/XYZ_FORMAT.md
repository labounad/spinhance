# SpinHance XYZ Format — Interface spec for Task 1 → Task 2

**For:** Yiming (mol_to_matrix)  
**From:** Sam (generate)  
**File produced by:** `python generate/cli.py xyz` → `generate/data/8spin.xyz.gz`

---

## What the file is

A single **gzip-compressed multi-XYZ file** containing every molecule that
passed the 8 spin-group screen.  Each molecule is one contiguous block in
the standard extended-XYZ format.  Tools like ASE read it natively:

```python
from ase.io import read
mols = read("8spin.xyz.gz", index=":")   # list of all Atoms objects
```

To read without ASE, iterate the decompressed text — each block starts
with a line containing just the atom count.

---

## Block format

```
<n_atoms>
<JSON comment>
<symbol>  <x>  <y>  <z>  [<group> <tier_class>]
...
```

### Line 1 — atom count

Plain integer.  Includes **all** atoms: heavy atoms and all hydrogens
(including exchangeable ones like N-H, O-H).

### Line 2 — JSON comment

One-line JSON with four keys:

```json
{"smiles":"Cc1ccccc1","inchikey":"YXFVVABEGXRONW-UHFFFAOYSA-N","chembl_id":"CHEMBL14688","inchi":"InChI=1S/..."}
```

| Key | Type | Content |
|-----|------|---------|
| `smiles` | str | Canonical SMILES from ChEMBL |
| `inchikey` | str | Standard InChIKey |
| `chembl_id` | str | ChEMBL compound ID |
| `inchi` | str | Full standard InChI |

### Lines 3..N+2 — atom records

Tab/space-separated columns:

```
symbol   x_angstrom   y_angstrom   z_angstrom   [group_label   tier_class]
```

**Non-hydrogen atoms** (C, N, O, …) — four columns only:

```
C    1.234567    0.987654   -0.543210
```

**Hydrogen atoms** — four additional annotation columns:

```
H   -0.123456    1.876543    0.234567   B   S2
```

The annotation is only present on **non-exchangeable** C-H protons.
Exchangeable protons (N-H, O-H, S-H) appear without annotation:

```
H    0.456789   -1.234567    0.987654
```

---

## Annotation columns

### `group_label` — spin-group letter

Excel-style letter (A, B, C … Z, AA, AB …).  **Each letter is one spin
group** — a set of protons that MNova simulates as a single NMR resonance.

The spin-group count is always exactly 8 (by construction of the screen).

### `tier_class` — equivalence tier + class number

| Format | Meaning |
|--------|---------|
| `H{n}` | **HARD** — homotopic and magnetically equivalent.  All protons sharing label and class number have *identical* shifts *and* coupling patterns.  Collapse to one group in simulation.  Example: all 3 protons of a freely-rotating CH₃ get the same label and same class number. |
| `S{n}` | **SOFT** — chemically equivalent but magnetically inequivalent.  Protons sharing a class number have the *same averaged chemical shift* but *different J-coupling patterns* and must be simulated as separate spin groups.  Example: the two enantiotopic H's of a CH₂ or the AA′ protons of an aromatic ring. |
| `N`    | **NONE** — chemically distinct singleton.  Unique environment, unique shift.  No class number. |

The class number `{n}` is a sequential integer.  **HARD and SOFT protons
that share a class number are chemical equivalents of each other** — they
should receive the same predicted chemical shift in your shift assignment
step before averaging.

---

## Worked example — toluene (7 spin groups if it passed the screen)

```
17
{"smiles":"Cc1ccccc1","inchikey":"YXFVVABEGXRONW-UHFFFAOYSA-N","chembl_id":"CHEMBL14688","inchi":"InChI=1S/C7H8/c1-7-5-3-2-4-6-7/h2-6H,1H3"}
C    0.000000    0.000000    0.000000
C    1.397000    0.000000    0.000000
C    2.094000    1.211000    0.000000
C    1.397000    2.422000    0.000000
C    0.000000    2.422000    0.000000
C   -0.697000    1.211000    0.000000
C   -0.507000   -1.432000    0.000000
H    1.930000   -0.938000    0.000000   A   N       ← ortho-1, unique
H    3.178000    1.211000    0.000000   B   N       ← para, unique
H    1.930000    3.360000    0.000000   C   N       ← ortho-2, unique
H   -0.534000    3.360000    0.000000   D   N       ← meta-1, unique
H   -1.781000    1.211000    0.000000   E   N       ← meta-2, unique
H   -0.172000   -1.965000    1.022000   F   H1      ← CH3 proton
H   -0.172000   -1.965000   -1.022000   F   H1      ← CH3 proton (same group)
H   -1.598000   -1.432000    0.000000   F   H1      ← CH3 proton (same group)
H   ...                                             ← (exchangeable H if any)
```

In this example:
- Groups A–E are NONE (each aromatic H is chemically distinct)
- Group F is HARD class 1 (all 3 CH₃ protons — same shift, same J to ring)
- Total: 6 spin groups (toluene would not pass the 8-group screen; this is illustrative)

---

## What mol_to_matrix needs to do with this

1. **Read** each XYZ block (atom positions + spin-group labels).
2. **Assign shifts**: for each unique class number, predict one δ (ppm)
   — e.g. via your NMRShiftDB lookup or heuristic on the heavy-atom
   environment.  All protons sharing a class number get the *same* predicted
   δ (that is what "chemically equivalent" means here).
3. **Assign couplings**: compute J(Hz) between every pair of distinct spin
   groups using the 3-D geometry (Karplus for vicinal, tables for geminal/
   aromatic/allylic).  HARD protons within the same group don't need
   intra-group J — MNova handles that implicitly via the `number=N`
   degeneracy attribute.
4. **Output** the 8×8 J-matrix + 8-element shift diagonal + 8-element
   degeneracy vector, ready for `simulation/xml_io.matrix_to_xml`.

---

## Reading the file in Python

```python
import gzip, json

def iter_xyz_blocks(path):
    """Yield (comment_dict, atoms) for each molecule in a multi-XYZ.gz."""
    atoms = []
    with gzip.open(path, "rt") as f:
        while True:
            count_line = f.readline()
            if not count_line:
                break
            n = int(count_line.strip())
            comment = json.loads(f.readline())
            block = []
            for _ in range(n):
                parts = f.readline().split()
                sym = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                group = parts[4] if len(parts) > 4 else None
                tier  = parts[5] if len(parts) > 5 else None
                block.append((sym, x, y, z, group, tier))
            yield comment, block

for meta, atoms in iter_xyz_blocks("generate/data/8spin.xyz.gz"):
    smiles    = meta["smiles"]
    chembl_id = meta["chembl_id"]
    spin_h    = [(sym,x,y,z,g,t) for sym,x,y,z,g,t in atoms
                 if sym == "H" and g is not None]
    print(f"{chembl_id}: {len(spin_h)} labelled H atoms")
```

---

## Quick stats (expected)

| Property | Value |
|----------|-------|
| Molecules | ~tens of thousands after full ChEMBL screen |
| Atoms per molecule | ~20–40 |
| File size (compressed) | ~10–15 MB |
| Spin groups per molecule | exactly 8 |
| H atoms with annotation | all non-exchangeable C-H |

Questions? → Sam (smansfield@scripps.edu)
