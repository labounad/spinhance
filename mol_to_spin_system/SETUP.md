# mol_to_spin_system — setup

> ⚠️ **The production shift engine is now `shifts_pretsch.py`** — a pure-Python Pretsch
> additive model, **no Java/NMRShiftDB required** (the whole HPC rebuild runs predictor-free).
> The Java setup below is for the **legacy** `shifts.py` HOSE path only — optional/historical.

Chemical-shift prediction's legacy path (`shifts.py`) shells out to the standalone
**NMRShiftDB2 predictor JARs** (Java + CDK), which are **not** part of this repo.
Follow these steps once after cloning *only if you need the old HOSE predictor*.
Coupling/grouping code needs only RDKit, and production Pretsch shifts need only Python,
so most workflows skip this.

## 1. Prerequisites

- **Java 8+** runtime (the JARs are built for Java 8; newer JDKs work too)
- **Subversion** (`svn`) to fetch the predictor

```bash
# Arch
sudo pacman -S --noconfirm jdk-openjdk subversion
# Debian/Ubuntu
sudo apt-get install -y default-jre subversion
# macOS (Homebrew)
brew install openjdk subversion
```

Verify: `java -version` prints a version.

## 2. Get the NMRShiftDB2 predictor

Check it out **next to** this repo (sibling directory). `shifts.py` looks for
`../nmrshiftdb2` by default.

```bash
cd ..                 # parent of the spinhance repo
svn checkout https://svn.code.sf.net/p/nmrshiftdb2/code/ nmrshiftdb2
```

This provides:
- predictor JARs — `nmrshiftdb2/trunk/snapshots/predictor{c,h}.jar`
- CDK dependencies — `nmrshiftdb2/branches/beta-maintenance/lib/*.jar`

If you put it elsewhere, point `NMRSHIFTDB_HOME` at it:

```bash
export NMRSHIFTDB_HOME=/path/to/nmrshiftdb2
```

## 3. Verify

```bash
python -m mol_to_spin_system.shifts     # prints ethanol 1H shifts (CH3 ~1.2, CH2 ~3.7)
```

If you see predicted shifts, you're set. A `predictor jar not found` error means
the checkout isn't where `shifts.py` expects — fix the location or set
`NMRSHIFTDB_HOME`.

## Note: input data (Git LFS)

The Task 1 dataset `generate/data/chembl_8spin.xyz.gz` is stored with **Git LFS**.
After cloning, install git-lfs and pull it:

```bash
sudo pacman -S --noconfirm git-lfs   # or apt/brew
git lfs install
git lfs pull
```

Without git-lfs the file is just a ~130-byte pointer text.
