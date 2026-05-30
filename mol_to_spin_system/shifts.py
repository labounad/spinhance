from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

# --- predictor location -----------------------------------------------------
# The nmrshiftdb2 checkout (predictor JARs + CDK deps) lives OUTSIDE the repo,
# as a sibling directory: <repo>/../nmrshiftdb2. We resolve it relative to this
# file so it works regardless of the current working directory or where the repo
# is cloned. Override with the NMRSHIFTDB_HOME environment variable if needed.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_HOME = _REPO_ROOT.parent / "nmrshiftdb2"
NMRSHIFTDB_HOME = Path(os.environ.get("NMRSHIFTDB_HOME", _DEFAULT_HOME))
_SNAPSHOTS = NMRSHIFTDB_HOME / "trunk" / "snapshots"
_LIB = NMRSHIFTDB_HOME / "branches" / "beta-maintenance" / "lib"

_PREDICTOR_JAR = {"H": "predictorh.jar", "C": "predictorc.jar"}

# The JAR only accepts these three solvent strings (verbatim).
VALID_SOLVENTS = {
    "Chloroform-D1 (CDCl3)",
    "Methanol-D4 (CD3OD)",
    "Dimethylsulphoxide-D6 (DMSO-D6, C2D6SO)",
}
DEFAULT_SOLVENT = "Chloroform-D1 (CDCl3)"

# rows look like: " 19:    1.42    1.49    1.56"
_ROW = re.compile(r"^\s*(\d+):\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")


def _classpath(nucleus: str) -> str:
    """Predictor JAR + all CDK dependency jars, joined for `java -cp`."""
    try:
        jar = _SNAPSHOTS / _PREDICTOR_JAR[nucleus]
    except KeyError:
        raise ValueError(f"nucleus must be one of {sorted(_PREDICTOR_JAR)}, got {nucleus!r}")
    if not jar.exists():
        raise FileNotFoundError(
            f"predictor jar not found: {jar}\nSet NMRSHIFTDB_HOME to your nmrshiftdb2 checkout."
        )
    libs = sorted(_LIB.glob("*.jar"))
    if not libs:
        raise FileNotFoundError(f"no CDK dependency jars under {_LIB}")
    return os.pathsep.join([str(jar)] + [str(p) for p in libs])


def _parse_prediction(stdout: str) -> dict[int, dict[str, float]]:
    """Parse the predictor's table into {atom_index (1-based): {min, mean, max}}."""
    shifts: dict[int, dict[str, float]] = {}
    for line in stdout.splitlines():
        m = _ROW.match(line)
        if m:
            shifts[int(m.group(1))] = {
                "min": float(m.group(2)),
                "mean": float(m.group(3)),
                "max": float(m.group(4)),
            }
    return shifts


def predict_shifts_from_molfile(
    molfile: str | Path,
    nucleus: str = "H",
    solvent: str = DEFAULT_SOLVENT,
    use_3d: bool = True,
) -> dict[int, dict[str, float]]:
    """Run the NMRShiftDB2 predictor on a MOL file.

    Returns {atom_index (1-based, matching MOL atom order): {min, mean, max} ppm}.
    Only atoms of the requested nucleus appear in the result.
    """
    if solvent not in VALID_SOLVENTS:
        raise ValueError(f"solvent must be one of {sorted(VALID_SOLVENTS)}, got {solvent!r}")
    molfile = Path(molfile)
    if not molfile.exists():
        raise FileNotFoundError(molfile)

    args = ["java", "-cp", _classpath(nucleus), "Test", str(molfile), solvent]
    if not use_3d:
        args.append("no3d")  # tell the predictor to ignore 3D coords

    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"predictor failed (exit {proc.returncode}):\n{proc.stderr or proc.stdout}")
    return _parse_prediction(proc.stdout)


def make_test_mol_3d(smiles: str = "CCO", seed: int = 0xF00D) -> Chem.Mol:
    """Build an RDKit mol with explicit H's and a minimized 3D conformer.

    Mirrors the Task 2 embedding step (ETKDGv3 + MMFF94) so the result is a valid
    input for the shift predictor.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"could not parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError(f"3D embedding failed for {smiles!r}")
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def predict_shifts(
    mol: Chem.Mol,
    nucleus: str = "H",
    solvent: str = DEFAULT_SOLVENT,
    use_3d: bool = True,
) -> dict[int, dict[str, float]]:
    """Predict shifts for an RDKit mol that already carries a 3D conformer.

    Writes a temporary MOL file (3D coords + explicit H's), submits it to the
    predictor, and re-keys the result on RDKit atom indices (0-based).
    """
    if mol.GetNumConformers() == 0 or not mol.GetConformer().Is3D():
        raise ValueError("mol needs a 3D conformer; embed it first (see make_test_mol_3d).")

    with tempfile.NamedTemporaryFile("w", suffix=".mol", delete=False) as fh:
        path = Path(fh.name)
    try:
        Chem.MolToMolFile(mol, str(path))  # writes the 3D conformer
        raw = predict_shifts_from_molfile(path, nucleus=nucleus, solvent=solvent, use_3d=use_3d)
    finally:
        path.unlink(missing_ok=True)

    # MOL atom indices are 1-based and follow RDKit atom order -> shift by one.
    return {idx - 1: vals for idx, vals in raw.items()}


if __name__ == "__main__":
    # Smoke test: ethanol, 1H shifts.
    test_mol = make_test_mol_3d("CCO")
    result = predict_shifts(test_mol, nucleus="H")
    print("ethanol 1H predictions (RDKit atom idx -> ppm):")
    for idx in sorted(result):
        atom = test_mol.GetAtomWithIdx(idx)
        v = result[idx]
        print(f"  atom {idx:>2} ({atom.GetSymbol()}): mean {v['mean']:6.2f}  [{v['min']:.2f}, {v['max']:.2f}]")
