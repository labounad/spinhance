"""
Batch-export white-on-black ChemDraw SVGs for every molecule in spin_viewer.json.
Uses ChemDraw COM automation via win32com (Windows Python / Anaconda).

Run from WSL2 with Windows Python:
    /mnt/c/Users/SMansfield/AppData/Local/anaconda3/python.exe \
        docs/data/gen_chemdraw_svgs.py

Writes: docs/data/cd_svgs/<chembl_id>.svg
Then call gen_spin_viewer_svgs.py to embed them into spin_viewer.json.
"""
import sys, os, json, time, re
from pathlib import Path

# ── paths (Windows-native since we're running under Windows Python) ──────────
REPO   = Path(__file__).resolve().parent.parent.parent   # spinhance/
DATA   = REPO / "docs" / "data"
OUT    = DATA / "cd_svgs"
STYLE  = Path(r"C:\ProgramData\RevvitySignalsSoftware\ChemDrawApplications\ChemDraw\ChemDraw Items\Seiple Lab Bold.cds")
JSON_F = DATA / "spin_viewer.json"

OUT.mkdir(exist_ok=True)

import win32com.client

print("Connecting to ChemDraw…", flush=True)
app = win32com.client.Dispatch("ChemDraw_x64.Application")
app.Visible = False

with open(JSON_F) as f:
    data = json.load(f)

mols = data["molecules"]
print(f"Exporting {len(mols)} molecules with 'Seiple Lab Bold' style…", flush=True)

done = skipped = errors = 0

for i, mol in enumerate(mols):
    out_path = OUT / f"{mol['id']}.svg"
    if out_path.exists():
        skipped += 1
        continue

    smiles = mol["smiles"]
    try:
        doc = app.Documents.Add()
        doc.LoadStyleSheet(str(STYLE))

        # Import SMILES
        objs = doc.Objects
        obj  = objs.AddChemical(smiles, "SMILES")

        # Invert colours: make all atoms/bonds white, background black
        # ChemDraw COM: set foreground color on the structure object
        # Color is an RGB long: R + G*256 + B*65536
        WHITE = 0xFFFFFF
        try:
            obj.ForegroundColor = WHITE
        except Exception:
            pass

        # Export as SVG
        doc.SaveAs(str(out_path), 0x4100)   # 0x4100 = cdFormatSVG

        doc.Close(0)   # 0 = don't save
        done += 1

        if i % 10 == 0:
            print(f"  {i+1}/{len(mols)} done={done} skip={skipped} err={errors}",
                  flush=True)

    except Exception as e:
        errors += 1
        print(f"  ERROR {mol['id']}: {e}", flush=True)
        try:
            doc.Close(0)
        except Exception:
            pass

print(f"\nDone. {done} exported, {skipped} skipped, {errors} errors → {OUT}")
app.Quit()
