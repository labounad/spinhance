"""
Batch-export ChemDraw SVGs using the 'Seiple Lab Bold' style sheet.
Run with Windows Anaconda Python (needs win32com):

  C:\Users\SMansfield\AppData\Local\anaconda3\python.exe ^
      \\wsl.localhost\Debian\home\smansfield\spinhance\docs\data\export_chemdraw_svgs.py
"""
import json, re, sys, time
from pathlib import Path, PureWindowsPath

# ── paths ────────────────────────────────────────────────────────────────────
BASE   = Path(r"\\wsl.localhost\Debian\home\smansfield\spinhance\docs\data")
MOL_IN = BASE / "mol_files"
SVG_OUT= BASE / "cd_svgs"
JSON_F = BASE / "spin_viewer.json"
STYLE  = Path(r"C:\ProgramData\RevvitySignalsSoftware\ChemDrawApplications\ChemDraw\ChemDraw Items\Seiple Lab Bold.cds")

SVG_OUT.mkdir(exist_ok=True)

import win32com.client

print("Connecting to ChemDraw …")
app = win32com.client.Dispatch("ChemDraw_x64.Application")
app.Visible = False

with open(JSON_F) as f:
    mols = json.load(f)["molecules"]

done = skipped = errors = 0

for i, mol in enumerate(mols):
    mol_path = MOL_IN / f"{mol['id']}.mol"
    svg_path = SVG_OUT / f"{mol['id']}.svg"

    if svg_path.exists():
        skipped += 1
        continue

    if not mol_path.exists():
        print(f"  MISSING mol: {mol['id']}")
        errors += 1
        continue

    try:
        doc = app.Documents.Open(str(mol_path))
        doc.Settings.ApplySettings(str(STYLE), "")

        # White bonds/atoms on transparent background
        doc.Settings.BackgroundColor = 0x000000
        doc.Settings.Color = 0xFFFFFF

        doc.SaveAs(str(svg_path), "image/svg+xml")
        doc.Close(0)
        done += 1

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(mols)}  done={done} skip={skipped} err={errors}")

    except Exception as e:
        errors += 1
        print(f"  ERROR {mol['id']}: {e}")
        try: doc.Close(0)
        except: pass

app.Quit()
print(f"\nDone. {done} exported, {skipped} already existed, {errors} errors")
print(f"SVGs written to: {SVG_OUT}")
