"""
discover_mnova_api3.py
Scan for all importable .so modules in the MNova bundle,
then probe what each one exposes.
"""
import os, sys, importlib

bundle = "/Applications/MestReNova.app/Contents"

# ── 1. Find all .so / .dylib files that could be Python extension modules ─────
print("--- .so files in MNova bundle ---")
so_files = []
for root, dirs, files in os.walk(bundle):
    for f in files:
        if f.endswith(".so") or f.endswith(".dylib"):
            so_files.append(os.path.join(root, f))
            print(" ", os.path.join(root, f))

# ── 2. Try importing each one by adding its dir to sys.path ───────────────────
print("\n--- Attempting imports ---")
tried = set()
for path in so_files:
    d   = os.path.dirname(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    # strip .cpython-310-darwin suffix if present
    if ".cpython" in stem:
        stem = stem.split(".")[0]
    if stem in tried or stem.startswith("_"):
        continue
    tried.add(stem)
    if d not in sys.path:
        sys.path.insert(0, d)
    try:
        mod = importlib.import_module(stem)
        members = [x for x in dir(mod) if not x.startswith("_")]
        print(f"  IMPORTED: {stem}")
        print(f"    members: {members[:30]}")
    except Exception as e:
        pass   # silently skip failures

# ── 3. Also scan Resources/scripts for .qs examples ──────────────────────────
print("\n--- .qs script files in bundle ---")
for root, dirs, files in os.walk(bundle):
    for f in files:
        if f.endswith(".qs"):
            path = os.path.join(root, f)
            print(f"  {path}")
            with open(path, errors="replace") as fh:
                print(fh.read()[:800])
            print()

print("\nDone.")
