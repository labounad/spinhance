"""
discover_mnova_api2.py — deep dive into MnovaCore
"""
import MnovaCore
import os, sys

def show(label, obj):
    members = [x for x in dir(obj) if not x.startswith("_")]
    print(f"\n--- {label} ---")
    print(f"  type: {type(obj)}")
    print(f"  members ({len(members)}): {members}")

# ── 1. CoreApplication ────────────────────────────────────────────────────────
show("MnovaCore.CoreApplication", MnovaCore.CoreApplication)

# Try instantiating / getting the singleton
for attr in ("instance", "getInstance", "app", "application"):
    if hasattr(MnovaCore.CoreApplication, attr):
        try:
            result = getattr(MnovaCore.CoreApplication, attr)()
            print(f"  CoreApplication.{attr}() = {result}")
            show(f"CoreApplication.{attr}()", result)
        except Exception as e:
            print(f"  CoreApplication.{attr}() raised: {e}")

# ── 2. Desktop ────────────────────────────────────────────────────────────────
show("MnovaCore.Desktop", MnovaCore.Desktop)

# ── 3. Scan MNova py-scripts dirs for examples ────────────────────────────────
print("\n--- MNova py-scripts files ---")
script_dirs = [
    "/Applications/MestReNova.app/Contents/Resources/py-scripts",
    "/Library/Application Support/Mestrelab Research S.L./MestReNova/py-scripts",
    os.path.expanduser("~/Library/Application Support/Mestrelab Research S.L./MestReNova/py-scripts"),
]
for d in script_dirs:
    if os.path.isdir(d):
        for root, dirs, files in os.walk(d):
            for f in files:
                print(f"  {os.path.join(root, f)}")
    else:
        print(f"  (not found) {d}")

# ── 4. Site-packages contents ─────────────────────────────────────────────────
print("\n--- site-packages top-level ---")
sp = "/Applications/MestReNova.app/Contents/python/lib/python3.10/site-packages"
if os.path.isdir(sp):
    for name in sorted(os.listdir(sp)):
        print(f"  {name}")

# ── 5. Show a sample py-script if any exist ───────────────────────────────────
for d in script_dirs:
    if os.path.isdir(d):
        for root, dirs, files in os.walk(d):
            for f in files:
                if f.endswith(".py"):
                    path = os.path.join(root, f)
                    print(f"\n--- SAMPLE SCRIPT: {path} ---")
                    with open(path) as fh:
                        print(fh.read()[:3000])
                    break
            break
        break

print("\nDone.")
