"""
batch_simulate.py
-----------------
MestReNova Python batch script — invoke via:

    /Applications/MestReNova.app/Contents/MacOS/MestReNova \
        --nogui --py simulation/batch_simulate.py \
        -- /path/to/xml_dir /path/to/output_dir

MNova injects its own Python environment; standard library is available.
The 'Mnova' module (or 'mnova') gives access to the application object.

This script:
  1. Iterates all *.xml files in xml_dir
  2. Opens each as a spin simulation (MNova auto-runs the sim on open)
  3. Extracts the intensity array from the simulated spectrum
  4. Writes one .txt file per spectrum (one float per line)
  5. Closes the document
"""

import os
import sys


# ── Argument parsing ──────────────────────────────────────────────────────────
# MNova passes script args after '--' as sys.argv[1:]
def _parse_args():
    args = sys.argv[1:]
    # Strip leading '--' sentinel if present
    if args and args[0] == "--":
        args = args[1:]
    if len(args) >= 2:
        return args[0], args[1]
    # Fallback: hard-code for GUI/console testing
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xml_dir = os.path.join(repo, "data", "processed", "xmls", "100MHz")
    out_dir = os.path.join(repo, "data", "processed", "txt", "100MHz")
    print(f"[batch_simulate] No args supplied; using defaults:\n  xml_dir={xml_dir}\n  out_dir={out_dir}")
    return xml_dir, out_dir


# ── MNova API wrapper ─────────────────────────────────────────────────────────
# MNova 16 exposes its API through the 'Mnova' module (capital M).
# If that import fails we fall back to 'mnova' (lowercase) for older builds.
def _get_app():
    try:
        import Mnova
        return Mnova.app()
    except (ImportError, AttributeError):
        pass
    try:
        import mnova
        return mnova.app()
    except (ImportError, AttributeError):
        pass
    raise ImportError(
        "Could not import MNova Python module.\n"
        "This script must be run via MestReNova --py, not a regular Python interpreter."
    )


# ── Core: simulate one XML, return intensity list ─────────────────────────────
def simulate_xml(app, xml_path: str) -> list:
    """
    Open xml_path in MNova, wait for spin simulation, return intensity array.
    Returns empty list on failure.
    """
    doc = app.open(xml_path)
    if doc is None:
        print(f"  WARN: app.open() returned None for {xml_path}")
        return []

    # Wait for the background spin-simulation thread to finish
    app.waitForAllThreads()   # MNova 16 API; fall back below if needed

    page = doc.currentPage()
    if page is None:
        print(f"  WARN: no page in {xml_path}")
        doc.close(False)
        return []

    # Get the first spectrum item on the page
    spectrum = None
    for i in range(page.itemCount()):
        item = page.item(i)
        # NMR spectrum items expose a .realData() method
        if hasattr(item, "realData"):
            spectrum = item
            break

    if spectrum is None:
        # Alternative: try the application-level activeSpectrum
        spectrum = app.activeSpectrum()

    if spectrum is None:
        print(f"  WARN: no spectrum found in {xml_path}")
        doc.close(False)
        return []

    data = spectrum.realData()   # returns list/array of floats
    doc.close(False)             # close without saving

    if data is None or len(data) == 0:
        print(f"  WARN: empty realData() for {xml_path}")
        return []

    return list(data)


# ── Write intensity array to text file ────────────────────────────────────────
def write_txt(intensities: list, out_path: str) -> bool:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        with open(out_path, "w") as f:
            f.write("\n".join(str(v) for v in intensities))
        return True
    except OSError as e:
        print(f"  ERROR writing {out_path}: {e}")
        return False


# ── Main batch loop ───────────────────────────────────────────────────────────
def main():
    xml_dir, out_dir = _parse_args()
    os.makedirs(out_dir, exist_ok=True)

    xml_files = sorted(f for f in os.listdir(xml_dir) if f.lower().endswith(".xml"))
    if not xml_files:
        print(f"No XML files found in: {xml_dir}")
        return

    print(f"Found {len(xml_files)} XML files. Starting batch simulation...")
    app = _get_app()

    succeeded, failed = 0, 0
    for i, fname in enumerate(xml_files, 1):
        xml_path = os.path.join(xml_dir, fname)
        stem = fname[:-4]  # strip .xml
        out_path = os.path.join(out_dir, stem + ".txt")

        print(f"[{i}/{len(xml_files)}] {fname} ...", end=" ", flush=True)
        data = simulate_xml(app, xml_path)

        if data:
            ok = write_txt(data, out_path)
            if ok:
                print(f"ok ({len(data)} pts)")
                succeeded += 1
            else:
                print("write failed")
                failed += 1
        else:
            print("simulation failed")
            failed += 1

    print(f"\nDone.  Succeeded: {succeeded}  Failed: {failed}")


main()
