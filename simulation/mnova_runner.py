"""
mnova_runner.py
===============
Thin wrapper around the MestReNova command-line interface for batch spin
simulation. This is the only module that knows how to *invoke* MNova; the
higher-level orchestration lives in :mod:`simulation.pipeline`.

How MNova batch simulation works
--------------------------------
1. A JavaScript function ``spinhanceBatch(xmlDir, outDir)`` lives in
   ``simulation/mnova_scripts/spinhanceBatch.qs``. It opens every ``*.xml`` in
   ``xmlDir`` (MNova runs the QM simulation synchronously on open), reads the
   simulated spectrum, and writes one ``<stem>.txt`` of intensities per file.
2. We launch MNova with ``-sf spinhanceBatch,<xmlDir>,<outDir>``.

Hard-won invocation rules (do not regress these)
-------------------------------------------------
- Use a **single dash** ``-sf`` and the bare function **name** (no parentheses),
  with arguments comma-separated: ``-sf spinhanceBatch,A,B``. The ``--sf "fn()"``
  form fails with "Not Found".
- The folder holding ``spinhanceBatch.qs`` must be registered once via
  Edit → Preferences → Scripting → Directories, then MNova restarted. ``-sf``
  only resolves names from registered directories. The file name must equal the
  function name.
- Do **not** pass ``-nogui``. Under it ``Application.quit()`` cannot terminate
  the process (no GUI event loop) and MNova hangs forever, even though output is
  written. With the window visible it runs the whole batch in one launch and
  exits cleanly (exit code 0).
- The ``.qs`` script must have no top-level auto-executing call, or MNova
  crashes at startup.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

__all__ = [
    "MNOVA_DEFAULT",
    "MNOVA_SCRIPTS_DIR",
    "QS_SCRIPT",
    "CONFIG_PATH",
    "check_qs_script",
    "run_mnova_batch",
]

# Default MNova executable location on macOS.
MNOVA_DEFAULT = Path("/Applications/MestReNova.app/Contents/MacOS/MestReNova")

# Canonical, version-controlled location of the batch script. Register THIS
# folder in MNova (Preferences → Scripting → Directories). No copying needed.
MNOVA_SCRIPTS_DIR = Path(__file__).resolve().parent / "mnova_scripts"
QS_SCRIPT = MNOVA_SCRIPTS_DIR / "spinhanceBatch.qs"

# Fallback config read by the .qs script when run from the GUI without args.
CONFIG_PATH = Path.home() / ".spinhance_batch_config.json"


def check_qs_script() -> Path:
    """Verify the in-repo ``.qs`` script exists; remind to register its folder.

    Raises
    ------
    FileNotFoundError
        If ``spinhanceBatch.qs`` is missing.
    """
    if not QS_SCRIPT.exists():
        raise FileNotFoundError(f"Missing script: {QS_SCRIPT}")
    print(f"  Using script: {QS_SCRIPT}")
    print(f"  (Ensure this folder is registered in MNova: {MNOVA_SCRIPTS_DIR})")
    return QS_SCRIPT


def run_mnova_batch(
    mnova_exe: Path,
    xml_dir: Path,
    txt_out_dir: Path,
    timeout_per_file: float = 30.0,
) -> None:
    """Simulate every ``*.xml`` in ``xml_dir``, writing one ``.txt`` per file.

    Parameters
    ----------
    mnova_exe
        Path to the MestReNova executable.
    xml_dir
        Directory of ``mnova-spinsim`` XML files to simulate.
    txt_out_dir
        Destination for the exported intensity ``.txt`` files (created if absent).
    timeout_per_file
        Seconds allowed per file; total timeout is ``max(120, n * this)``.

    Raises
    ------
    RuntimeError
        If MNova exits with a non-zero status.
    subprocess.TimeoutExpired
        If the batch exceeds the total timeout.
    """
    txt_out_dir.mkdir(parents=True, exist_ok=True)

    n_xml = len(list(xml_dir.glob("*.xml")))
    total_timeout = max(120, n_xml * timeout_per_file)

    xml_dir_abs = str(xml_dir.resolve())
    out_dir_abs = str(txt_out_dir.resolve())

    # Fallback config (used when the function is run from the GUI play button).
    CONFIG_PATH.write_text(json.dumps({"xml_dir": xml_dir_abs, "out_dir": out_dir_abs}))
    print(f"  Config written to {CONFIG_PATH}")

    check_qs_script()

    # Single dash, function NAME (no parens), comma-separated args. No -nogui.
    cmd = [
        str(mnova_exe),
        "-sf", f"spinhanceBatch,{xml_dir_abs},{out_dir_abs}",
    ]
    print(f"  Running MNova on {n_xml} files in {xml_dir.name} ...")
    print(f"  CMD: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=total_timeout)

    if result.returncode != 0:
        print("  MNova stderr:", result.stderr[:2000])
        raise RuntimeError(f"MNova exited with code {result.returncode}")

    print(result.stdout.strip())
