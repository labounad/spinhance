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
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

__all__ = [
    "MNOVA_DEFAULT",
    "MNOVA_SCRIPTS_DIR",
    "QS_SCRIPT",
    "CONFIG_PATH",
    "check_qs_script",
    "run_mnova_batch",
    "run_mnova_parallel",
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


# ── Parallel execution across multiple MNova instances ────────────────────────

def _shard(items: list, workers: int) -> list[list]:
    """Round-robin partition ``items`` into ``workers`` lists.

    Round-robin (not contiguous) so the expensive high-field simulations are
    spread evenly across workers rather than piling onto one shard.
    """
    workers = max(1, min(workers, len(items)))
    return [items[i::workers] for i in range(workers)]


def _app_bundle(mnova_exe: Path) -> str:
    """Return the ``.app`` bundle path for ``open -na`` (macOS)."""
    for parent in Path(mnova_exe).parents:
        if parent.suffix == ".app":
            return str(parent)
    return str(mnova_exe)


def _launch_cmd(mnova_exe: Path, launcher: str, xml_dir: Path, out_dir: Path) -> list[str]:
    """Build the launch command for one shard.

    launcher="open"   — ``open -na <App> --args -sf …`` forces a NEW macOS
                        instance (bypasses single-instance handoff). The ``open``
                        command itself returns immediately; the instance runs in
                        the background, so completion is detected by polling
                        output files.
    launcher="direct" — run the binary directly; the process stays alive until
                        that MNova exits. Use only if concurrent direct launches
                        do NOT hand off to one instance on your machine.
    """
    arg = f"spinhanceBatch,{xml_dir},{out_dir}"
    if launcher == "open":
        return ["open", "-na", _app_bundle(mnova_exe), "--args", "-sf", arg]
    if launcher == "direct":
        return [str(mnova_exe), "-sf", arg]
    raise ValueError(f"unknown launcher: {launcher!r} (use 'open' or 'direct')")


def run_mnova_parallel(
    mnova_exe: Path,
    xml_dir: Path,
    txt_out_dir: Path,
    workers: int = 4,
    launcher: str = "open",
    timeout_per_file: float = 30.0,
    poll_interval: float = 1.0,
) -> dict:
    """Simulate all ``*.xml`` in ``xml_dir`` across ``workers`` MNova processes.

    The XMLs are round-robin sharded; each shard is copied to a temp directory
    and simulated by its own MNova instance. Outputs are merged into
    ``txt_out_dir``. ``workers=1`` delegates to :func:`run_mnova_batch`.

    Parameters
    ----------
    workers
        Number of concurrent MNova instances. Capped at the number of XMLs.
    launcher
        ``"open"`` (default, reliable parallel on macOS) or ``"direct"``.
    timeout_per_file
        Per-file budget; the overall wait scales with files-per-worker.

    Returns
    -------
    dict
        ``{"workers", "expected", "produced", "complete", "elapsed_s"}``.
    """
    xmls = sorted(Path(xml_dir).glob("*.xml"))
    if not xmls:
        raise FileNotFoundError(f"No XML files found in {xml_dir}")

    txt_out_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(workers, len(xmls)))

    if workers == 1:
        t0 = time.perf_counter()
        run_mnova_batch(mnova_exe, xml_dir, txt_out_dir, timeout_per_file)
        produced = len(list(txt_out_dir.glob("*.txt")))
        return {"workers": 1, "expected": len(xmls), "produced": produced,
                "complete": produced >= len(xmls),
                "elapsed_s": time.perf_counter() - t0}

    check_qs_script()
    shards = _shard(xmls, workers)
    expected = len(xmls)
    tmp = Path(tempfile.mkdtemp(prefix="spinhance_par_"))
    shard_out_dirs: list[Path] = []
    procs = []

    print(f"  Parallel: {expected} sims across {len(shards)} workers "
          f"(launcher={launcher})")
    try:
        for wi, shard in enumerate(shards):
            sdir = tmp / f"shard_{wi}" / "xml"
            sodir = tmp / f"shard_{wi}" / "out"
            sdir.mkdir(parents=True)
            sodir.mkdir(parents=True)
            for x in shard:
                shutil.copy2(x, sdir / x.name)
            shard_out_dirs.append(sodir)
            procs.append(subprocess.Popen(_launch_cmd(mnova_exe, launcher, sdir, sodir)))

        # Wait for completion. For "open", the launch commands return at once and
        # work continues in the background, so we poll output counts. For
        # "direct", processes stay alive, so we can also break when all exit.
        use_proc_exit = (launcher == "direct")
        total_timeout = max(120.0, (expected / workers) * timeout_per_file * 2)
        # Use a single monotonic clock for both the deadline and elapsed time.
        t0 = time.monotonic()
        deadline = t0 + total_timeout
        while time.monotonic() < deadline:
            produced = sum(len(list(d.glob("*.txt"))) for d in shard_out_dirs)
            if produced >= expected:
                break
            if use_proc_exit and all(p.poll() is not None for p in procs):
                break
            time.sleep(poll_interval)
        elapsed = time.monotonic() - t0

        # Merge outputs.
        produced = 0
        for d in shard_out_dirs:
            for t in d.glob("*.txt"):
                shutil.copy2(t, txt_out_dir / t.name)
                produced += 1

        complete = produced >= expected
        status = "OK" if complete else f"INCOMPLETE ({produced}/{expected})"
        print(f"  Parallel done: {status} in {elapsed:.2f}s")
        if not complete:
            print("    *** Some shards did not finish. If using launcher='open' "
                  "and 0 outputs appeared, -sf may not pass through `open --args` "
                  "on your machine — try launcher='direct'. ***")
        return {"workers": workers, "expected": expected, "produced": produced,
                "complete": complete, "elapsed_s": elapsed}
    finally:
        for p in procs:
            if p.poll() is None:
                continue
        shutil.rmtree(tmp, ignore_errors=True)
