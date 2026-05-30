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
    timeout: float | None = None,
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
    total_timeout = timeout if timeout is not None else max(120, n_xml * timeout_per_file)

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


def _mnova_pids() -> set[str]:
    """Return the set of running MestReNova process IDs (macOS/Linux)."""
    try:
        out = subprocess.run(["pgrep", "-f", "MestReNova"],
                             capture_output=True, text=True)
        return set(out.stdout.split()) if out.returncode == 0 else set()
    except FileNotFoundError:
        return set()


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
    max_retries: int = 1,
) -> dict:
    """Simulate all ``*.xml`` in ``xml_dir`` across ``workers`` MNova processes.

    The XMLs are round-robin sharded across ``workers`` MNova instances. After
    each pass, molecules with no output are retried (up to ``max_retries`` extra
    passes) — so a license seat cap or a flaky worker won't silently drop
    spectra. ``workers=1`` delegates to a single batch launch.

    Parameters
    ----------
    workers
        Number of concurrent MNova instances. Capped at the number of XMLs.
        Set this to your confirmed concurrent-instance / license-seat limit.
    launcher
        ``"open"`` (default, reliable parallel on macOS) or ``"direct"``.
    timeout_per_file
        Per-file budget; also sets the stall timeout (``4×``) per pass.
    max_retries
        Extra passes over still-missing molecules after the first pass.

    Returns
    -------
    dict
        ``{"workers", "expected", "produced", "complete", "elapsed_s"}``.
    """
    xmls = sorted(Path(xml_dir).glob("*.xml"))
    if not xmls:
        raise FileNotFoundError(f"No XML files found in {xml_dir}")

    txt_out_dir.mkdir(parents=True, exist_ok=True)
    expected = len(xmls)
    workers = max(1, min(workers, expected))

    def _missing(paths: list[Path]) -> list[Path]:
        return [x for x in paths if not (txt_out_dir / f"{x.stem}.txt").exists()]

    t0 = time.monotonic()
    pending = list(xmls)
    # attempt 0 = first pass; up to max_retries additional passes over leftovers.
    for attempt in range(max_retries + 1):
        if not pending:
            break
        if attempt > 0:
            print(f"  Retry {attempt}/{max_retries}: {len(pending)} unfinished "
                  "molecule(s)")
        _one_parallel_pass(mnova_exe, pending, txt_out_dir,
                           workers=min(workers, len(pending)), launcher=launcher,
                           timeout_per_file=timeout_per_file,
                           poll_interval=poll_interval)
        pending = _missing(pending)

    elapsed = time.monotonic() - t0
    produced = expected - len(pending)
    complete = not pending
    status = "OK" if complete else f"INCOMPLETE ({produced}/{expected})"
    print(f"  Parallel total: {status} in {elapsed:.2f}s "
          f"({workers} workers, {launcher})")
    if not complete:
        print(f"    *** {len(pending)} molecule(s) still missing after "
              f"{max_retries} retries. If launcher='open' produced nothing, "
              "try launcher='direct'; if a license cap starves workers, lower "
              "--workers to the confirmed concurrent-instance limit. ***")
    return {"workers": workers, "expected": expected, "produced": produced,
            "complete": complete, "elapsed_s": elapsed}


def _one_parallel_pass(
    mnova_exe: Path,
    xml_paths: list[Path],
    txt_out_dir: Path,
    workers: int,
    launcher: str,
    timeout_per_file: float,
    poll_interval: float,
) -> None:
    """Run a single sharded parallel pass over ``xml_paths`` into ``txt_out_dir``.

    One MNova instance per shard. Outputs are merged into ``txt_out_dir`` as they
    complete. Instances spawned by this pass that don't self-quit are killed at
    the end so they don't hold license seats.
    """
    if workers <= 1:
        # Single instance: copy paths into a temp dir and run the batch script.
        tmp = Path(tempfile.mkdtemp(prefix="spinhance_seq_"))
        try:
            for x in xml_paths:
                shutil.copy2(x, tmp / x.name)
            run_mnova_batch(mnova_exe, tmp, txt_out_dir, timeout_per_file)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return

    check_qs_script()
    shards = _shard(xml_paths, workers)
    expected = len(xml_paths)
    tmp = Path(tempfile.mkdtemp(prefix="spinhance_par_"))
    shard_out_dirs: list[Path] = []
    procs = []

    pids_before = _mnova_pids()
    if pids_before:
        print(f"  WARNING: {len(pids_before)} MestReNova instance(s) already "
              "running — they hold license seats and may starve workers. "
              "Quit them first (pkill -i mestrenova).")

    print(f"  Parallel pass: {expected} sims across {len(shards)} workers "
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

        use_proc_exit = (launcher == "direct")
        total_timeout = max(120.0, expected * timeout_per_file)
        stall_timeout = max(60.0, timeout_per_file * 4)

        t0 = time.monotonic()
        deadline = t0 + total_timeout
        last_count = -1
        last_progress = t0
        while time.monotonic() < deadline:
            produced = sum(len(list(d.glob("*.txt"))) for d in shard_out_dirs)
            if produced != last_count:
                print(f"    progress: {produced}/{expected} "
                      f"(+{produced - max(0, last_count)})")
                last_count = produced
                last_progress = time.monotonic()
            if produced >= expected:
                break
            if use_proc_exit and all(p.poll() is not None for p in procs):
                break
            if time.monotonic() - last_progress > stall_timeout:
                print(f"    *** no new outputs for {stall_timeout:.0f}s — ending "
                      "pass (leftovers will be retried). ***")
                break
            time.sleep(poll_interval)

        for d in shard_out_dirs:
            for t in d.glob("*.txt"):
                shutil.copy2(t, txt_out_dir / t.name)
    finally:
        # Kill any MNova instances THIS pass spawned but that did not self-quit.
        import os
        import signal
        leftover = _mnova_pids() - pids_before
        for pid in leftover:
            try:
                os.kill(int(pid), signal.SIGTERM)
            except (ProcessLookupError, ValueError, PermissionError):
                pass
        if leftover:
            print(f"  Cleaned up {len(leftover)} leftover MNova instance(s).")
        shutil.rmtree(tmp, ignore_errors=True)
