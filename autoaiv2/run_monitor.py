"""autoaiv2.run_monitor — plateau and stall detection for a running modelv2 training job."""
from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path

MAX_STALL_SECS   = 600    # no status update in 10 min → stall
MAX_EPOCH_SECS   = 1800   # single epoch taking > 30 min → stall


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def read_status(run_dir: Path) -> dict:
    return _read_json(run_dir / "status.json")


def poll(run_dir: Path, pid: int | None = None) -> dict:
    """
    Return a summary dict describing the current training state.

    Keys:
      state         "running" | "finished" | "stalled" | "dead" | "unknown"
      epoch         current epoch (int or None)
      epochs        total planned epochs (int or None)
      best_score    float or None
      best_epoch    int or None
      best_metrics  dict (from latest val row)
      stall_reason  str or None
      done          bool — training is over (finished, stalled, or dead)
    """
    status = read_status(run_dir)
    state  = status.get("state", "unknown")

    result: dict = {
        "state":       state,
        "epoch":       status.get("epoch"),
        "epochs":      status.get("epochs"),
        "best_score":  status.get("best_score"),
        "best_epoch":  status.get("best_epoch"),
        "best_metrics": _best_val_metrics(run_dir),
        "stall_reason": None,
        "done":        state == "finished",
    }

    if state == "finished":
        return result

    # Check process liveness
    if pid is not None and not _pid_alive(pid):
        result["state"] = "dead"
        result["done"]  = True
        result["stall_reason"] = f"process {pid} is no longer running"
        return result

    # Stall: status.json hasn't been updated in MAX_STALL_SECS
    last_update = status.get("last_update_time")
    if last_update is not None:
        age = time.time() - float(last_update)
        if age > MAX_STALL_SECS:
            result["state"]       = "stalled"
            result["done"]        = True
            result["stall_reason"] = f"no status update for {age:.0f}s (limit {MAX_STALL_SECS}s)"
            return result

    # Stall: single epoch taking too long (infer from events.jsonl timestamps)
    epoch_secs = _latest_epoch_duration(run_dir)
    if epoch_secs is not None and epoch_secs > MAX_EPOCH_SECS:
        result["state"]       = "stalled"
        result["done"]        = True
        result["stall_reason"] = f"epoch wall time {epoch_secs:.0f}s > {MAX_EPOCH_SECS}s limit"
        return result

    return result


def stop(run_dir: Path, pid: int | None = None) -> str:
    """Send SIGTERM to the training process. Returns a status message."""
    # Try PID first
    if pid is not None and _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
            return f"SIGTERM sent to pid {pid}"
        except Exception as e:
            return f"SIGTERM to pid {pid} failed: {e}"

    # Fall back to pid file
    pid_file = run_dir / "pid"
    if pid_file.exists():
        try:
            stored_pid = int(pid_file.read_text().strip())
            if _pid_alive(stored_pid):
                os.kill(stored_pid, signal.SIGTERM)
                return f"SIGTERM sent to pid {stored_pid} (from pid file)"
        except Exception as e:
            return f"Stop via pid file failed: {e}"

    return "No live process found to stop"


def write_pid(run_dir: Path, pid: int) -> None:
    (run_dir / "pid").write_text(str(pid))


def read_pid(run_dir: Path) -> int | None:
    pid_file = run_dir / "pid"
    if not pid_file.exists():
        return None
    try:
        return int(pid_file.read_text().strip())
    except Exception:
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _best_val_metrics(run_dir: Path) -> dict:
    rows = _read_jsonl(run_dir / "metrics.jsonl")
    val_rows = [r for r in rows if r.get("split") == "val"]
    if not val_rows:
        return {}

    def _score(r):
        m = r.get("metrics", {})
        return m.get("shift_mae_ppm", 999) + m.get("j_mae_hz", 999) / 10.0

    return min(val_rows, key=_score).get("metrics", {})


def _latest_epoch_duration(run_dir: Path) -> float | None:
    """Estimate wall time of the most recently completed epoch from events.jsonl."""
    rows = _read_jsonl(run_dir / "events.jsonl")
    # Look for consecutive train_step events within the same epoch
    by_epoch: dict[int, list[float]] = {}
    for r in rows:
        if r.get("event") == "train_step":
            ep = r.get("epoch")
            t  = r.get("time")
            if ep is not None and t is not None:
                by_epoch.setdefault(ep, []).append(float(t))
    if not by_epoch:
        return None
    # Most recent completed epoch
    last_ep = max(by_epoch)
    times   = by_epoch[last_ep]
    if len(times) < 2:
        return None
    return max(times) - min(times)
