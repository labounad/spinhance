"""
autoaiv2/loop.py — Autonomous ML research loop for SpinHance.

Runs directly on EC2. No SSH, no SCP. Calls Bedrock for reasoning,
launches modelv2 training as a subprocess, writes artifacts to S3.

Usage:
    PYTHONPATH=. python autoaiv2/loop.py \\
        --spin_systems s3://spinhance-data/spin_systems_chembl.json \\
        --spectra      s3://spinhance-data/spectra/90MHz.tar.gz

    AUTOAI_MAX_CYCLES=5 AUTOAI_MAX_HOURS=8 AUTOAI_MAX_SPEND_USD=60 \\
        PYTHONPATH=. python autoaiv2/loop.py --spin_systems ... --spectra ...
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from autoaiv2.bedrock import BedrockClient, _log
from autoaiv2.budget import BudgetGuard, BudgetExceeded
from autoaiv2.cycle_record import CycleRecord, append_record, summarize_for_context
from autoaiv2.idea_spec import IdeaSpec
from autoaiv2 import tools as T

REPO_ROOT  = Path(__file__).parent.parent
RUNS_DIR   = Path(__file__).parent / "runs"
IDEAS_FILE = REPO_ROOT / "autoai" / "IDEAS.md"
MAX_NUDGES = 3
S3_TRAINING_ROOT = "s3://spinhance-data/autoaiv2"


# ── System prompts ─────────────────────────────────────────────────────────────

_OPUS_SYSTEM = """\
You are the ANALYSIS agent in an autonomous ML research loop for SpinHance.

== Project ==
SpinHance trains a neural network to invert low-field ¹H NMR spectra to spin-system parameters.
Input:  16384-point normalized spectrum (90 MHz, 0–12 ppm).
Output: 8 spin groups → chemical shifts (ppm), scalar couplings J (Hz),
        coupling presence flags, proton degeneracies.
        Represented as an 8×9 matrix (8×8 shift/J + degeneracy column).

Critical: spin-group labels are arbitrary (8! permutations). Every loss must be
permutation-invariant. The model can use canonical ordering as a convenience
but the loss must not rely on label identity.

== Your role: steps 1–4 ==
1. Read S3 diagnostics and experiment history.
2. Identify the dominant failure mode.
3. Search ML literature for techniques that address it.
4. Emit a single, well-reasoned IdeaSpec via emit_idea_spec.

== Constraints ==
- Do NOT emit multiple IdeaSpecs. Pick your single best idea.
- Inputs (16384-pt spectrum) and outputs (8×9 matrix) are FROZEN. Do not change them.
- The codebase is in modelv2/ (4 files). Read it before proposing changes.
- Ideas may change anything else: architecture, loss, preprocessing, augmentation,
  optimizer, scheduler, representation — as long as it runs on a single GPU overnight.
- Be concrete: specify which functions to change and how.

== Tools ==
read_file, list_directory: explore the repo.
read_s3_json, list_s3_prefix: read training artifacts from S3.
web_search: search arXiv for relevant techniques.
emit_idea_spec: emit your final idea. This ends your turn.\
"""

_SONNET_SYSTEM = """\
You are the IMPLEMENTATION agent in an autonomous ML research loop for SpinHance.

== Your role: steps 5–9 ==
5. Trim the IdeaSpec: flag anything that will crash or run forever. If so, simplify before implementing.
6. Implement: write modified modelv2/model.py or modelv2/train.py if needed.
7. Launch training via run_training.
8. Monitor via poll_training (every ~90 seconds). Stop via stop_training if:
   - plateau detected (state == "finished" from early stopping), or
   - stall detected (state == "stalled" or "dead"), or
   - epochs taking unreasonably long.
9. Read diagnostics via read_diagnostics. Evaluate vs prior runs. Call submit_cycle.

== Constraints ==
- Write modified files only to modelv2/ or autoaiv2/runs/.
- If you modify modelv2/model.py or modelv2/train.py, write the full modified file.
  The original is not modified unless you explicitly overwrite it.
- Do NOT hardcode metric values into submit_cycle — read them from read_diagnostics.
- No comments unless genuinely non-obvious. No docstrings. No dead code.
- poll_training returns a log tail and status. Use it to judge when to stop.
- If training finishes naturally (state == "finished"), do not call stop_training.

== Tools ==
read_file, list_directory: explore the repo.
write_file: write to modelv2/ or autoaiv2/runs/ only.
run_training: launch training in background; returns run_id.
poll_training: check status; call every 90s until done or stalled.
stop_training: send SIGTERM; use on plateau or stall.
read_diagnostics: parse run artifacts; call after training ends.
submit_cycle: finalize; required to complete the cycle.\
"""


# ── Opening messages ───────────────────────────────────────────────────────────

def _opus_opening(cycle: int) -> str:
    ideas   = (IDEAS_FILE.read_text() if IDEAS_FILE.exists()
               else "_IDEAS.md not found — use web_search and repo exploration._")
    history = summarize_for_context(n=3)

    latest_diag = "_No prior run diagnostics available._"
    try:
        from autoai.run_reader import find_latest_run, analyze_run
        latest = find_latest_run(REPO_ROOT / "modelv2" / "runs")
        if latest is None:
            # try autoaiv2 runs
            latest = find_latest_run(RUNS_DIR)
        if latest is not None:
            analysis   = analyze_run(latest)
            latest_diag = json.dumps(analysis, indent=2, default=str)
    except Exception as e:
        latest_diag = f"(run_reader error: {e})"

    repo_tree = T.tool_list_directory(".")

    return (
        f"## Cycle {cycle}\n\n"
        f"### Experiment history (last 3 cycles)\n{history}\n\n"
        f"### Latest run diagnostics\n```json\n{latest_diag}\n```\n\n"
        f"### Repo root\n```\n{repo_tree}\n```\n\n"
        f"### IDEAS.md (ranked experiment menu)\n{ideas}\n\n"
        "Analyze the diagnostics, search literature if needed, then call emit_idea_spec "
        "with your single best idea for this cycle."
    )


def _sonnet_opening(cycle: int, idea: IdeaSpec) -> str:
    return (
        f"## Cycle {cycle} — Implementation\n\n"
        f"{idea.as_prompt()}\n\n"
        "The training entry point is:\n"
        "```\npython -m modelv2.train --spin_systems <pre-filled> --spectra <pre-filled>"
        " --out <auto-set> [extra args]\n```\n"
        "Use run_training with extra_args for any overrides.\n\n"
        "Read modelv2/model.py and modelv2/train.py before writing any changes. "
        "Implement, train, monitor, read diagnostics, then call submit_cycle."
    )


# ── Phase runners ──────────────────────────────────────────────────────────────

def run_opus_phase(cycle: int, client: BedrockClient) -> IdeaSpec:
    """Run Opus (steps 1–4). Returns an IdeaSpec."""
    messages = [{"role": "user", "content": _opus_opening(cycle)}]
    nudges   = 0

    while True:
        payload     = client.invoke_opus(messages, T.OPUS_TOOLS, _OPUS_SYSTEM)
        stop_reason = payload["stop_reason"]
        content     = payload["content"]

        for block in content:
            if block["type"] == "text" and block["text"].strip():
                _log("opus", block["text"][:300].replace("\n", " "))

        messages.append({"role": "assistant", "content": content})

        if stop_reason == "tool_use":
            results = []
            emit_done = False
            for block in content:
                if block["type"] != "tool_use":
                    continue
                name   = block["name"]
                inputs = block["input"]
                _log("otool", f"{name}({json.dumps(inputs)[:80]})")
                output = T.dispatch_opus(name, inputs)
                _log("otool", f"→ {output[:120].replace(chr(10), ' ')}")
                results.append({"type": "tool_result",
                                 "tool_use_id": block["id"],
                                 "content": output})
                if name == "emit_idea_spec" and T._emitted_idea is not None:
                    emit_done = True

            messages.append({"role": "user", "content": results})
            if emit_done:
                spec = T._emitted_idea
                T._emitted_idea = None
                return spec

        elif stop_reason == "end_turn":
            nudges += 1
            if nudges >= MAX_NUDGES:
                _log("warn", "Opus stopped without emitting an idea — using fallback")
                return IdeaSpec(
                    objective             = "Tune loss weights: increase shift weight to 2.0, based on large_shift_error failure mode.",
                    architecture_changes  = "none",
                    loss_changes          = "Set loss_weights shift=2.0 in TrainConfig.",
                    preprocessing_changes = "none",
                    training_overrides    = "--epochs 60",
                    feasibility_notes     = "Weight change only; cannot crash.",
                    success_criteria      = "shift_mae_ppm < 0.35",
                )
            messages.append({"role": "user", "content":
                              "You have not called emit_idea_spec. Call it now with your best idea."})


def run_sonnet_phase(cycle: int, idea: IdeaSpec, client: BedrockClient,
                     cycle_dir: Path) -> CycleRecord:
    """Run Sonnet (steps 5–9). Returns a CycleRecord."""
    messages = [{"role": "user", "content": _sonnet_opening(cycle, idea)}]
    nudges   = 0
    last_run_id = None

    while True:
        payload     = client.invoke_sonnet(messages, T.SONNET_TOOLS, _SONNET_SYSTEM)
        stop_reason = payload["stop_reason"]
        content     = payload["content"]

        for block in content:
            if block["type"] == "text" and block["text"].strip():
                _log("sonnet", block["text"][:300].replace("\n", " "))

        messages.append({"role": "assistant", "content": content})

        if stop_reason == "tool_use":
            results = []
            submit_done = False
            for block in content:
                if block["type"] != "tool_use":
                    continue
                name   = block["name"]
                inputs = block["input"]
                _log("stool", f"{name}({json.dumps(inputs)[:80]})")
                output = T.dispatch_sonnet(name, inputs)
                _log("stool", f"→ {output[:120].replace(chr(10), ' ')}")
                results.append({"type": "tool_result",
                                 "tool_use_id": block["id"],
                                 "content": output})
                if name == "run_training":
                    try:
                        last_run_id = json.loads(output).get("run_id")
                    except Exception:
                        pass
                if name == "submit_cycle" and T._cycle_done:
                    submit_done = True

            messages.append({"role": "user", "content": results})

            if submit_done:
                return _build_cycle_record(cycle, idea, last_run_id, cycle_dir)

        elif stop_reason == "end_turn":
            nudges += 1
            if nudges >= MAX_NUDGES:
                _log("warn", f"Sonnet stopped {MAX_NUDGES}x without submitting — force-closing cycle")
                return _build_cycle_record(cycle, idea, last_run_id, cycle_dir,
                                           status="partial")
            messages.append({"role": "user", "content":
                              "You have not called submit_cycle. "
                              "Read diagnostics from the last run and call submit_cycle now."})


def _build_cycle_record(
    cycle: int,
    idea: IdeaSpec,
    run_id: str | None,
    cycle_dir: Path,
    status: str = "success",
) -> CycleRecord:
    best_metrics     = {}
    dominant_failure = "unknown"

    if run_id is not None:
        run_dir = T._resolve_run_dir(run_id)
        if run_dir and run_dir.exists():
            best_metrics = T._best_val_metrics(run_dir)
            try:
                from autoai.run_reader import analyze_run
                analysis = analyze_run(run_dir)
                failure  = analysis.get("failure_summary") or {}
                dominant_failure = failure.get("dominant_failure", "none")
                if not best_metrics:
                    best_metrics = analysis.get("best_metrics") or {}
            except Exception:
                pass

    # notes.md written by submit_cycle
    notes_path = cycle_dir / "notes.md"
    lesson = notes_path.read_text().strip() if notes_path.exists() else ""
    lesson = lesson[:120]  # keep it a one-liner

    return CycleRecord(
        cycle            = cycle,
        timestamp        = datetime.now(timezone.utc).isoformat(),
        run_id           = run_id or f"cycle_{cycle:03d}_no_run",
        status           = status,
        idea_spec        = idea.to_dict(),
        best_metrics     = best_metrics,
        dominant_failure = dominant_failure,
        lesson           = lesson,
        code_hash        = T.code_hash(),
    )


# ── Main loop ──────────────────────────────────────────────────────────────────

def run_cycle(cycle: int, client: BedrockClient) -> None:
    ts        = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    cycle_dir = RUNS_DIR / f"cycle_{cycle:03d}_{ts}"
    cycle_dir.mkdir(parents=True, exist_ok=True)

    _log("info", "=" * 64)
    _log("info", f"Cycle {cycle}  {ts}")
    _log("info", "=" * 64)

    T.init(REPO_ROOT, _args.spin_systems, _args.spectra, cycle, cycle_dir)

    # Steps 1–4: Opus
    _log("info", "--- Opus phase (analysis + ideation) ---")
    idea = run_opus_phase(cycle, client)
    (cycle_dir / "idea_spec.json").write_text(
        json.dumps(idea.to_dict(), indent=2))
    _log("info", f"IdeaSpec: {idea.objective[:100]}")

    # Steps 5–9: Sonnet
    _log("info", "--- Sonnet phase (implementation + execution) ---")
    record = run_sonnet_phase(cycle, idea, client, cycle_dir)
    (cycle_dir / "cycle_record.json").write_text(
        json.dumps(record.__dict__, indent=2, default=str))

    append_record(record)
    _log("info", f"Cycle {cycle} done — {record.status}  "
                 f"shift={record.best_metrics.get('shift_mae_ppm', '?')}  "
                 f"j={record.best_metrics.get('j_mae_hz', '?')}  "
                 f"failure={record.dominant_failure}")


_args = None


def main() -> None:
    global _args
    ap = argparse.ArgumentParser(description="AutoAI v2 — autonomous SpinHance training loop")
    ap.add_argument("--spin_systems", required=True,
                    help="Path or s3:// URI to spin_systems JSON/tar.gz")
    ap.add_argument("--spectra", required=True,
                    help="Path or s3:// URI to 90MHz.tar.gz spectra archive")
    _args = ap.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    budget = BudgetGuard.from_env()
    _log("info", f"Budget: {budget.status()}")

    client = BedrockClient(budget=budget)
    cycle  = len(list(RUNS_DIR.glob("cycle_*"))) + 1

    try:
        while True:
            try:
                budget.check()
                run_cycle(cycle, client)
                budget.record_cycle()
                _log("info", f"Budget status: {budget.status()}")
                cycle += 1
            except BudgetExceeded as e:
                _log("info", f"Budget limit: {e} — stopping.")
                break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                import traceback
                _log("error", f"Cycle {cycle} crashed: {type(e).__name__}: {e}")
                traceback.print_exc()
                _log("info", "Waiting 30s before next cycle...")
                import time
                time.sleep(30)
                cycle += 1
    except KeyboardInterrupt:
        _log("info", "Interrupted — stopping cleanly.")
        sys.exit(0)


if __name__ == "__main__":
    main()
