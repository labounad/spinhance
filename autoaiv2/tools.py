"""autoaiv2.tools — tool implementations and Bedrock JSON schemas for Opus and Sonnet."""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from autoaiv2.idea_spec import IdeaSpec
from autoaiv2.run_monitor import poll, stop, write_pid, read_pid, _best_val_metrics

# ── Module-level state (set by init() at the start of each cycle) ──────────────

_repo_root:    Path         = Path(".")
_spin_systems: str          = ""
_spectra:      str          = ""
_cycle:        int          = 0
_cycle_dir:    Path         = Path(".")
_emitted_idea: IdeaSpec | None = None   # set when Opus calls emit_idea_spec
_cycle_done:   bool         = False     # set when Sonnet calls submit_cycle
_last_run_id:  str | None   = None
_last_run_dir: Path | None  = None
_last_pid:     int | None   = None
_written_files: list[Path]  = []

_EVAL_GUARD = None   # set in init()
MAX_OUTPUT  = 10_000

_WRITE_ALLOWED_PREFIXES: tuple[str, ...] = ("modelv2/", "autoaiv2/runs/")


def init(
    repo_root:    Path,
    spin_systems: str,
    spectra:      str,
    cycle:        int,
    cycle_dir:    Path,
) -> None:
    global _repo_root, _spin_systems, _spectra, _cycle, _cycle_dir
    global _emitted_idea, _cycle_done, _last_run_id, _last_run_dir, _last_pid
    global _written_files, _EVAL_GUARD
    _repo_root    = repo_root
    _spin_systems = spin_systems
    _spectra      = spectra
    _cycle        = cycle
    _cycle_dir    = cycle_dir
    _emitted_idea = None
    _cycle_done   = False
    _last_run_id  = None
    _last_run_dir = None
    _last_pid     = None
    _written_files = []
    _EVAL_GUARD   = (_repo_root / "data" / "eval").resolve()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _trunc(s: str, n: int = MAX_OUTPUT) -> str:
    if len(s) <= n:
        return s
    return f"...[truncated — last {n} chars]\n" + s[-n:]


def _guard_read(path: Path) -> str | None:
    """Return error string if read is blocked, else None."""
    if _EVAL_GUARD and path.is_relative_to(_EVAL_GUARD):
        return "Access denied: eval data is immutable."
    return None


def code_hash() -> str:
    for p in reversed(_written_files):
        if p.suffix == ".py" and p.exists():
            return hashlib.sha256(p.read_bytes()).hexdigest()[:12]
    return ""


# ── Tool implementations ───────────────────────────────────────────────────────

def tool_read_file(path: str) -> str:
    p = (_repo_root / path).resolve()
    err = _guard_read(p)
    if err:
        return err
    if not p.exists():
        return f"File not found: {path}"
    try:
        return _trunc(p.read_text())
    except Exception as e:
        return f"Error reading {path}: {e}"


def tool_list_directory(path: str) -> str:
    p = _repo_root / path
    if not p.exists():
        return f"Directory not found: {path}"
    lines = []
    for item in sorted(p.iterdir()):
        if item.name.startswith("."):
            continue
        tag  = "/" if item.is_dir() else ""
        size = f"  ({item.stat().st_size:,}B)" if item.is_file() else ""
        lines.append(f"{item.name}{tag}{size}")
    return "\n".join(lines) or "(empty)"


def tool_read_s3_json(uri: str) -> str:
    try:
        import boto3, os
        region  = os.environ.get("AWS_REGION", "us-west-2")
        profile = os.environ.get("AWS_PROFILE")
        session = boto3.Session(region_name=region,
                                **( {"profile_name": profile} if profile else {}))
        rest   = uri[5:]
        bucket, _, key = rest.partition("/")
        body   = session.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        return _trunc(body.decode())
    except Exception as e:
        return f"S3 read failed ({uri}): {e}"


def tool_list_s3_prefix(uri: str) -> str:
    try:
        import boto3, os
        region  = os.environ.get("AWS_REGION", "us-west-2")
        profile = os.environ.get("AWS_PROFILE")
        session = boto3.Session(region_name=region,
                                **( {"profile_name": profile} if profile else {}))
        rest   = uri[5:]
        bucket, _, prefix = rest.partition("/")
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        names = []
        pager = session.client("s3").get_paginator("list_objects_v2")
        for page in pager.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
            for cp in page.get("CommonPrefixes", []):
                names.append(cp["Prefix"].rstrip("/").rsplit("/", 1)[-1])
        return "\n".join(names) if names else "(empty)"
    except Exception as e:
        return f"S3 list failed ({uri}): {e}"


def tool_web_search(query: str) -> str:
    """Search arXiv for ML literature. Returns up to 5 results."""
    try:
        import urllib.request
        import urllib.parse
        import xml.etree.ElementTree as ET
        q   = urllib.parse.quote(query)
        url = (f"http://export.arxiv.org/api/query"
               f"?search_query=all:{q}&max_results=5&sortBy=relevance&sortOrder=descending")
        req = urllib.request.Request(url, headers={"User-Agent": "autoaiv2/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            xml_bytes = r.read()
        ns   = "http://www.w3.org/2005/Atom"
        root = ET.fromstring(xml_bytes)
        results = []
        for entry in root.findall(f"{{{ns}}}entry"):
            title   = (entry.findtext(f"{{{ns}}}title")   or "").strip().replace("\n", " ")
            summary = (entry.findtext(f"{{{ns}}}summary") or "").strip().replace("\n", " ")[:300]
            link    = next((l.get("href") for l in entry.findall(f"{{{ns}}}link")
                            if l.get("type") == "text/html"), "")
            year_raw = entry.findtext(f"{{{ns}}}published") or ""
            year     = year_raw[:4]
            results.append(f"**{title}** ({year})\n{summary}...\n{link}")
        return ("\n\n---\n\n".join(results) or "No results found.") if results else "No results."
    except Exception as e:
        return f"Web search failed: {e}"


# ── Opus-only tool ─────────────────────────────────────────────────────────────

def tool_emit_idea_spec(spec_dict: dict) -> str:
    global _emitted_idea
    try:
        _emitted_idea = IdeaSpec.from_dict(spec_dict)
        return "IdeaSpec received. Handing off to Sonnet."
    except Exception as e:
        return f"Invalid IdeaSpec: {e}"


# ── Sonnet-only tools ──────────────────────────────────────────────────────────

def tool_write_file(path: str, content: str) -> str:
    global _written_files
    allowed = any(path.startswith(p) for p in _WRITE_ALLOWED_PREFIXES)
    # also allow writing into the current cycle dir
    cycle_rel = str(_cycle_dir.relative_to(_repo_root))
    if path.startswith(cycle_rel):
        allowed = True
    if not allowed:
        return (f"Write denied: may only write inside {_WRITE_ALLOWED_PREFIXES} "
                f"or {cycle_rel}/")
    local = _repo_root / path
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(content)
    _written_files.append(local)
    return f"Written: {path}"


def tool_run_training(extra_args: str = "") -> str:
    global _last_run_id, _last_run_dir, _last_pid

    ts     = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"cycle_{_cycle:03d}_{ts}"
    run_dir = _cycle_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"

    cmd = (
        f"PYTHONPATH={_repo_root} python -m modelv2.train"
        f" --spin_systems {shlex.quote(_spin_systems)}"
        f" --spectra {shlex.quote(_spectra)}"
        f" --out {shlex.quote(str(run_dir))}"
    )
    if extra_args.strip():
        cmd += " " + extra_args.strip()

    with open(log_path, "w") as log_fh:
        proc = subprocess.Popen(
            cmd, shell=True, stdout=log_fh, stderr=subprocess.STDOUT,
            cwd=str(_repo_root),
        )

    _last_run_id  = run_id
    _last_run_dir = run_dir
    _last_pid     = proc.pid
    write_pid(run_dir, proc.pid)

    return json.dumps({
        "run_id":  run_id,
        "run_dir": str(run_dir.relative_to(_repo_root)),
        "log":     str(log_path.relative_to(_repo_root)),
        "pid":     proc.pid,
        "cmd":     cmd,
    })


def tool_poll_training(run_id: str) -> str:
    run_dir = _resolve_run_dir(run_id)
    if run_dir is None:
        return json.dumps({"error": f"run_id {run_id!r} not found"})

    pid    = read_pid(run_dir) if _last_pid is None else _last_pid
    status = poll(run_dir, pid=pid)

    # append tail of log for context
    log_path = run_dir / "train.log"
    tail = ""
    if log_path.exists():
        lines = log_path.read_text().splitlines()
        tail  = "\n".join(lines[-30:])

    return json.dumps({**status, "log_tail": tail}, default=str)


def tool_stop_training(run_id: str) -> str:
    run_dir = _resolve_run_dir(run_id)
    if run_dir is None:
        return f"run_id {run_id!r} not found"
    pid = read_pid(run_dir) if _last_pid is None else _last_pid
    return stop(run_dir, pid=pid)


def tool_read_diagnostics(run_id: str) -> str:
    run_dir = _resolve_run_dir(run_id)
    if run_dir is None:
        return json.dumps({"error": f"run_id {run_id!r} not found"})
    try:
        import sys
        sys.path.insert(0, str(_repo_root))
        from autoai.run_reader import analyze_run
        analysis = analyze_run(run_dir)
        analysis["available"] = True
        analysis["run_dir"]   = str(run_dir.relative_to(_repo_root))
        return json.dumps(analysis, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "run_dir": str(run_dir)})


def tool_submit_cycle(run_id: str, notes: str = "") -> str:
    global _cycle_done
    _cycle_done = True
    # Save notes alongside cycle artifacts
    (_cycle_dir / "notes.md").write_text(notes)
    return "Cycle submitted. Saving record."


# ── Resolution helper ──────────────────────────────────────────────────────────

def _resolve_run_dir(run_id: str) -> Path | None:
    if _last_run_dir is not None and _last_run_dir.name == run_id:
        return _last_run_dir
    candidate = _cycle_dir / run_id
    if candidate.exists():
        return candidate
    # search broader
    for p in (_repo_root / "autoaiv2" / "runs").glob(f"**/{run_id}"):
        if p.is_dir():
            return p
    return None


# ── Tool schemas ───────────────────────────────────────────────────────────────

_SCHEMA_READ_FILE = {
    "name": "read_file",
    "description": "Read a file from the repository. Path relative to repo root.",
    "input_schema": {"type": "object",
                     "properties": {"path": {"type": "string"}},
                     "required": ["path"]},
}

_SCHEMA_LIST_DIRECTORY = {
    "name": "list_directory",
    "description": "List a directory's contents. Path relative to repo root.",
    "input_schema": {"type": "object",
                     "properties": {"path": {"type": "string"}},
                     "required": ["path"]},
}

_SCHEMA_READ_S3_JSON = {
    "name": "read_s3_json",
    "description": "Read a JSON or JSONL file from S3. Returns raw text.",
    "input_schema": {"type": "object",
                     "properties": {"uri": {"type": "string", "description": "s3:// URI"}},
                     "required": ["uri"]},
}

_SCHEMA_LIST_S3_PREFIX = {
    "name": "list_s3_prefix",
    "description": "List immediate child keys/prefixes under an S3 URI prefix.",
    "input_schema": {"type": "object",
                     "properties": {"uri": {"type": "string"}},
                     "required": ["uri"]},
}

_SCHEMA_WEB_SEARCH = {
    "name": "web_search",
    "description": "Search ML literature on arXiv. Use for finding techniques relevant to the current failure mode.",
    "input_schema": {"type": "object",
                     "properties": {"query": {"type": "string", "description": "Search query"}},
                     "required": ["query"]},
}

_SCHEMA_EMIT_IDEA_SPEC = {
    "name": "emit_idea_spec",
    "description": (
        "Emit a structured IdeaSpec — your final, single best idea for this cycle. "
        "Call this when you have finished analysis and literature search. "
        "It ends your turn and hands the spec to Sonnet for implementation."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "objective":             {"type": "string", "description": "What to change and why."},
            "architecture_changes":  {"type": "string", "description": "Changes to modelv2/model.py, or 'none'."},
            "loss_changes":          {"type": "string", "description": "Loss function changes. Must preserve permutation invariance."},
            "preprocessing_changes": {"type": "string", "description": "Changes to data.py encoding/augmentation, or 'none'."},
            "training_overrides":    {"type": "string", "description": "CLI args string, e.g. '--epochs 60 --batch 64 --lr 3e-4'."},
            "feasibility_notes":     {"type": "string", "description": "Why this won't crash or run forever."},
            "success_criteria":      {"type": "string", "description": "What improvement would constitute success."},
        },
        "required": ["objective", "loss_changes", "success_criteria"],
    },
}

_SCHEMA_WRITE_FILE = {
    "name": "write_file",
    "description": (
        "Write a file in the repo. "
        f"Allowed prefixes: {_WRITE_ALLOWED_PREFIXES}. "
        "No comments unless genuinely non-obvious. No docstrings. No dead code."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path":    {"type": "string", "description": "Repo-relative path."},
            "content": {"type": "string", "description": "Full file content."},
        },
        "required": ["path", "content"],
    },
}

_SCHEMA_RUN_TRAINING = {
    "name": "run_training",
    "description": (
        "Launch modelv2 training in the background. "
        "--spin_systems and --spectra are pre-filled. "
        "Pass extra CLI args (--epochs, --batch, --lr, etc.). "
        "Returns run_id, run_dir, log path, and pid."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "extra_args": {"type": "string",
                           "description": "Extra CLI args, e.g. '--epochs 60 --batch 64 --lr 3e-4 --max_records 50000'"},
        },
        "required": [],
    },
}

_SCHEMA_POLL_TRAINING = {
    "name": "poll_training",
    "description": (
        "Check the status of a running training job. "
        "Returns state (running/finished/stalled/dead), best metrics, and log tail. "
        "Poll every 60–120 seconds."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"run_id": {"type": "string"}},
        "required": ["run_id"],
    },
}

_SCHEMA_STOP_TRAINING = {
    "name": "stop_training",
    "description": "Send SIGTERM to a running training process. Use when plateau or stall is detected.",
    "input_schema": {
        "type": "object",
        "properties": {"run_id": {"type": "string"}},
        "required": ["run_id"],
    },
}

_SCHEMA_READ_DIAGNOSTICS = {
    "name": "read_diagnostics",
    "description": (
        "Parse the canonical run diagnostics for a completed training run. "
        "Returns best metrics, dominant failure mode, and recommendation."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"run_id": {"type": "string"}},
        "required": ["run_id"],
    },
}

_SCHEMA_SUBMIT_CYCLE = {
    "name": "submit_cycle",
    "description": (
        "Finalize this cycle. Call after training is complete and you have read diagnostics. "
        "Provide a one-sentence lesson for the next cycle."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "run_id": {"type": "string"},
            "notes":  {"type": "string", "description": "One-sentence lesson for the next cycle."},
        },
        "required": ["run_id", "notes"],
    },
}

OPUS_TOOLS = [
    _SCHEMA_READ_FILE,
    _SCHEMA_LIST_DIRECTORY,
    _SCHEMA_READ_S3_JSON,
    _SCHEMA_LIST_S3_PREFIX,
    _SCHEMA_WEB_SEARCH,
    _SCHEMA_EMIT_IDEA_SPEC,
]

SONNET_TOOLS = [
    _SCHEMA_READ_FILE,
    _SCHEMA_LIST_DIRECTORY,
    _SCHEMA_WRITE_FILE,
    _SCHEMA_RUN_TRAINING,
    _SCHEMA_POLL_TRAINING,
    _SCHEMA_STOP_TRAINING,
    _SCHEMA_READ_DIAGNOSTICS,
    _SCHEMA_SUBMIT_CYCLE,
]


# ── Dispatch ───────────────────────────────────────────────────────────────────

def dispatch_opus(name: str, inputs: dict) -> str:
    match name:
        case "read_file":       return tool_read_file(inputs["path"])
        case "list_directory":  return tool_list_directory(inputs["path"])
        case "read_s3_json":    return tool_read_s3_json(inputs["uri"])
        case "list_s3_prefix":  return tool_list_s3_prefix(inputs["uri"])
        case "web_search":      return tool_web_search(inputs["query"])
        case "emit_idea_spec":  return tool_emit_idea_spec(inputs)
        case _:                 return f"Unknown Opus tool: {name}"


def dispatch_sonnet(name: str, inputs: dict) -> str:
    match name:
        case "read_file":        return tool_read_file(inputs["path"])
        case "list_directory":   return tool_list_directory(inputs["path"])
        case "write_file":       return tool_write_file(inputs["path"], inputs["content"])
        case "run_training":     return tool_run_training(inputs.get("extra_args", ""))
        case "poll_training":    return tool_poll_training(inputs["run_id"])
        case "stop_training":    return tool_stop_training(inputs["run_id"])
        case "read_diagnostics": return tool_read_diagnostics(inputs["run_id"])
        case "submit_cycle":     return tool_submit_cycle(inputs["run_id"], inputs.get("notes", ""))
        case _:                  return f"Unknown Sonnet tool: {name}"
