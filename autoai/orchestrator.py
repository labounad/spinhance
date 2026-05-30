#!/usr/bin/env python3
"""
autoai/orchestrator.py — Autonomous ML training loop for SpinHance.

Each cycle:
  read IDEAS.md + previous summaries → explore codebase → write + run training
  → evaluate → complete_cycle(metrics, summary) → repeat

Run:  python autoai/orchestrator.py
Stop: Ctrl-C  (terminates EC2 instance cleanly)
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import boto3
import botocore.exceptions as _bce

from autoai.config import Role, get_role_config
from autoai.contract import TaskSpec, WorkerResult
from autoai.experiment_log import ExperimentRecord, append_record, summarize_for_context

# ── Config ────────────────────────────────────────────────────────────────────

PROFILE          = "hack-scripps"
REGION           = "us-west-2"
REPO_ROOT        = Path(__file__).parent.parent
AUTOAI_DIR       = REPO_ROOT / "autoai"
IDEAS_FILE       = AUTOAI_DIR / "IDEAS.md"
RUNS_DIR         = AUTOAI_DIR / "runs"
EICE_KEY         = Path(tempfile.gettempdir()) / "autoai-eice-key"
EC2_WORKSPACE    = "/home/ec2-user/workspace"
MAX_SUMMARIES    = 3
MAX_TOOL_OUTPUT  = 8000
MAX_NUDGES       = 3

AMI_ID           = "ami-00563078bca04e287"
INSTANCE_TYPE    = "t3.xlarge"
SUBNET_ID        = "subnet-0096ffc9c05bebab3"
SECURITY_GROUP   = "sg-09d5ef7889a26f56a"
INSTANCE_PROFILE = "hackathon-ec2-profile"

# Backoff limits
THROTTLE_MAX_DELAY = 600   # 10 min
NETWORK_MAX_DELAY  = 120
SSH_RETRIES        = 3

# ── Budget circuit breaker ────────────────────────────────────────────────────

class BudgetExceeded(Exception):
    pass


class BudgetGuard:
    def __init__(
        self,
        max_cycles:    int   | None = None,
        max_wall_secs: float | None = None,
        max_spend_usd: float | None = None,
    ):
        self.max_cycles    = max_cycles
        self.max_wall_secs = max_wall_secs
        self.max_spend_usd = max_spend_usd
        self._start        = time.monotonic()
        self._cycles       = 0
        self._spend        = 0.0

    def check(self) -> None:
        if self.max_cycles is not None and self._cycles >= self.max_cycles:
            raise BudgetExceeded(f"max_cycles={self.max_cycles} reached")
        if self.max_wall_secs is not None:
            elapsed = time.monotonic() - self._start
            if elapsed >= self.max_wall_secs:
                raise BudgetExceeded(
                    f"max_wall_time reached ({elapsed / 3600:.2f}h elapsed)"
                )
        if self.max_spend_usd is not None and self._spend >= self.max_spend_usd:
            raise BudgetExceeded(
                f"max_spend_usd=${self.max_spend_usd:.2f} reached (${self._spend:.2f} spent)"
            )

    def record_cycle(self) -> None:
        self._cycles += 1

    def record_spend(self, usd: float) -> None:
        self._spend += usd
        _log("cost", f"call ~${usd:.4f}  session total ~${self._spend:.3f}")

    def status(self) -> str:
        elapsed = (time.monotonic() - self._start) / 3600
        return (
            f"cycles={self._cycles}/{self.max_cycles or '∞'}  "
            f"wall={elapsed:.2f}h/{(self.max_wall_secs or 0)/3600:.1f}h  "
            f"spend=${self._spend:.2f}/${self.max_spend_usd or '∞'}"
        )


# Approximate Bedrock pricing (USD per million tokens, input/output).
# Used for cost estimation only — not billing-accurate.
_PRICE_PER_MTOK: dict[str, tuple[float, float]] = {
    "us.anthropic.claude-opus-4-6-v1":             (15.0, 75.0),
    "us.anthropic.claude-opus-4-5-20251101-v1:0":  (15.0, 75.0),
    "us.anthropic.claude-opus-4-1-20250805-v1:0":  (15.0, 75.0),
    "us.anthropic.claude-sonnet-4-6":              ( 3.0, 15.0),
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0":( 3.0, 15.0),
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": ( 0.8,  4.0),
}

# ── State ─────────────────────────────────────────────────────────────────────

_instance_id: str | None = None
_session  = boto3.Session(profile_name=PROFILE, region_name=REGION)
_bedrock  = _session.client("bedrock-runtime")
_current_run_dir: Path | None = None
_written_files: list[Path] = []
_budget: BudgetGuard | None = None

# ── Logging ───────────────────────────────────────────────────────────────────

def _log(level: str, msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level.upper():5}] {msg}", flush=True)


# ── Session management ────────────────────────────────────────────────────────

def _refresh_session() -> None:
    global _session, _bedrock
    _session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    _bedrock = _session.client("bedrock-runtime")


def _reauth() -> None:
    _log("warn", "━" * 52)
    _log("warn", "SSO TOKEN EXPIRED — running aws sso login")
    _log("warn", "Complete the browser auth (or copy the URL + code)")
    _log("warn", "to any browser. The orchestrator will wait.")
    _log("warn", "━" * 52)
    # Run interactively — output flows to terminal so user sees the device code URL
    result = subprocess.run(["aws", "sso", "login", "--profile", PROFILE])
    if result.returncode != 0:
        _log("error", "aws sso login exited non-zero — will retry shortly")
        return
    _refresh_session()
    try:
        _session.client("sts").get_caller_identity()
        _log("info", "Re-authentication successful.")
    except Exception as e:
        _log("warn", f"Identity check after re-auth failed: {e}")


def _is_auth_error(exc: Exception) -> bool:
    sso_types = tuple(filter(None, [
        getattr(_bce, "TokenRetrievalError",        None),
        getattr(_bce, "UnauthorizedSSOTokenError",  None),
        getattr(_bce, "SSOTokenLoadError",          None),
        getattr(_bce, "NoCredentialsError",         None),
    ]))
    if isinstance(exc, sso_types):
        return True
    if isinstance(exc, _bce.ClientError):
        code = exc.response["Error"]["Code"]
        return code in ("ExpiredTokenException", "InvalidClientTokenId", "AuthFailure")
    return False


# ── EC2 ───────────────────────────────────────────────────────────────────────

def launch_ec2() -> None:
    global _instance_id
    _log("info", "Launching EC2 instance...")
    ec2 = _session.client("ec2")
    resp = ec2.run_instances(
        ImageId=AMI_ID, InstanceType=INSTANCE_TYPE,
        MinCount=1, MaxCount=1,
        SubnetId=SUBNET_ID, SecurityGroupIds=[SECURITY_GROUP],
        IamInstanceProfile={"Name": INSTANCE_PROFILE},
        MetadataOptions={"HttpTokens": "required"},
        TagSpecifications=[{"ResourceType": "instance",
                            "Tags": [{"Key": "Name", "Value": "spinhance-autoai"}]}],
    )
    _instance_id = resp["Instances"][0]["InstanceId"]
    _log("info", f"  {_instance_id} — waiting for running state...")
    ec2.get_waiter("instance_running").wait(InstanceIds=[_instance_id])
    _log("info", "  Running. Waiting 30s for SSH daemon...")
    time.sleep(30)
    _generate_eice_key()
    _bootstrap_ec2()
    _log("info", f"EC2 ready: {_instance_id}")


def terminate_ec2() -> None:
    if not _instance_id:
        return
    _log("info", f"Terminating {_instance_id}...")
    try:
        _session.client("ec2").terminate_instances(InstanceIds=[_instance_id])
    except Exception as e:
        _log("warn", f"Terminate failed: {e}")


def _ensure_ec2_healthy() -> None:
    global _instance_id
    if not _instance_id:
        launch_ec2()
        return
    try:
        resp  = _session.client("ec2").describe_instances(InstanceIds=[_instance_id])
        state = resp["Reservations"][0]["Instances"][0]["State"]["Name"]
        if state not in ("running", "pending"):
            _log("warn", f"Instance {_instance_id} is {state} — relaunching...")
            _instance_id = None
            launch_ec2()
    except Exception as e:
        _log("warn", f"EC2 health check failed ({e}) — relaunching...")
        _instance_id = None
        launch_ec2()


def _generate_eice_key() -> None:
    if EICE_KEY.exists():
        EICE_KEY.unlink()
        Path(str(EICE_KEY) + ".pub").unlink(missing_ok=True)
    subprocess.run(
        ["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(EICE_KEY), "-q"],
        check=True, input=b"y\n",
    )


def _push_eice_key() -> None:
    subprocess.run([
        "aws", "ec2-instance-connect", "send-ssh-public-key",
        "--instance-id", _instance_id,
        "--instance-os-user", "ec2-user",
        "--ssh-public-key", f"file://{EICE_KEY}.pub",
        "--profile", PROFILE, "--region", REGION,
    ], check=True, capture_output=True)


def _ssh(cmd: str, timeout: int = 3600) -> tuple[str, str, int]:
    _push_eice_key()
    r = subprocess.run(
        [
            "ssh", "-i", str(EICE_KEY),
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
            "-o", (f"ProxyCommand=aws ec2-instance-connect open-tunnel "
                   f"--instance-id {_instance_id} "
                   f"--profile {PROFILE} --region {REGION}"),
            f"ec2-user@{_instance_id}", cmd,
        ],
        capture_output=True, text=True, timeout=timeout,
    )
    return r.stdout, r.stderr, r.returncode


def _ssh_robust(cmd: str, timeout: int = 3600) -> tuple[str, str, int]:
    for attempt in range(SSH_RETRIES):
        try:
            return _ssh(cmd, timeout)
        except subprocess.TimeoutExpired:
            return "", f"Command timed out after {timeout}s", 1
        except Exception as e:
            if attempt < SSH_RETRIES - 1:
                _log("warn", f"SSH attempt {attempt + 1} failed: {e} — checking EC2...")
                time.sleep(10)
                _ensure_ec2_healthy()
            else:
                return "", f"SSH failed after {SSH_RETRIES} attempts: {e}", 1
    return "", "unreachable", 1


def _scp_to_ec2(local: Path, remote: str) -> None:
    for attempt in range(SSH_RETRIES):
        try:
            _push_eice_key()
            subprocess.run(
                [
                    "scp", "-i", str(EICE_KEY),
                    "-o", "StrictHostKeyChecking=no",
                    "-o", (f"ProxyCommand=aws ec2-instance-connect open-tunnel "
                           f"--instance-id {_instance_id} "
                           f"--profile {PROFILE} --region {REGION}"),
                    str(local), f"ec2-user@{_instance_id}:{remote}",
                ],
                check=True, capture_output=True,
            )
            return
        except Exception as e:
            if attempt < SSH_RETRIES - 1:
                _log("warn", f"SCP attempt {attempt + 1} failed: {e} — retrying...")
                time.sleep(10)
                _ensure_ec2_healthy()
            else:
                raise


def _bootstrap_ec2() -> None:
    _log("info", "  Bootstrapping EC2 environment...")
    for cmd in [
        f"mkdir -p {EC2_WORKSPACE}",
        "pip install --quiet --upgrade torch numpy scipy pandas tqdm scikit-learn matplotlib boto3",
    ]:
        _, stderr, rc = _ssh_robust(cmd, timeout=300)
        if rc != 0:
            _log("warn", f"Bootstrap: '{cmd}' exited {rc}: {stderr[:200]}")


# ── Tools ─────────────────────────────────────────────────────────────────────

_EVAL_GUARD_PATHS = (
    REPO_ROOT / "data" / "eval",
    REPO_ROOT / "autoai" / "eval_harness.py",
)

def tool_read_file(path: str) -> str:
    p = (REPO_ROOT / path).resolve()
    for guard in _EVAL_GUARD_PATHS:
        try:
            p.relative_to(guard.resolve())
            return "Access denied: eval data and harness are immutable and not readable by models."
        except ValueError:
            pass
    if not p.exists():
        return f"File not found: {path}"
    try:
        text = p.read_text()
    except Exception as e:
        return f"Error reading {path}: {e}"
    if len(text) > MAX_TOOL_OUTPUT:
        return text[:MAX_TOOL_OUTPUT] + f"\n...[truncated at {MAX_TOOL_OUTPUT} chars]"
    return text


def tool_write_file(path: str, content: str) -> str:
    local = REPO_ROOT / path
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(content)
    _written_files.append(local)
    remote = f"{EC2_WORKSPACE}/{local.name}"
    try:
        _scp_to_ec2(local, remote)
    except Exception as e:
        return f"Written locally to {path}. EC2 sync failed: {e}"
    return f"Written to {path} and synced to EC2:{remote}"


def tool_run_on_ec2(command: str, timeout: int = 3600) -> str:
    stdout, stderr, rc = _ssh_robust(command, timeout=timeout)
    parts = []
    if stdout.strip():
        parts.append(f"stdout:\n{stdout.strip()}")
    if stderr.strip():
        parts.append(f"stderr:\n{stderr.strip()}")
    parts.append(f"exit_code: {rc}")
    result = "\n".join(parts)
    if len(result) > MAX_TOOL_OUTPUT:
        result = f"...[truncated — showing last {MAX_TOOL_OUTPUT} chars]\n" + result[-MAX_TOOL_OUTPUT:]
    return result


def tool_list_directory(path: str) -> str:
    p = REPO_ROOT / path
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


def tool_complete_cycle(metrics: dict, summary: str) -> str:
    assert _current_run_dir is not None
    (_current_run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (_current_run_dir / "summary.md").write_text(summary)
    for src in _written_files:
        if src.exists():
            (_current_run_dir / src.name).write_text(src.read_text())
    return "Cycle saved."


def _auto_lesson(result: WorkerResult) -> str:
    """Derive a one-line lesson from a WorkerResult for the experiment log."""
    if result.status == "failure":
        err = (result.errors or "unknown error")[:80]
        return f"failed — {err}"
    if not result.metrics:
        return f"{result.status} — no metrics recorded"
    top = list(result.metrics.items())[:2]
    return f"{result.status} — " + "  ".join(
        f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
        for k, v in top
    )


def _code_hash() -> str:
    """SHA-256 (12 chars) of the most recently written .py file, or ''."""
    for p in reversed(_written_files):
        if p.suffix == ".py" and p.exists():
            return hashlib.sha256(p.read_bytes()).hexdigest()[:12]
    return ""


def tool_delegate_to_worker(task_spec_dict: dict) -> str:
    assert _current_run_dir is not None
    try:
        spec = TaskSpec.from_dict(task_spec_dict)
    except (KeyError, TypeError) as e:
        return json.dumps(WorkerResult(
            status="failure", artifact_paths={}, metrics={},
            errors=f"Invalid TaskSpec: {e}", notes="",
        ).to_dict())

    (_current_run_dir / "task_spec.json").write_text(json.dumps(spec.to_dict(), indent=2))
    _log("info", f"Delegating to worker — {spec.objective[:80]}")

    result = _run_worker(spec)

    (_current_run_dir / "worker_result.json").write_text(json.dumps(result.to_dict(), indent=2))
    _log("info", f"Worker returned — status={result.status}  metrics={list(result.metrics.keys())}")

    # Run the immutable eval harness if a checkpoint was produced.
    # Called by the harness, not by any model — models cannot invoke this.
    from autoai.eval_harness import run_eval  # imported here to keep the stub isolated
    eval_metrics = run_eval(result.artifact_paths.get("checkpoint"))
    if eval_metrics:
        result.metrics["eval"] = eval_metrics

    # Append to the persistent experiment log
    append_record(ExperimentRecord(
        run_id         = _current_run_dir.name,
        timestamp      = datetime.now().isoformat(),
        status         = result.status,
        task_spec      = spec.to_dict(),
        artifact_paths = result.artifact_paths,
        metrics        = result.metrics,
        errors         = result.errors,
        code_hash      = _code_hash(),
        lesson         = _auto_lesson(result),
    ))

    return json.dumps(result.to_dict())


def tool_stop_loop(reason: str, final_summary: str) -> str:
    assert _current_run_dir is not None
    (_current_run_dir / "summary.md").write_text(final_summary)
    (_current_run_dir / "metrics.json").write_text(
        json.dumps({"stop_reason": reason}, indent=2)
    )
    return "Loop stopped."


# ── Tool schemas ──────────────────────────────────────────────────────────────

_SCHEMA_READ_FILE = {
    "name": "read_file",
    "description": "Read a file from the repository. Path relative to repo root.",
    "input_schema": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
}

_SCHEMA_LIST_DIRECTORY = {
    "name": "list_directory",
    "description": "List directory contents. Path relative to repo root.",
    "input_schema": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
}

_SCHEMA_WRITE_FILE = {
    "name": "write_file",
    "description": (
        "Write a file (path relative to repo root) and sync to EC2 workspace. "
        "Minimal code only: no comments unless non-obvious, no docstrings, no dead code."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    },
}

_SCHEMA_RUN_ON_EC2 = {
    "name": "run_on_ec2",
    "description": (
        "Execute a shell command on the EC2 training instance. "
        "Long runs: nohup python workspace/training.py > /tmp/train.log 2>&1 & echo $! "
        "Poll: tail -n 50 /tmp/train.log && ps -p <pid> -o pid="
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "integer", "description": "Seconds, default 3600"},
        },
        "required": ["command"],
    },
}

# Orchestrator tools — reasoning, delegation, and direct shell access.
ORCHESTRATOR_TOOLS = [
    _SCHEMA_READ_FILE,
    _SCHEMA_LIST_DIRECTORY,
    _SCHEMA_WRITE_FILE,
    _SCHEMA_RUN_ON_EC2,
    {
        "name": "delegate_to_worker",
        "description": (
            "Hand off an implementation task to the worker. "
            "Provide a complete, unambiguous TaskSpec. "
            "The worker will implement it, run it, and return a structured WorkerResult. "
            "You must NOT write any training code yourself — only specify what is needed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task_spec": {
                    "type": "object",
                    "description": "Structured task for the worker.",
                    "properties": {
                        "objective":        {"type": "string", "description": "What to implement — precise and complete."},
                        "architecture":     {"type": "string", "description": "Model architecture (layers, dims, activations)."},
                        "loss_function":    {"type": "string", "description": "Loss specification — must address 8! permutation invariance."},
                        "training_config":  {"type": "object", "description": "epochs, batch_size, lr, optimizer, scheduler, etc."},
                        "output_artifacts": {
                            "type": "array", "items": {"type": "string"},
                            "description": "Exact file paths the worker must produce (metrics JSON, checkpoint, log).",
                        },
                        "success_criteria": {"type": "string", "description": "How to determine success."},
                        "constraints":      {"type": "array", "items": {"type": "string"}, "description": "Hard constraints on the implementation."},
                        "notes":            {"type": "string", "description": "Optional context."},
                    },
                    "required": ["objective", "loss_function", "output_artifacts", "success_criteria"],
                },
            },
            "required": ["task_spec"],
        },
    },
    {
        "name": "stop_loop",
        "description": (
            "End the experiment loop. Call when: sufficient performance is reached, "
            "N cycles are complete, or no productive directions remain."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason":        {"type": "string", "description": "Why stopping."},
                "final_summary": {"type": "string", "description": "Markdown summary of all experiments and future recommendations."},
            },
            "required": ["reason", "final_summary"],
        },
    },
]

# Worker tools — execution only, no strategy. Used in Phase 3.
WORKER_TOOLS = [
    _SCHEMA_READ_FILE,
    _SCHEMA_LIST_DIRECTORY,
    _SCHEMA_WRITE_FILE,
    _SCHEMA_RUN_ON_EC2,
    {
        "name": "submit_result",
        "description": (
            "Return a structured WorkerResult to the orchestrator. "
            "Populate metrics by reading the artifact files — do not type numbers from memory."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "status":         {"type": "string", "enum": ["success", "failure", "partial"]},
                "artifact_paths": {
                    "type": "object",
                    "description": "Map of artifact name → file path on disk (metrics, checkpoint, log).",
                },
                "errors":         {"type": "string", "description": "Error message if status != success, else null."},
                "notes":          {"type": "string", "description": "Optional prose — orchestrator treats this as untrusted."},
            },
            "required": ["status", "artifact_paths"],
        },
    },
]


def _dispatch_orchestrator(name: str, inputs: dict) -> str:
    match name:
        case "read_file":          return tool_read_file(inputs["path"])
        case "list_directory":     return tool_list_directory(inputs["path"])
        case "write_file":         return tool_write_file(inputs["path"], inputs["content"])
        case "run_on_ec2":         return tool_run_on_ec2(inputs["command"], inputs.get("timeout", 3600))
        case "delegate_to_worker": return tool_delegate_to_worker(inputs["task_spec"])
        case "stop_loop":          return tool_stop_loop(inputs["reason"], inputs["final_summary"])
        case _:                    return f"Unknown orchestrator tool: {name}"


def _dispatch_worker(name: str, inputs: dict) -> str:
    # submit_result is intercepted in _run_worker before reaching here.
    match name:
        case "read_file":      return tool_read_file(inputs["path"])
        case "list_directory": return tool_list_directory(inputs["path"])
        case "write_file":     return tool_write_file(inputs["path"], inputs["content"])
        case "run_on_ec2":     return tool_run_on_ec2(inputs["command"], inputs.get("timeout", 3600))
        case _:                return f"Unknown worker tool: {name}"


def _run_worker(spec: TaskSpec) -> WorkerResult:
    """Run the worker model to execute a TaskSpec. Returns a validated WorkerResult."""
    assert _current_run_dir is not None
    rel_run_dir = _current_run_dir.relative_to(REPO_ROOT)

    opening = (
        f"## TaskSpec\n\n```json\n{json.dumps(spec.to_dict(), indent=2)}\n```\n\n"
        f"Save all artifacts under: `{rel_run_dir}/`\n"
        f"Required artifacts: {json.dumps(spec.output_artifacts)}\n"
        f"EC2 workspace: {EC2_WORKSPACE}\n\n"
        "When done, call submit_result. The 'metrics' key in artifact_paths must point "
        "to a JSON file you wrote — the orchestrator reads it directly from disk.\n\n"
        "Begin."
    )
    messages = [{"role": "user", "content": opening}]
    nudges   = 0

    while True:
        payload     = _invoke_bedrock(messages, role=Role.WORKER)
        stop_reason = payload["stop_reason"]
        content     = payload["content"]

        for block in content:
            if block["type"] == "text" and block["text"].strip():
                _log("worker", block["text"].strip())

        messages.append({"role": "assistant", "content": content})

        if stop_reason == "tool_use":
            results = []
            submission: WorkerResult | None = None

            for block in content:
                if block["type"] != "tool_use":
                    continue
                name   = block["name"]
                inputs = block["input"]
                _log("wtool", f"{name}({json.dumps(inputs)[:80]}{'...' if len(json.dumps(inputs)) > 80 else ''})")

                if name == "submit_result":
                    submission = WorkerResult.from_submission(
                        status         = inputs["status"],
                        artifact_paths = inputs.get("artifact_paths", {}),
                        errors         = inputs.get("errors"),
                        notes          = inputs.get("notes", ""),
                        repo_root      = REPO_ROOT,
                    )
                    output = "Result submitted and validated."
                else:
                    output = _dispatch_worker(name, inputs)

                _log("wtool", f"→ {output[:100].replace(chr(10), ' ')}{'...' if len(output) > 100 else ''}")
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": output,
                })

            messages.append({"role": "user", "content": results})

            if submission is not None:
                return submission

        elif stop_reason == "end_turn":
            nudges += 1
            if nudges >= MAX_NUDGES:
                return WorkerResult(
                    status         = "failure",
                    artifact_paths = {},
                    metrics        = {},
                    errors         = f"Worker stopped {MAX_NUDGES}x without calling submit_result.",
                    notes          = "",
                )
            messages.append({
                "role": "user",
                "content": "You have not called submit_result yet. Submit your results now.",
            })


# ── Bedrock call — robust ────────────────────────────────────────────────────

def _trim_messages(messages: list[dict]) -> list[dict]:
    """Halve the length of all tool results to reduce context size."""
    trimmed = []
    for msg in messages:
        if msg["role"] == "user" and isinstance(msg["content"], list):
            new_blocks = []
            for block in msg["content"]:
                if block.get("type") == "tool_result":
                    content = block.get("content", "")
                    if len(content) > 1000:
                        half = len(content) // 4
                        block = dict(block, content=(
                            content[:half] + "\n...[trimmed]...\n" + content[-half:]
                        ))
                new_blocks.append(block)
            msg = dict(msg, content=new_blocks)
        trimmed.append(msg)
    return trimmed


def _invoke_bedrock(messages: list[dict], role: Role = Role.ORCHESTRATOR) -> dict:
    """
    Call Bedrock with full resilience:
      - Exponential backoff on throttling / service errors
      - SSO re-authentication on token expiry
      - Model fallback on unavailability / access denial
      - Context trimming on ValidationException (input too long)
      - Indefinite retry — never gives up
    """
    cfg = get_role_config(role)
    skip_models: set[str] = set()
    throttle_delay = 5.0
    network_delay  = 5.0
    auth_retried   = False

    while True:
        available = [m for m in cfg.model_priority() if m not in skip_models]
        if not available:
            _log("warn", "All models skipped — clearing and waiting 5 min")
            skip_models.clear()
            time.sleep(300)
            throttle_delay = 5.0
            continue

        model = available[0]

        system_prompt = (ORCHESTRATOR_SYSTEM_PROMPT if role == Role.ORCHESTRATOR
                         else WORKER_SYSTEM_PROMPT)
        tools         = (ORCHESTRATOR_TOOLS if role == Role.ORCHESTRATOR
                         else WORKER_TOOLS)
        body: dict = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": cfg.max_tokens,
            "system": system_prompt,
            "tools": tools,
            "messages": messages,
        }
        if cfg.temperature is not None:
            body["temperature"] = cfg.temperature
        if cfg.thinking_enabled:
            body["thinking"] = {"type": "enabled", "budget_tokens": cfg.thinking_budget}
            body["temperature"] = 1.0  # required by API when thinking is on

        try:
            resp = _bedrock.invoke_model(
                modelId=model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
            if model != cfg.model:
                _log("info", f"Using fallback model: {model}")
            throttle_delay = 5.0  # reset on success
            network_delay  = 5.0
            auth_retried   = False
            payload = json.loads(resp["body"].read())
            if _budget is not None:
                usage = payload.get("usage", {})
                in_p, out_p = _PRICE_PER_MTOK.get(model, (15.0, 75.0))
                cost = (usage.get("input_tokens", 0) * in_p
                      + usage.get("output_tokens", 0) * out_p) / 1_000_000
                _budget.record_spend(cost)
            return payload

        except _bce.ClientError as e:
            code = e.response["Error"]["Code"]
            msg  = str(e).lower()

            if code in ("ThrottlingException", "TooManyRequestsException"):
                jitter = random.uniform(0, throttle_delay * 0.2)
                _log("warn", f"Throttled — waiting {throttle_delay + jitter:.0f}s")
                time.sleep(throttle_delay + jitter)
                throttle_delay = min(throttle_delay * 2, THROTTLE_MAX_DELAY)

            elif code in ("ServiceUnavailableException", "InternalServerException"):
                _log("warn", f"Service error ({code}) — waiting {throttle_delay:.0f}s")
                time.sleep(throttle_delay)
                throttle_delay = min(throttle_delay * 2, THROTTLE_MAX_DELAY)

            elif code == "ModelNotReadyException":
                _log("warn", "Model not ready — waiting 30s")
                time.sleep(30)

            elif code in ("ModelErrorException", "ModelStreamErrorException"):
                _log("warn", f"Model error on {model} — trying next")
                skip_models.add(model)

            elif code == "ResourceNotFoundException":
                _log("warn", f"{model} not found — trying next")
                skip_models.add(model)

            elif code == "ValidationException":
                if any(x in msg for x in ("too long", "maximum", "exceeds", "token limit", "input length")):
                    _log("warn", "Context too long — trimming messages")
                    messages = _trim_messages(messages)
                else:
                    raise

            elif code in ("ExpiredTokenException", "InvalidClientTokenId", "AuthFailure"):
                if not auth_retried:
                    _reauth()
                    _log("info", "Retrying after re-auth...")
                    auth_retried = True
                else:
                    _log("error", "Re-auth did not resolve auth error — waiting 5 min")
                    time.sleep(300)
                    auth_retried = False

            elif code == "AccessDeniedException":
                if ("expired" in msg or "token" in msg) and not auth_retried:
                    _reauth()
                    auth_retried = True
                else:
                    _log("warn", f"Access denied on {model} — trying next")
                    skip_models.add(model)

            else:
                _log("error", f"Unhandled ClientError {code} — waiting {throttle_delay:.0f}s")
                time.sleep(throttle_delay)
                throttle_delay = min(throttle_delay * 2, THROTTLE_MAX_DELAY)

        except tuple(filter(None, [
            getattr(_bce, "TokenRetrievalError",       None),
            getattr(_bce, "UnauthorizedSSOTokenError", None),
            getattr(_bce, "SSOTokenLoadError",         None),
            getattr(_bce, "NoCredentialsError",        None),
        ])) as e:
            if not auth_retried:
                _log("warn", f"Auth error ({type(e).__name__}) — re-authenticating")
                _reauth()
                auth_retried = True
            else:
                _log("error", "Persistent auth failure — waiting 5 min")
                time.sleep(300)
                auth_retried = False

        except (_bce.EndpointConnectionError,
                getattr(_bce, "ConnectTimeoutError", type(None)),
                getattr(_bce, "ReadTimeoutError",    type(None))) as e:
            _log("warn", f"Network error ({type(e).__name__}) — waiting {network_delay:.0f}s")
            time.sleep(network_delay)
            network_delay = min(network_delay * 2, NETWORK_MAX_DELAY)

        except Exception as e:
            _log("error", f"Unexpected error: {type(e).__name__}: {e} — waiting 30s")
            time.sleep(30)


# ── System prompts ────────────────────────────────────────────────────────────

ORCHESTRATOR_SYSTEM_PROMPT = """\
You are the ORCHESTRATOR in an autonomous ML research loop for SpinHance — a project \
that trains a neural network to invert low-field ¹H NMR spectra back to spin-system parameters.

== Project ==
Input:  16384-point normalized spectrum vector (90 MHz simulation, 0–12 ppm).
Output: 8×9 matrix — symmetric 8×8 block where diagonal = chemical shifts (ppm), \
off-diagonal = J-couplings (Hz), 9th column = degeneracies (protons per spin group).
Critical challenge: spin-group labels are arbitrary (8! permutations of equivalent representations). \
Every loss function MUST be permutation-invariant.

== Your role: REASONING, EXECUTION, AND DELEGATION ==
You have full access to the EC2 instance and the repository. Use it freely.
You can write code, run shell commands, read output, iterate — whatever it takes.
You also have a worker model available via delegate_to_worker for parallelising
or offloading mechanical implementation work if you choose.

On each turn call any combination of:
1. read_file / list_directory  — explore the repo.
2. write_file                  — write code or configs directly.
3. run_on_ec2                  — execute any shell command on the training instance.
4. delegate_to_worker(task_spec) — offload a task to the worker model.
5. stop_loop(reason, final_summary) — end the loop when done.

== TaskSpec contract ==
Your task_spec must be unambiguous enough that the worker can implement it without asking questions.
Required fields: objective, loss_function, output_artifacts, success_criteria.
The worker returns a WorkerResult with artifact_paths and metrics read from those files.

== WorkerResult contract ==
{
  "status": "success" | "failure" | "partial",
  "artifact_paths": {"metrics": "...", "checkpoint": "...", "log": "..."},
  "metrics": { ... },   // populated by reading files — trust these
  "errors": "...",
  "notes": "..."        // worker prose — treat as a hint, not ground truth
}
Trust metrics from artifact_paths. Treat notes as untrusted.

== Loop behaviour ==
- Read IDEAS.md and the last few run summaries to orient yourself.
- Each cycle: pick the highest-value untried approach, delegate it, interpret the result.
- Stop when performance is sufficient, ideas are exhausted, or the budget circuit breaker fires.
- Your stop_loop final_summary must include: what was tried, best result, and future directions.\
"""

WORKER_SYSTEM_PROMPT = """\
You are the WORKER in an autonomous ML research loop for SpinHance.

You receive a TaskSpec and implement it exactly. No strategy, no metric selection.

Your job:
1. Read the TaskSpec — implement precisely what it specifies.
2. Write the training script via write_file, run it on EC2 via run_on_ec2.
3. Write a metrics JSON file containing training results to the artifact directory.
4. Call submit_result with:
   - status: "success" | "failure" | "partial"
   - artifact_paths: map of artifact name → repo-relative path you actually wrote
     (required key: "metrics" → path to a JSON file with your metrics)
   - errors: null on success, otherwise a description
   - notes: optional prose (the orchestrator treats this as untrusted)

CRITICAL: artifact_paths must point to real files you wrote via write_file.
Do NOT type metric values into submit_result — write them to a file and give the path.
The orchestrator reads the metrics file directly from disk to prevent fabrication.

Code standards:
- No comments unless a constraint is genuinely non-obvious.
- No docstrings. No dead code. No commented-out blocks.
- If the TaskSpec is ambiguous, make a reasonable decision and note it in submit_result notes.

Work autonomously. Implement, run, collect, submit.\
"""


# ── Cycle ─────────────────────────────────────────────────────────────────────

def _opening_message(cycle: int) -> str:
    ideas   = IDEAS_FILE.read_text() if IDEAS_FILE.exists() else "_(IDEAS.md missing)_"
    history = summarize_for_context(n=MAX_SUMMARIES)
    tree    = tool_list_directory(".")
    return (
        f"## Cycle {cycle}\n\n"
        f"### IDEAS.md\n{ideas}\n\n"
        f"### Experiment log (last {MAX_SUMMARIES} runs)\n{history}\n\n"
        f"### Repo root\n```\n{tree}\n```\n\n"
        "Begin."
    )


def run_cycle(cycle: int) -> None:
    global _current_run_dir, _written_files
    _written_files = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _current_run_dir = RUNS_DIR / f"run_{cycle:03d}_{ts}"
    _current_run_dir.mkdir(parents=True)

    _log("info", "=" * 64)
    _log("info", f"Cycle {cycle}   {ts}")
    _log("info", "=" * 64)

    messages = [{"role": "user", "content": _opening_message(cycle)}]
    nudges = 0
    done   = False

    while not done:
        payload     = _invoke_bedrock(messages, role=Role.ORCHESTRATOR)
        stop_reason = payload["stop_reason"]
        content     = payload["content"]

        for block in content:
            if block["type"] == "text" and block["text"].strip():
                _log("opus", block["text"].strip())

        messages.append({"role": "assistant", "content": content})

        if stop_reason == "tool_use":
            results = []
            for block in content:
                if block["type"] != "tool_use":
                    continue
                name    = block["name"]
                inputs  = block["input"]
                preview = json.dumps(inputs)[:100]
                _log("tool", f"{name}({preview}{'...' if len(json.dumps(inputs)) > 100 else ''})")

                output = _dispatch_orchestrator(name, inputs)
                _log("tool", f"→ {output[:120].replace(chr(10), ' ')}{'...' if len(output) > 120 else ''}")

                results.append({
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": output,
                })
                if name == "stop_loop":
                    done = True

            messages.append({"role": "user", "content": results})

        elif stop_reason == "end_turn":
            nudges += 1
            if nudges >= MAX_NUDGES:
                _log("warn", f"Orchestrator stopped {MAX_NUDGES}x without completing — force-ending cycle")
                tool_stop_loop(
                    "forced",
                    "Orchestrator did not produce a stop_loop call — cycle force-ended.",
                )
                done = True
            else:
                _log("info", f"Nudging orchestrator ({nudges}/{MAX_NUDGES})...")
                messages.append({
                    "role": "user",
                    "content": (
                        "You have not called delegate_to_worker or stop_loop. "
                        "Decide on the next action and call one of them now."
                    ),
                })

    _log("info", f"Cycle {cycle} saved → {_current_run_dir.name}")


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    global _budget
    _budget = BudgetGuard(
        max_cycles    = int(os.environ["AUTOAI_MAX_CYCLES"])    if os.environ.get("AUTOAI_MAX_CYCLES")    else None,
        max_wall_secs = float(os.environ["AUTOAI_MAX_HOURS"]) * 3600 if os.environ.get("AUTOAI_MAX_HOURS") else None,
        max_spend_usd = float(os.environ["AUTOAI_MAX_SPEND_USD"]) if os.environ.get("AUTOAI_MAX_SPEND_USD") else None,
    )
    _log("info", f"Budget: {_budget.status()}")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    launch_ec2()
    cycle = len(list(RUNS_DIR.glob("run_*"))) + 1
    try:
        while True:
            try:
                _budget.check()
                run_cycle(cycle)
                _budget.record_cycle()
                _log("info", f"Budget status: {_budget.status()}")
                cycle += 1
            except BudgetExceeded as e:
                _log("info", f"Circuit breaker: {e} — stopping loop.")
                break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                _log("error", f"Cycle {cycle} crashed: {type(e).__name__}: {e}")
                _log("info",  "Waiting 30s before next cycle...")
                time.sleep(30)
                _ensure_ec2_healthy()
    except KeyboardInterrupt:
        _log("info", "Interrupted — stopping cleanly.")
    finally:
        terminate_ec2()


if __name__ == "__main__":
    main()
