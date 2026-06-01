"""autoaiv2.cycle_record — CycleRecord: persistent JSONL log + S3 sync."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

LOG_FILE = Path(__file__).parent / "experiment_log.jsonl"
S3_LOG   = "s3://spinhance-data/autoaiv2/experiment_log.jsonl"


@dataclass
class CycleRecord:
    cycle:            int
    timestamp:        str
    run_id:           str
    status:           str           # "success" | "failure" | "partial"
    idea_spec:        dict
    best_metrics:     dict          # read from disk — never model-typed
    dominant_failure: str
    lesson:           str           # one-line takeaway for next cycle
    code_hash:        str           # SHA-256[:12] of any modified .py, "" if none


def append_record(record: CycleRecord) -> None:
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(asdict(record)) + "\n")
    _sync_to_s3()


def load_records(n: int | None = None) -> list[CycleRecord]:
    if not LOG_FILE.exists():
        return []
    records = [
        CycleRecord(**json.loads(line))
        for line in LOG_FILE.read_text().splitlines()
        if line.strip()
    ]
    return records[-n:] if n is not None else records


def summarize_for_context(n: int = 3) -> str:
    records = load_records(n)
    if not records:
        return "_No experiments logged yet._"
    lines = [
        "| cycle | status | shift_mae | j_mae | dominant_failure | lesson |",
        "|-------|--------|-----------|-------|-----------------|--------|",
    ]
    for r in records:
        m = r.best_metrics
        shift = f"{m['shift_mae_ppm']:.3f}" if "shift_mae_ppm" in m else "—"
        j     = f"{m['j_mae_hz']:.2f}"      if "j_mae_hz"      in m else "—"
        lines.append(
            f"| {r.cycle} | {r.status} | {shift} | {j} "
            f"| {r.dominant_failure} | {r.lesson} |"
        )
    return "\n".join(lines)


def _sync_to_s3() -> None:
    if not LOG_FILE.exists():
        return
    try:
        import boto3
        region  = os.environ.get("AWS_REGION", "us-west-2")
        profile = os.environ.get("AWS_PROFILE")
        session = boto3.Session(region_name=region, **( {"profile_name": profile} if profile else {}))
        bucket, key = "spinhance-data", "autoaiv2/experiment_log.jsonl"
        session.client("s3").put_object(
            Bucket=bucket, Key=key,
            Body=LOG_FILE.read_bytes(),
            ContentType="application/x-ndjson",
        )
    except Exception as e:
        print(f"[cycle_record] S3 sync failed (non-fatal): {e}")
