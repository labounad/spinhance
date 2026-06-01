# autoaiv2 — Design Document

## What this is

`autoaiv2/` is a fully autonomous ML research loop for SpinHance. It runs
**directly on an EC2 GPU instance** (not from a local machine orchestrating EC2
remotely), calls Bedrock for reasoning, and writes all artifacts to S3.

No SSH. No SCP. No instance management. The script is started once; it loops
until a budget limit is hit or the model decides to stop.

```bash
# On the EC2 instance, repo checked out, spinhance conda env active:
AUTOAI_MAX_CYCLES=10 AUTOAI_MAX_HOURS=8 AUTOAI_MAX_SPEND_USD=60 \
  PYTHONPATH=. python autoaiv2/loop.py \
    --spin_systems s3://spinhance-data/spin_systems_chembl.json \
    --spectra      s3://spinhance-data/spectra/90MHz.tar.gz
```

---

## Which training codebase to target

`autoaiv2/` targets **`modelv2/`**.

`modelv2/` is four files (`data.py`, `model.py`, `train.py`, `gui.py`) with no
indirection. An AI can read the entire training codebase in one context window,
understand every design decision, and write modifications without accidentally
breaking something in a file it hasn't seen. That property is essential for
autonomous iteration.

`model/` is more mature but its 20+ module structure means the AI will
constantly mis-predict where changes land. The diagnostic infrastructure in
`model/` (failure classification, run_reader) is valuable and will be adapted
for use here; the training code itself will not be touched.

---

## Loop structure

Each iteration follows the stated workflow exactly:

```
┌─────────────────────────────────────────────────────────────────┐
│  OPUS — analysis and ideation (steps 1–4)                       │
│  1. Read S3 diagnostics from the previous run                   │
│  2. Determine dominant failure modes                            │
│  3. Search ML literature for techniques that address them       │
│  4. Generate candidate ideas (anything goes — architecture,     │
│     loss, preprocessing — as long as inputs/outputs are fixed)  │
│     Output: a structured IdeaSpec                               │
└────────────────────────┬────────────────────────────────────────┘
                         │ IdeaSpec
┌────────────────────────▼────────────────────────────────────────┐
│  SONNET — implementation and execution (steps 5–9)              │
│  5. Trim ideas: flag anything likely to crash or run forever    │
│  6. Implement: write modified train.py / model.py / config      │
│  7. Launch training subprocess                                  │
│  8. Monitor: poll metrics; stop on plateau or epoch stall       │
│  9. Collect diagnostics; evaluate vs prior runs                 │
│     Output: a CycleRecord                                       │
└─────────────────────────────────────────────────────────────────┘
                         │ CycleRecord → S3 + local JSONL
                         └── repeat
```

---

## Model assignments

| Steps | Model | Why |
|---|---|---|
| 1–4 (analysis, literature, ideas) | `us.anthropic.claude-opus-4-6-v1` | Deep reasoning, synthesis across literature |
| 5–9 (trim, implement, run, eval) | `us.anthropic.claude-sonnet-4-6` | Fast, reliable code writing and execution |

Both via AWS Bedrock `bedrock-runtime`, using IAM instance-profile credentials
(no SSO, no browser auth needed on EC2).

---

## Tools

### Opus tools (read + reason + search)

| Tool | Description |
|---|---|
| `read_file(path)` | Read any repo file. `data/eval/` is blocked. |
| `list_directory(path)` | List a repo directory. |
| `read_s3_json(uri)` | Read a JSON artifact from S3. |
| `list_s3_prefix(uri)` | List immediate children of an S3 prefix. |
| `web_search(query)` | Search ML literature (Bedrock tool or search API). |
| `emit_idea_spec(spec)` | Emit the final structured IdeaSpec. Ends Opus's turn. |

### Sonnet tools (implement + run + evaluate)

| Tool | Description |
|---|---|
| `read_file(path)` | Read any repo file. |
| `list_directory(path)` | List a repo directory. |
| `write_file(path, content)` | Write a file. May only write inside `modelv2/` or `autoaiv2/runs/<cycle>/`. |
| `run_training(extra_args)` | Launch `python -m modelv2.train` with the data paths pre-filled and any extra CLI args. Returns `run_id`. Streams stdout to a log file. |
| `poll_training(run_id)` | Read `autoaiv2/runs/<run_id>/status.json` + last N metric rows. Returns current state, best score, plateau signal, stall signal. |
| `stop_training(run_id)` | Write `autoaiv2/runs/<run_id>/stop` sentinel file. The trainer watches for it each epoch. |
| `read_diagnostics(run_id)` | Parse run artifacts and return the structured analysis dict. |
| `submit_cycle(run_id, notes)` | Finalize the cycle: save CycleRecord, sync to S3. |

---

## IdeaSpec contract

Opus ends its turn by calling `emit_idea_spec`. The spec becomes Sonnet's
opening message.

```json
{
  "objective":              "What to change and why — one paragraph",
  "architecture_changes":   "Specific changes to modelv2/model.py, or 'none'",
  "loss_changes":           "Loss function changes. MUST preserve permutation invariance.",
  "preprocessing_changes":  "Changes to data.py target encoding or augmentation, or 'none'",
  "training_overrides":     "--epochs 60 --batch_size 64 --lr 3e-4",
  "feasibility_notes":      "Why this won't crash or run forever",
  "success_criteria":       "What improvement in which metrics would constitute success"
}
```

---

## Training integration

`modelv2/train.py` CLI (no changes to this):

```bash
PYTHONPATH=. python -m modelv2.train \
  --spin_systems <path_or_s3_uri> \
  --spectra      <path_or_s3_uri> \
  [--smoke | --dry-run] \
  [--epochs N] [--batch_size N] [--lr F] [--out dir] \
  [--run_id name] [...]
```

The `run_training` tool pre-fills `--spin_systems` and `--spectra` from the
loop's startup arguments and appends whatever `extra_args` Sonnet provides.

All training output (checkpoints, metrics, diagnostics) goes to
`autoaiv2/runs/<run_id>/` by default (overridable via `--out`).

---

## Monitoring and plateau detection

`poll_training` reads the metrics log written by `modelv2/train.py` and applies:

- **Plateau**: best validation score unchanged for `patience` epochs (default 10)
- **Epoch stall**: current epoch wall time > 3× median of first 3 epochs
- **Hard time cap**: if projected total time > `MAX_EPOCH_SECONDS` (default 1800s),
  stop immediately

When any condition triggers, `stop_training` drops a sentinel file. The trainer
is expected to check for it at the end of each epoch and exit cleanly. If the
trainer ignores it after 2 epochs, `poll_training` sends `SIGTERM` to the
subprocess.

---

## Experiment log

`autoaiv2/experiment_log.jsonl` — one record per cycle, synced to
`s3://spinhance-data/autoaiv2/experiment_log.jsonl` after each cycle.

```json
{
  "cycle": 3,
  "timestamp": "ISO8601",
  "run_id": "cycle_003_20260601_120000",
  "status": "success",
  "idea_spec": { "..." },
  "best_metrics": { "shift_mae_ppm": 0.42, "j_mae_hz": 2.1, "presence_f1": 0.74 },
  "dominant_failure": "large_shift_error",
  "lesson": "one-line takeaway for the next cycle",
  "code_hash": "abc123def456"
}
```

The last 3 records are injected verbatim into Opus's opening message each cycle.

---

## Bedrock resilience

- Exponential backoff on throttle / service errors (5 s → 600 s max)
- Model fallback chain: Opus 4.6 → 4.5 → 4.1 → Sonnet 4.6 → 4.5 → Haiku
- Context trimming on `ValidationException` (input too long)
- IAM instance profile — no interactive auth possible or needed

---

## Budget circuit breaker

All optional; if none are set, the loop runs until `stop_loop` is called.

| Env var | Meaning |
|---|---|
| `AUTOAI_MAX_CYCLES` | Stop after N complete cycles |
| `AUTOAI_MAX_HOURS` | Stop after N wall-clock hours |
| `AUTOAI_MAX_SPEND_USD` | Stop after estimated Bedrock spend |

---

## Hard constraints (never break)

1. **Inputs and outputs are frozen.** The spectrum in (16 384 points) and the
   8×9 matrix + degeneracies out do not change. The viewer and workflow code
   must continue to work.
2. **The AI may change anything else.** Architecture, loss, preprocessing,
   augmentation, optimizer, scheduler, representation — all fair game.
3. **Eval data is immutable.** `data/eval/` is never readable by models.
4. **Metrics are always read from disk.** The model may never type metric values
   into its submission; values are read from the artifact files.
5. **Core training code (`modelv2/*.py`) may be modified**, but changes are
   written to `autoaiv2/runs/<cycle>/` as patched copies, then symlinked or
   passed as overrides. The canonical files are only overwritten when Sonnet is
   explicitly confident the change is sound and records it in the cycle log.

---

## File layout

```
autoaiv2/
  DESIGN.md               ← this file
  loop.py                 ← entry point and cycle runner
  bedrock.py              ← Bedrock client: retry, fallback, context trim
  budget.py               ← BudgetGuard (max cycles / wall time / spend)
  tools.py                ← tool implementations + JSON schemas
  idea_spec.py            ← IdeaSpec dataclass
  cycle_record.py         ← CycleRecord dataclass + JSONL append/load
  run_monitor.py          ← plateau detection, stall detection, stop signal
  runs/                   ← per-cycle artifacts (gitignored)
  experiment_log.jsonl    ← persistent cycle history
```

S3 layout:

```
s3://spinhance-data/
  autoaiv2/
    experiment_log.jsonl
    cycles/<cycle_id>/
      idea_spec.json
      cycle_record.json
      diagnostics_analysis.json
```
