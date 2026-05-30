# autoai — Autonomous ML Training Loop

Runs Claude Opus in a continuous loop to design, implement, train, and evaluate
neural network approaches for SpinHance.

## How it works

Each cycle:
1. Opus reads `IDEAS.md` (team's ideas) + the last 3 run summaries + repo structure
2. Opus explores the codebase, writes a training script, and runs it on EC2
3. Opus evaluates results and calls `complete_cycle(metrics, summary)`
4. Everything is archived to `runs/run_NNN_<timestamp>/` and the next cycle begins

## Usage

```bash
# Make sure your SSO token is fresh first
./context/setup_aws_login.sh

# Start the loop (Ctrl-C to stop — terminates EC2 cleanly)
python autoai/orchestrator.py
```

## Adding ideas

Edit `IDEAS.md`. Use the `## Idea name` + **Approach / Motivation** format.
The orchestrator picks up changes at the start of the next cycle — no restart needed.

## Run artifacts

Each `runs/run_NNN_<timestamp>/` contains:
- `training.py` — the script Opus wrote and ran
- `metrics.json` — loss, accuracy, and any other recorded metrics
- `summary.md` — Opus's analysis and recommendations for next cycle

## Config

Edit the top of `orchestrator.py` to change:
- `MODEL` — swap Opus version
- `INSTANCE_TYPE` — upgrade to a GPU instance (e.g. `g4dn.xlarge`) for real training
- `MAX_SUMMARIES` — how many previous summaries Opus reads each cycle
