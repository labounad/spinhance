# Hello World

Three tiny exercises to confirm your setup works **and** to introduce
the three environments you'll be working in during the hackathon:

| # | Where         | File                                | What it proves                                       |
|---|---------------|-------------------------------------|------------------------------------------------------|
| 1 | Claude Code   | [hello_claude.md](hello_claude.md)  | You can drive Claude (incl. slash commands, modes).  |
| 2 | The HPC       | [hello_hpc.slurm](hello_hpc.slurm)  | You can log in to Garibaldi and submit a job.        |
| 3 | AWS           | [test_skill.sh](../test_skill.sh)   | Your AWS SSO works end-to-end (S3 + EC2).            |

Do them in order. Each takes 1–10 minutes. **If something fails, that's
fine** — the error message tells you which step to fix, and you can ask
Claude in this repo "what does this error mean?"

> **Do I need Python?** Not for any of the three. The optional
> [hello_aws.py](hello_aws.py) bonus exercise uses Python+boto3 to
> additionally smoke-test Bedrock, but the main AWS test is pure bash.

---

## 1. Hello, Claude Code

Open this repo in Claude Code (`claude` from the repo root, or use the
VS Code / JetBrains extension), then work through
[hello_claude.md](hello_claude.md). It's a guided tour that has you:

- Ask Claude to read and write files in this repo
- Try the `/hackathon-aws` **slash command**
- Cycle through permission modes with `Shift+Tab` (including
  **auto-accept edits** and **plan mode**)
- Try **auto mode**, where Claude works without stopping for
  clarifying questions

By the end you'll know the moves you'll be doing all weekend.

> **Why first?** From step 2 onward, you'll mostly *ask Claude to do
> the work* rather than typing commands yourself. Getting comfortable
> here makes everything below easier.

---

## 2. Hello, HPC

First time on the cluster? Walk through [HPC_SETUP.md](../HPC_SETUP.md)
to get your account and log in.

Then copy the script to Garibaldi and submit it:

```bash
# From your laptop — replace <username>
scp hello-world/hello_hpc.slurm <username>@garibaldi.scripps.edu:~/

# Then on the cluster (after ssh):
sbatch hello_hpc.slurm
squeue -u $USER          # wait for it to flip from PD to R to disappear
cat hello_hpc_*.out      # see the result
```

You're looking for output that includes your hostname (something like
`compute-XX-YY`), the date, and a "your HPC setup is working" line.

> **What just happened?** You told Slurm (the cluster's job scheduler)
> to run a small shell script on whichever compute node was free. The
> `#SBATCH` lines at the top of the file are how you ask for resources
> — CPUs, RAM, time. The skill at
> [.claude/skills/scripps-garibaldi-hpc/SKILL.md](../.claude/skills/scripps-garibaldi-hpc/SKILL.md)
> teaches Claude how to write these for you, so next time just ask:
> *"give me an sbatch script that runs my training job on one GPU for
> 6 hours."*

---

## 3. Hello, AWS

First time on the AWS account? Walk through [AWS_SETUP.md](../AWS_SETUP.md)
to install the AWS CLI, configure the SSO profile, and log in.

Then, from the repo root:

```bash
./test_skill.sh
```

It walks through every AWS capability the skill exposes — SSO login,
S3 bucket create, S3 round-trip, Amazon Open Data read, EC2 launch,
EC2 terminate — printing PASS/FAIL for each step.

Edit the `PROFILE` variable at the top of the script if your SSO
profile isn't `bgood-scripps`.

If you don't have bash (e.g. native Windows PowerShell without WSL),
ask Claude to walk you through the same checks one command at a time:

> "Open `test_skill.sh` and run me through each step interactively
> in PowerShell using my profile `<your-profile>`."

### Bonus — confirm Bedrock works (optional, needs Python)

You'll use Bedrock to call Claude from inside your hackathon code.
If you have Python 3.10+ available:

```bash
pip install boto3
python hello-world/hello_aws.py --profile <your-profile>
```

Have [`uv`](https://docs.astral.sh/uv/) instead? Skip the install and
just run:

```bash
uv run --with boto3 hello-world/hello_aws.py --profile <your-profile>
```

You should see Claude greet you in the last step. If you do, your
entire AWS pipeline — credentials, S3, Bedrock — is wired up.

---

## What's next?

In the same Claude Code session, try asking:

> "Use the `/hackathon-aws` command to launch a t3.medium instance,
> SSH in, and tell me the kernel version."

> "Write me an `sbatch` script for Garibaldi that runs a quick Python
> training loop on one GPU for 30 minutes."

Claude already has all the context it needs from the slash command and
the HPC skill — you just describe what you want, and it fills in the
boilerplate.
