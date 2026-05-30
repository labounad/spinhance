# Context — AWS & Claude Reference

| File | Contents |
|------|----------|
| [aws_setup.md](aws_setup.md) | SSO profile config, login, S3 bucket creation/sharing, troubleshooting |
| [bedrock_claude.md](bedrock_claude.md) | Calling Claude via Bedrock: model IDs, invoke_model, streaming, errors |
| [ec2.md](ec2.md) | Launch instances, EICE SSH, Bedrock from EC2, VPC/networking resource IDs |
| [setup_aws_login.sh](setup_aws_login.sh) | Writes `~/.aws/config` (if needed) and runs `aws sso login` |
| [example_claude_call.py](example_claude_call.py) | Minimal working Bedrock call — smoke test after login |

## Quick start

```bash
# First time, or token expired (run every 8–12 hours):
./context/setup_aws_login.sh

# With identity + S3 verification:
./context/setup_aws_login.sh --verify

# Confirm Claude/Bedrock works end-to-end:
python context/example_claude_call.py
```

## Account at a glance

| Field | Value |
|-------|-------|
| Account ID | `127696279288` |
| Region | `us-west-2` |
| SSO start URL | `https://d-9267e96a16.awsapps.com/start` |
| Profile name | `hack-scripps` |
| Shared data bucket | `s3://scrippsresearch-hackathon/` |

## Available Claude models (verified)

| Key | Model ID | Notes |
|-----|----------|-------|
| `haiku` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Fastest / cheapest |
| `sonnet-4-5` | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | |
| `sonnet` | `us.anthropic.claude-sonnet-4-6` | Latest Sonnet |
| `opus-4-1` | `us.anthropic.claude-opus-4-1-20250805-v1:0` | |
| `opus-4-5` | `us.anthropic.claude-opus-4-5-20251101-v1:0` | |
| `opus` | `us.anthropic.claude-opus-4-6-v1` | Latest Opus |
