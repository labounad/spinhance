# Calling Claude via AWS Bedrock

All calls go through the `bedrock-runtime` boto3 client using the `hack-scripps` SSO profile.
See `aws_setup.md` for credential setup. Run `aws sso login --profile hack-scripps` before making calls.

---

## Model IDs

All IDs use the `us.*` cross-region inference profile format. Bare model IDs (without `us.`) return a `ValidationException`.

| Alias | Model ID | Notes |
|-------|----------|-------|
| `haiku` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Fastest / cheapest — good default for iteration |
| `sonnet-4-5` | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | |
| `sonnet` | `us.anthropic.claude-sonnet-4-6` | Latest Sonnet |
| `opus-4-1` | `us.anthropic.claude-opus-4-1-20250805-v1:0` | |
| `opus-4-5` | `us.anthropic.claude-opus-4-5-20251101-v1:0` | |
| `opus` | `us.anthropic.claude-opus-4-6-v1` | Latest Opus |

Not available on this account: `claude-opus-4-7`, `claude-opus-4-8` (`AccessDeniedException`); `claude-sonnet-4-20250514`, `claude-opus-4-20250514` (`ResourceNotFoundException`).

---

## Basic pattern

This is the pattern used throughout this project (see `example_claude_call.py`):

```python
import json
import boto3

PROFILE = "hack-scripps"
REGION  = "us-west-2"
MODEL   = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

session = boto3.Session(profile_name=PROFILE, region_name=REGION)
bedrock = session.client("bedrock-runtime")

resp = bedrock.invoke_model(
    modelId=MODEL,
    contentType="application/json",
    accept="application/json",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "messages": [{"role": "user", "content": "Your prompt here."}],
    }),
)
text = json.loads(resp["body"].read())["content"][0]["text"].strip()
print(text)
```

---

## With a system prompt

```python
body = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 512,
    "system": "You are an NMR spectroscopy expert. Be concise and precise.",
    "messages": [{"role": "user", "content": "What does a 2 Hz coupling constant indicate?"}],
}
```

---

## Multi-turn conversation

```python
messages = [
    {"role": "user",      "content": "What is a spin system?"},
    {"role": "assistant", "content": "A spin system is a set of magnetically coupled nuclei..."},
    {"role": "user",      "content": "How does field strength affect it?"},
]
body = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 512,
    "messages": messages,
}
```

---

## Streaming

```python
resp = bedrock.invoke_model_with_response_stream(
    modelId=MODEL,
    contentType="application/json",
    accept="application/json",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "messages": [{"role": "user", "content": "Explain J-coupling in 3 paragraphs."}],
    }),
)
for event in resp["body"]:
    chunk = json.loads(event["chunk"]["bytes"])
    if chunk.get("type") == "content_block_delta":
        print(chunk["delta"].get("text", ""), end="", flush=True)
print()
```

---

## Error handling

```python
from botocore.exceptions import BotoCoreError, ClientError

try:
    resp = bedrock.invoke_model(...)
except ClientError as e:
    code = e.response["Error"]["Code"]
    if code == "ThrottlingException":
        # Back off and retry
        pass
    elif code == "AccessDeniedException":
        # Token expired — run: aws sso login --profile hack-scripps
        pass
    else:
        raise
```

Common errors:

| Code | Cause |
|------|-------|
| `ThrottlingException` | Rate limit — back off and retry |
| `AccessDeniedException` | SSO token expired — re-run `aws sso login` |
| `ValidationException` | Bad request — wrong model ID or malformed message |
| `ModelErrorException` | Model-side error — safe to retry |

---

## Smoke test

`example_claude_call.py` (in this directory) runs a single-turn call asking Claude for 8-spin-group SMILES — use it to confirm the full stack is working after login:

```bash
python context/example_claude_call.py
```
