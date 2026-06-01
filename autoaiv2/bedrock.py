"""autoaiv2.bedrock — Bedrock client with retry, model fallback, and context trimming."""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime

import boto3
import botocore.exceptions as _bce

from autoaiv2.budget import BudgetGuard

# ── Model config ───────────────────────────────────────────────────────────────

OPUS_MODEL   = "us.anthropic.claude-opus-4-6-v1"
SONNET_MODEL = "us.anthropic.claude-sonnet-4-6"

OPUS_FALLBACKS = (
    "us.anthropic.claude-opus-4-5-20251101-v1:0",
    "us.anthropic.claude-opus-4-1-20250805-v1:0",
    "us.anthropic.claude-sonnet-4-6",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
)
SONNET_FALLBACKS = (
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
)

_PRICE_PER_MTOK: dict[str, tuple[float, float]] = {
    "us.anthropic.claude-opus-4-6-v1":              (15.0, 75.0),
    "us.anthropic.claude-opus-4-5-20251101-v1:0":   (15.0, 75.0),
    "us.anthropic.claude-opus-4-1-20250805-v1:0":   (15.0, 75.0),
    "us.anthropic.claude-sonnet-4-6":               ( 3.0, 15.0),
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": ( 3.0, 15.0),
    "us.anthropic.claude-haiku-4-5-20251001-v1:0":  ( 0.8,  4.0),
}

THROTTLE_MAX_DELAY = 600
NETWORK_MAX_DELAY  = 120

# ── Logging ────────────────────────────────────────────────────────────────────

def _log(level: str, msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level.upper():6}] {msg}", flush=True)


# ── Client ─────────────────────────────────────────────────────────────────────

class BedrockClient:
    def __init__(self, budget: BudgetGuard | None = None):
        self.budget = budget
        self._client = self._make_client()

    def _make_client(self):
        region  = os.environ.get("AWS_REGION", "us-west-2")
        profile = os.environ.get("AWS_PROFILE")
        session = boto3.Session(
            region_name=region,
            **( {"profile_name": profile} if profile else {}),
        )
        return session.client("bedrock-runtime")

    def invoke_opus(
        self,
        messages:      list[dict],
        tools:         list[dict],
        system_prompt: str,
    ) -> dict:
        return self._invoke(messages, OPUS_MODEL, OPUS_FALLBACKS,
                            tools, system_prompt, max_tokens=16384)

    def invoke_sonnet(
        self,
        messages:      list[dict],
        tools:         list[dict],
        system_prompt: str,
    ) -> dict:
        return self._invoke(messages, SONNET_MODEL, SONNET_FALLBACKS,
                            tools, system_prompt, max_tokens=16384, temperature=0.3)

    def _invoke(
        self,
        messages:      list[dict],
        primary:       str,
        fallbacks:     tuple[str, ...],
        tools:         list[dict],
        system_prompt: str,
        max_tokens:    int   = 16384,
        temperature:   float | None = None,
    ) -> dict:
        skip:           set[str] = set()
        throttle_delay: float    = 5.0
        network_delay:  float    = 5.0
        auth_retried:   bool     = False

        while True:
            available = [m for m in (primary, *fallbacks) if m not in skip]
            if not available:
                _log("warn", "All models skipped — clearing and waiting 5 min")
                skip.clear()
                time.sleep(300)
                throttle_delay = 5.0
                continue

            model = available[0]
            body: dict = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens":        max_tokens,
                "system":            system_prompt,
                "tools":             tools,
                "messages":          messages,
            }
            if temperature is not None:
                body["temperature"] = temperature

            try:
                resp = self._client.invoke_model(
                    modelId=model,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(body),
                )
                if model != primary:
                    _log("info", f"Using fallback model: {model}")
                throttle_delay = 5.0
                network_delay  = 5.0
                auth_retried   = False
                payload = json.loads(resp["body"].read())
                if self.budget is not None:
                    usage = payload.get("usage", {})
                    ip, op = _PRICE_PER_MTOK.get(model, (15.0, 75.0))
                    cost = (usage.get("input_tokens", 0) * ip
                          + usage.get("output_tokens", 0) * op) / 1_000_000
                    self.budget.record_spend(cost)
                    _log("cost", f"~${cost:.4f}  total ~${self.budget._spend:.3f}")
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

                elif code in ("ModelErrorException", "ModelStreamErrorException",
                              "ResourceNotFoundException"):
                    _log("warn", f"{model} error/not-found — trying next")
                    skip.add(model)

                elif code == "ValidationException":
                    if any(x in msg for x in ("too long", "maximum", "exceeds",
                                              "token limit", "input length")):
                        _log("warn", "Context too long — trimming messages")
                        messages = _trim_messages(messages)
                    else:
                        raise

                elif code in ("ExpiredTokenException", "InvalidClientTokenId",
                              "AuthFailure", "AccessDeniedException"):
                    if not auth_retried:
                        _log("warn", f"Auth error ({code}) — refreshing client")
                        self._client = self._make_client()
                        auth_retried = True
                    else:
                        _log("error", "Persistent auth failure — waiting 5 min")
                        time.sleep(300)
                        auth_retried = False

                else:
                    _log("error", f"Unhandled ClientError {code} — waiting {throttle_delay:.0f}s")
                    time.sleep(throttle_delay)
                    throttle_delay = min(throttle_delay * 2, THROTTLE_MAX_DELAY)

            except _bce.EndpointConnectionError as e:
                _log("warn", f"Network error — waiting {network_delay:.0f}s")
                time.sleep(network_delay)
                network_delay = min(network_delay * 2, NETWORK_MAX_DELAY)

            except Exception as e:
                _log("error", f"Unexpected: {type(e).__name__}: {e} — waiting 30s")
                time.sleep(30)


def _trim_messages(messages: list[dict]) -> list[dict]:
    """Halve all tool-result content to reduce context size."""
    trimmed = []
    for msg in messages:
        if msg["role"] == "user" and isinstance(msg["content"], list):
            new_blocks = []
            for block in msg["content"]:
                if block.get("type") == "tool_result":
                    content = block.get("content", "")
                    if len(content) > 1000:
                        quarter = len(content) // 4
                        block = dict(block, content=(
                            content[:quarter] + "\n...[trimmed]...\n" + content[-quarter:]
                        ))
                new_blocks.append(block)
            msg = dict(msg, content=new_blocks)
        trimmed.append(msg)
    return trimmed
