"""
autoai/config.py — role-to-model routing for the orchestrator/worker split.

Override any model via environment variables:
  AUTOAI_ORCHESTRATOR_MODEL=us.anthropic.claude-opus-4-5-20251101-v1:0
  AUTOAI_WORKER_MODEL=us.anthropic.claude-haiku-4-5-20251001-v1:0
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum


class Role(str, Enum):
    ORCHESTRATOR = "ORCHESTRATOR"
    WORKER       = "WORKER"


@dataclass(frozen=True)
class RoleConfig:
    model:            str
    max_tokens:       int
    temperature:      float | None   # None → omit from request (use model default)
    thinking_enabled: bool
    thinking_budget:  int            # tokens; only used when thinking_enabled=True
    fallbacks:        tuple[str, ...]

    def model_priority(self) -> list[str]:
        return [self.model, *self.fallbacks]


# Verified working on account 127696279288 / us-west-2  (see context/bedrock_claude.md)
_DEFAULTS: dict[Role, RoleConfig] = {
    Role.ORCHESTRATOR: RoleConfig(
        model            = "us.anthropic.claude-opus-4-6-v1",
        max_tokens       = 16384,
        temperature      = None,    # extended-thinking requires temp=1; wired in Phase 2
        thinking_enabled = False,   # enabled in Phase 2 once response parsing handles it
        thinking_budget  = 8000,
        fallbacks        = (
            "us.anthropic.claude-opus-4-5-20251101-v1:0",
            "us.anthropic.claude-opus-4-1-20250805-v1:0",
            "us.anthropic.claude-sonnet-4-6",
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        ),
    ),
    Role.WORKER: RoleConfig(
        model            = "us.anthropic.claude-sonnet-4-6",
        max_tokens       = 16384,
        temperature      = 0.3,
        thinking_enabled = False,
        thinking_budget  = 0,
        fallbacks        = (
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        ),
    ),
}


def get_role_config(role: Role) -> RoleConfig:
    cfg = _DEFAULTS[role]
    override = os.environ.get(f"AUTOAI_{role.value}_MODEL")
    if override:
        cfg = RoleConfig(
            model            = override,
            max_tokens       = cfg.max_tokens,
            temperature      = cfg.temperature,
            thinking_enabled = cfg.thinking_enabled,
            thinking_budget  = cfg.thinking_budget,
            fallbacks        = cfg.fallbacks,
        )
    return cfg
