"""autoaiv2.budget — BudgetGuard: stop the loop on cycles / wall time / spend."""
from __future__ import annotations

import os
import time


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
        self._start  = time.monotonic()
        self._cycles = 0
        self._spend  = 0.0

    @classmethod
    def from_env(cls) -> "BudgetGuard":
        return cls(
            max_cycles    = int(os.environ["AUTOAI_MAX_CYCLES"])       if os.environ.get("AUTOAI_MAX_CYCLES")    else None,
            max_wall_secs = float(os.environ["AUTOAI_MAX_HOURS"])*3600 if os.environ.get("AUTOAI_MAX_HOURS")     else None,
            max_spend_usd = float(os.environ["AUTOAI_MAX_SPEND_USD"])  if os.environ.get("AUTOAI_MAX_SPEND_USD") else None,
        )

    def check(self) -> None:
        if self.max_cycles is not None and self._cycles >= self.max_cycles:
            raise BudgetExceeded(f"max_cycles={self.max_cycles} reached")
        if self.max_wall_secs is not None:
            elapsed = time.monotonic() - self._start
            if elapsed >= self.max_wall_secs:
                raise BudgetExceeded(f"wall time limit reached ({elapsed/3600:.2f}h)")
        if self.max_spend_usd is not None and self._spend >= self.max_spend_usd:
            raise BudgetExceeded(
                f"spend limit ${self.max_spend_usd:.2f} reached (${self._spend:.2f} spent)"
            )

    def record_cycle(self) -> None:
        self._cycles += 1

    def record_spend(self, usd: float) -> None:
        self._spend += usd

    def status(self) -> str:
        elapsed = (time.monotonic() - self._start) / 3600
        return (
            f"cycles={self._cycles}/{self.max_cycles or '∞'}  "
            f"wall={elapsed:.2f}h/{(self.max_wall_secs or 0)/3600:.1f}h  "
            f"spend=${self._spend:.2f}/${self.max_spend_usd or '∞'}"
        )
