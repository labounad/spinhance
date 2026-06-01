"""autoaiv2.idea_spec — IdeaSpec: structured handoff from Opus to Sonnet."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IdeaSpec:
    objective:               str
    architecture_changes:    str
    loss_changes:            str
    preprocessing_changes:   str
    training_overrides:      str   # raw CLI args, e.g. "--epochs 60 --batch 64 --lr 3e-4"
    feasibility_notes:       str
    success_criteria:        str

    @classmethod
    def from_dict(cls, d: dict) -> "IdeaSpec":
        return cls(
            objective             = d["objective"],
            architecture_changes  = d.get("architecture_changes", "none"),
            loss_changes          = d.get("loss_changes", "none"),
            preprocessing_changes = d.get("preprocessing_changes", "none"),
            training_overrides    = d.get("training_overrides", ""),
            feasibility_notes     = d.get("feasibility_notes", ""),
            success_criteria      = d.get("success_criteria", ""),
        )

    def to_dict(self) -> dict:
        return {
            "objective":             self.objective,
            "architecture_changes":  self.architecture_changes,
            "loss_changes":          self.loss_changes,
            "preprocessing_changes": self.preprocessing_changes,
            "training_overrides":    self.training_overrides,
            "feasibility_notes":     self.feasibility_notes,
            "success_criteria":      self.success_criteria,
        }

    def as_prompt(self) -> str:
        lines = [
            "## IdeaSpec — implement this exactly\n",
            f"**Objective:** {self.objective}\n",
        ]
        if self.architecture_changes.lower() not in ("none", ""):
            lines.append(f"**Architecture changes:** {self.architecture_changes}\n")
        if self.loss_changes.lower() not in ("none", ""):
            lines.append(f"**Loss changes:** {self.loss_changes}\n")
        if self.preprocessing_changes.lower() not in ("none", ""):
            lines.append(f"**Preprocessing changes:** {self.preprocessing_changes}\n")
        if self.training_overrides:
            lines.append(f"**Training overrides:** `{self.training_overrides}`\n")
        if self.feasibility_notes:
            lines.append(f"**Feasibility notes:** {self.feasibility_notes}\n")
        if self.success_criteria:
            lines.append(f"**Success criteria:** {self.success_criteria}\n")
        return "\n".join(lines)
