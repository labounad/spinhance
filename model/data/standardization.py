"""
model.data.standardization
===========================
Degeneracy vocabulary, z-score standardizer, and class-balance weights
(ported from legacy targets.py). Fit on TRAIN only.

The model predicts and the loss compares in STANDARDIZED space (shifts z-scored
over all groups, coupling magnitudes over present entries only). Evaluation
denormalizes via ``inverse_shifts`` / ``inverse_j`` to report ppm / Hz. Keeping
standardization in the data layer leaves losses pure tensor math.
"""
from __future__ import annotations

import numpy as np

from model.data.transforms import encode_target
from model.schemas.constants import DEFAULT_DEG_VOCAB

__all__ = ["DegeneracyVocab", "Standardizer", "class_balance", "DEFAULT_DEG_VOCAB"]


class DegeneracyVocab:
    def __init__(self, vocab=DEFAULT_DEG_VOCAB):
        self.vocab = tuple(int(v) for v in vocab)
        self._to = {v: i for i, v in enumerate(self.vocab)}

    def __len__(self):
        return len(self.vocab)

    def to_index(self, degeneracy):
        out = []
        for d in np.asarray(degeneracy).ravel():
            d = int(round(float(d)))
            if d not in self._to:
                raise KeyError(f"degeneracy {d} not in vocab {self.vocab}")
            out.append(self._to[d])
        return np.array(out, dtype=np.int64)

    def from_index(self, idx):
        idx = np.asarray(idx)
        shape = idx.shape
        flat = np.array(self.vocab, dtype=np.int64)[idx.ravel()]
        return flat.reshape(shape)


class Standardizer:
    """Z-score shifts (all groups) and coupling magnitudes (present entries only)."""

    def __init__(self):
        self.shift_mean = self.shift_std = None
        self.j_mean = self.j_std = None

    def fit(self, train_records, vocab: DegeneracyVocab):
        shifts_all, j_present = [], []
        for r in train_records:
            t = encode_target(r["shifts"], r["couplings"], r["degeneracy"], vocab)
            shifts_all.append(t["shifts"])
            mask = t["j_presence"] > 0
            j_present.append(t["j_mag"][mask])
        shifts_all = np.concatenate(shifts_all)
        j_present = np.concatenate(j_present) if any(len(x) for x in j_present) \
            else np.array([0.0])
        self.shift_mean, self.shift_std = float(shifts_all.mean()), float(shifts_all.std() + 1e-8)
        self.j_mean, self.j_std = float(j_present.mean()), float(j_present.std() + 1e-8)
        return self

    def transform(self, t):
        out = dict(t)
        out["shifts"] = (t["shifts"] - self.shift_mean) / self.shift_std
        jm = (t["j_mag"] - self.j_mean) / self.j_std
        out["j_mag"] = (jm * t["j_presence"]).astype(np.float32)   # zero where absent
        return out

    def inverse_shifts(self, x):
        return np.asarray(x) * self.shift_std + self.shift_mean

    def inverse_j(self, x):
        return np.asarray(x) * self.j_std + self.j_mean

    def state_dict(self) -> dict:
        return {"shift_mean": self.shift_mean, "shift_std": self.shift_std,
                "j_mean": self.j_mean, "j_std": self.j_std}

    def load_state_dict(self, d: dict) -> "Standardizer":
        self.shift_mean, self.shift_std = d["shift_mean"], d["shift_std"]
        self.j_mean, self.j_std = d["j_mean"], d["j_std"]
        return self


def class_balance(train_records, vocab: DegeneracyVocab, power=0.5, cap=8.0):
    """Weights to counter class imbalance (computed on train):
      deg_weights        (C,)  tempered inverse-frequency CE weights
      presence_pos_weight float #absent / #present couplings (BCE pos_weight)
    """
    C = len(vocab)
    deg_counts = np.zeros(C)
    n_present = 0
    n_pairs_total = 0
    for r in train_records:
        t = encode_target(r["shifts"], r["couplings"], r["degeneracy"], vocab)
        for c in t["deg_class"]:
            deg_counts[int(c)] += 1
        n_present += int(t["j_presence"].sum())
        n_pairs_total += t["j_presence"].size

    present = deg_counts > 0
    deg_weights = np.zeros(C)
    raw = (1.0 / deg_counts[present]) ** power
    raw = raw / raw.mean()
    raw = np.clip(raw, 1.0 / cap, cap)
    raw = raw / raw.mean()
    deg_weights[present] = raw
    n_absent = n_pairs_total - n_present
    presence_pos_weight = float(n_absent / max(n_present, 1))
    return dict(deg_weights=deg_weights.astype(np.float32),
                presence_pos_weight=presence_pos_weight,
                deg_counts=deg_counts.astype(int))
