"""
model.data.surrogate_dataset
============================
(matrix -> spectrum) pairs for training the differentiable surrogate renderer
(Branch 5). Input is the PHYSICAL spin matrix (ppm / Hz / protons — no
standardization; the surrogate is a physics model). Target is the pyspin dense
spectrum at each field.

Each item carries both fields' spectra; the trainer picks one field per batch
(the shared-kernel broadening needs a single linewidth/field per batch).
Spectra are read from rec[f"spec{field}"] (in-memory) or rec[f"spec{field}_path"]
(.npy) — so tests can pass arrays directly and production reads precomputed files.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from model.schemas import constants as C


def _load_spectrum(rec, field: int):
    key = f"spec{field}"
    if key in rec and rec[key] is not None:
        return np.asarray(rec[key], dtype=np.float32)
    path = rec.get(f"{key}_path")
    if path is None:
        raise KeyError(f"record {rec.get('mol_id')} missing '{key}' and '{key}_path'")
    return np.load(path, mmap_mode="r").astype(np.float32)


class SurrogateSpectrumDataset(Dataset):
    def __init__(self, records, fields=(90, 600)):
        self.records = list(records)
        self.fields = tuple(int(f) for f in fields)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        item = {
            "shifts": torch.as_tensor(np.asarray(r["shifts"], dtype=np.float32)),       # (G,)
            "couplings": torch.as_tensor(np.asarray(r["couplings"], dtype=np.float32)),  # (G,G)
            "degeneracy": torch.as_tensor(np.asarray(r["degeneracy"], dtype=np.float32)),  # (G,)
            "mol_id": r.get("mol_id", f"mol_{i:06d}"),
        }
        for f in self.fields:
            item[f"spec{f}"] = torch.as_tensor(_load_spectrum(r, f))                    # (P,)
        return item


def make_surrogate_collate(fields=(90, 600)):
    fields = tuple(int(f) for f in fields)

    def collate(batch):
        out = {
            "shifts": torch.stack([b["shifts"] for b in batch]),
            "couplings": torch.stack([b["couplings"] for b in batch]),
            "degeneracy": torch.stack([b["degeneracy"] for b in batch]),
            "molecule_ids": [b["mol_id"] for b in batch],
        }
        for f in fields:
            out[f"spec{f}"] = torch.stack([b[f"spec{f}"] for b in batch])
        return out

    return collate
