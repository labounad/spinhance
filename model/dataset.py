"""
model.dataset
================
PyTorch Dataset + samplers for Task 4. Thin torch layer over the verified,
torch-free transforms in ``model.targets``.

  * SpectrumMatrixDataset   - one 90 MHz spectrum -> standardized target dict
  * BucketByDegeneracySampler - yields batches whose members share a degeneracy
                                pattern, so the Stage-2 renderer can build the
                                operator ``struct`` once and reuse it (Decision 7,
                                "bucket + stochastic subset").
  * collate_fn              - stacks tensors; carries the shared degeneracy vector
                              and bucket key for the renderer.

Verify in your env:
    python3 -m model.dataset      # runs a tiny smoke test on synthetic data
Cross-check encoding against the torch-free oracle: targets.encode_target(...).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from model.targets import (DegeneracyVocab, Standardizer, encode_target,
                              augment_spectrum, bucket_key)


def _load_spectrum(rec, spectrum_field):
    """Spectrum from an in-memory array (rec[spectrum_field]) or a .npy path
    (rec[spectrum_field + '_path'])."""
    if spectrum_field in rec and rec[spectrum_field] is not None:
        return np.asarray(rec[spectrum_field], dtype=np.float32)
    path = rec.get(spectrum_field + "_path")
    if path is None:
        raise KeyError(f"record missing '{spectrum_field}' and "
                       f"'{spectrum_field}_path'")
    return np.load(path).astype(np.float32)


class SpectrumMatrixDataset(Dataset):
    def __init__(self, records, vocab: DegeneracyVocab, standardizer: Standardizer,
                 spectrum_field="spec90", augment=False, ppm_from=0.0, ppm_to=12.0,
                 aug_kwargs=None, seed=0):
        self.records = list(records)
        self.vocab = vocab
        self.std = standardizer
        self.spectrum_field = spectrum_field
        self.augment = augment
        self.ppm_from, self.ppm_to = ppm_from, ppm_to
        self.aug_kwargs = aug_kwargs or {}
        self.seed = seed
        # precompute bucket keys (cheap, used by the sampler)
        self.bucket_keys = [bucket_key(r["shifts"], r["couplings"], r["degeneracy"])
                            for r in self.records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        spec = _load_spectrum(r, self.spectrum_field)
        if self.augment:
            # per-sample rng so DataLoader workers are reproducible
            rng = np.random.default_rng((self.seed, i, torch.initial_seed() % (2**31)))
            spec = augment_spectrum(spec, self.ppm_from, self.ppm_to, rng=rng,
                                    **self.aug_kwargs)
        t = self.std.transform(
            encode_target(r["shifts"], r["couplings"], r["degeneracy"], self.vocab))
        deg_ordered = self.vocab.from_index(t["deg_class"])
        return {
            "spectrum": torch.from_numpy(np.ascontiguousarray(spec)),       # (P,)
            "shifts": torch.from_numpy(t["shifts"]),                        # (G,)
            "j_mag": torch.from_numpy(t["j_mag"]),                          # (28,)
            "j_presence": torch.from_numpy(t["j_presence"]),                # (28,)
            "deg_class": torch.from_numpy(t["deg_class"]),                  # (G,) long
            "degeneracy": torch.from_numpy(deg_ordered.astype(np.int64)),   # (G,) raw
            "bucket_key": self.bucket_keys[i],
        }


def collate_fn(batch):
    keys = {b["bucket_key"] for b in batch}
    out = {
        "spectrum": torch.stack([b["spectrum"] for b in batch]),
        "shifts": torch.stack([b["shifts"] for b in batch]),
        "j_mag": torch.stack([b["j_mag"] for b in batch]),
        "j_presence": torch.stack([b["j_presence"] for b in batch]),
        "deg_class": torch.stack([b["deg_class"] for b in batch]),
        "degeneracy": torch.stack([b["degeneracy"] for b in batch]),
        "bucket_keys": [b["bucket_key"] for b in batch],
        # shared degeneracy vector iff the batch is single-bucket (Stage-2 render)
        "shared_degeneracy": (batch[0]["degeneracy"] if len(keys) == 1 else None),
    }
    return out


class BucketByDegeneracySampler(Sampler):
    """Yield batches of indices that share a degeneracy pattern.

    Enables renderer ``struct`` reuse for the Stage-2 spectral loss. For Stage 1
    (matrix loss only) a plain shuffled sampler is fine; use this when the
    spectral term is active.
    """

    def __init__(self, bucket_keys, batch_size, shuffle=True, drop_last=False,
                 seed=0):
        self.bucket_keys = list(bucket_keys)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.buckets = {}
        for idx, k in enumerate(self.bucket_keys):
            self.buckets.setdefault(k, []).append(idx)
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng((self.seed, self.epoch))
        self.epoch += 1
        batches = []
        for idxs in self.buckets.values():
            idxs = list(idxs)
            if self.shuffle:
                rng.shuffle(idxs)
            for s in range(0, len(idxs), self.batch_size):
                b = idxs[s:s + self.batch_size]
                if self.drop_last and len(b) < self.batch_size:
                    continue
                batches.append(b)
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        total = 0
        for idxs in self.buckets.values():
            if self.drop_last:
                total += len(idxs) // self.batch_size
            else:
                total += (len(idxs) + self.batch_size - 1) // self.batch_size
        return total


# -----------------------------------------------------------------------------
# Smoke test (needs torch; run in your env)
# -----------------------------------------------------------------------------

def _smoke():
    from torch.utils.data import DataLoader
    rng = np.random.default_rng(0)
    G, P = 8, 1024

    def mol(i):
        c = np.zeros((G, G))
        for a in range(G):
            for b in range(a + 1, G):
                if rng.random() < 0.4:
                    c[a, b] = c[b, a] = float(rng.uniform(1, 10))
        return dict(mol_id=f"m{i}", shifts=rng.uniform(0.5, 9, G), couplings=c,
                    degeneracy=rng.choice([1, 2, 3, 6], size=G).astype(int),
                    spec90=rng.random(P).astype(np.float32))

    recs = [mol(i) for i in range(64)]
    vocab = DegeneracyVocab()
    std = Standardizer().fit(recs, vocab)

    ds = SpectrumMatrixDataset(recs, vocab, std, spectrum_field="spec90",
                               augment=True, ppm_to=12.0)
    # Stage 1: plain loader
    dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(dl))
    print("plain batch shapes:",
          {k: tuple(v.shape) for k, v in batch.items()
           if isinstance(v, torch.Tensor)})

    # Stage 2: bucketed loader (each batch single-bucket)
    samp = BucketByDegeneracySampler(ds.bucket_keys, batch_size=8)
    dlb = DataLoader(ds, batch_sampler=samp, collate_fn=collate_fn)
    b2 = next(iter(dlb))
    assert len(set(b2["bucket_keys"])) == 1, "bucketed batch not single-bucket"
    assert b2["shared_degeneracy"] is not None
    print("bucketed batch single-bucket OK; shared_degeneracy:",
          b2["shared_degeneracy"].tolist())
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    _smoke()
