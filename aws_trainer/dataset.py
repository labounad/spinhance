"""
aws_trainer.dataset
====================
Improved dataset for the 100k-molecule scale:

  SpectraCache           — pre-loads all spectra for one field into a contiguous
                           fp16 numpy array in RAM (~3.3 GB / field for 100k mols).
                           Eliminates per-epoch file I/O that dominates wall time.

  CachedSpectrumDataset  — drop-in replacement for model.SpectrumMatrixDataset.
                           Same __getitem__ output schema; compatible with
                           model.dataset.collate_fn.

  DistributedBucketSampler — DDP-aware version of model.BucketByDegeneracySampler.
                           Splits indices by rank first, then buckets within each
                           rank's shard.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from model.targets import (DegeneracyVocab, Standardizer, augment_spectrum,
                            bucket_key, encode_target)


# ── Spectra cache ─────────────────────────────────────────────────────────────

class SpectraCache:
    """Load all spectra for a single field into RAM as a contiguous fp16 array.

    Indexed by the position of the record in the records list.  Returns fp32
    slices at access time (zero-copy cast).

    Memory:  100k × 16384 × fp16 ≈ 3.3 GB per field.
    """

    def __init__(self, records: list[dict], field: int, verbose: bool = True):
        import io
        import tarfile
        from pathlib import Path

        key = f"spec{int(field)}_path"
        N = len(records)
        first_path = Path(records[0][key])
        tar_path = first_path.parent / "mol_all.tar.gz"

        if tar_path.exists():
            # Single sequential pass through the gzip tar — efficient for streaming.
            name_to_idx = {Path(r[key]).name: i for i, r in enumerate(records)}
            self._data: np.ndarray | None = None
            loaded = 0
            if verbose:
                print(f"[SpectraCache] field={field}MHz  loading from {tar_path.name} ...")
            with tarfile.open(tar_path, "r:gz") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    stem = Path(member.name).name
                    if stem not in name_to_idx:
                        continue
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    arr = np.load(io.BytesIO(f.read())).astype(np.float16)
                    if self._data is None:
                        self._data = np.empty((N, len(arr)), dtype=np.float16)
                    self._data[name_to_idx[stem]] = arr
                    loaded += 1
            if loaded != N:
                raise RuntimeError(
                    f"SpectraCache: loaded {loaded}/{N} spectra from {tar_path}. "
                    "Check that the tar contains all expected mol_*.npy files.")
        else:
            first = np.load(records[0][key])
            P = len(first)
            self._data = np.empty((N, P), dtype=np.float16)
            self._data[0] = first.astype(np.float16)
            for i in range(1, N):
                self._data[i] = np.load(records[i][key]).astype(np.float16)

        if verbose:
            mb = self._data.nbytes / 1e6
            P = self._data.shape[1]
            print(f"[SpectraCache] field={field}MHz  {N}×{P} fp16 = {mb:.0f} MB loaded")

    def __getitem__(self, i: int) -> np.ndarray:
        return self._data[i].astype(np.float32)

    def __len__(self) -> int:
        return len(self._data)

    @staticmethod
    def slice(base: "SpectraCache", offset: int) -> "_CacheSlice":
        return _CacheSlice(base, offset)


class _CacheSlice:
    """Offset view into a SpectraCache. Module-level so it is picklable."""
    def __init__(self, base: SpectraCache, offset: int):
        self._base = base
        self._off = offset

    def __getitem__(self, i: int) -> np.ndarray:
        return self._base[self._off + i]


# ── Dataset ───────────────────────────────────────────────────────────────────

class CachedSpectrumDataset(Dataset):
    """Same output schema as model.SpectrumMatrixDataset; uses SpectraCache for I/O."""

    def __init__(self, records: list[dict], vocab: DegeneracyVocab,
                 standardizer: Standardizer, cache: SpectraCache | None = None,
                 spectrum_field: str = "spec90", augment: bool = False,
                 ppm_from: float = 0.0, ppm_to: float = 12.0,
                 aug_kwargs: dict | None = None, seed: int = 0):
        self.records = list(records)
        self.vocab = vocab
        self.std = standardizer
        self.cache = cache
        self.field = int(spectrum_field.replace("spec", ""))
        self.augment = augment
        self.ppm_from, self.ppm_to = ppm_from, ppm_to
        self.aug_kwargs = aug_kwargs or {}
        self.seed = seed
        self.bucket_keys = [
            bucket_key(r["shifts"], r["couplings"], r["degeneracy"])
            for r in self.records
        ]

    def _load_spectrum(self, i: int) -> np.ndarray:
        if self.cache is not None:
            return self.cache[i]
        r = self.records[i]
        key = f"spec{self.field}"
        if key in r and r[key] is not None:
            return np.asarray(r[key], dtype=np.float32)
        return np.load(r[key + "_path"]).astype(np.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        r = self.records[i]
        clean = self._load_spectrum(i)
        inp = clean
        if self.augment:
            rng = np.random.default_rng((self.seed, i, torch.initial_seed() % (2 ** 31)))
            inp = augment_spectrum(clean, self.ppm_from, self.ppm_to, rng=rng,
                                   **self.aug_kwargs)
        t = self.std.transform(
            encode_target(r["shifts"], r["couplings"], r["degeneracy"], self.vocab))
        deg_ordered = self.vocab.from_index(t["deg_class"])
        return {
            "spectrum":      torch.from_numpy(np.ascontiguousarray(inp)),
            "spectrum_ref":  torch.from_numpy(np.ascontiguousarray(clean)),
            "shifts":        torch.from_numpy(t["shifts"]),
            "j_mag":         torch.from_numpy(t["j_mag"]),
            "j_presence":    torch.from_numpy(t["j_presence"]),
            "deg_class":     torch.from_numpy(t["deg_class"]),
            "degeneracy":    torch.from_numpy(deg_ordered.astype(np.int64)),
            "bucket_key":    self.bucket_keys[i],
        }


# ── Samplers ──────────────────────────────────────────────────────────────────

class DistributedBucketSampler(Sampler):
    """Bucket sampler compatible with DDP.

    Partitions indices by rank (round-robin), then buckets within each rank's
    shard.  For single-GPU use (rank=0, world_size=1) it is identical to
    model.BucketByDegeneracySampler.
    """

    def __init__(self, bucket_keys: list, batch_size: int,
                 rank: int = 0, world_size: int = 1,
                 shuffle: bool = True, drop_last: bool = False, seed: int = 0):
        self.bucket_keys = list(bucket_keys)
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        # Each rank owns every world_size-th index
        self._local = list(range(rank, len(bucket_keys), world_size))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng((self.seed, self.epoch, self.rank))
        buckets: dict = {}
        for i in self._local:
            buckets.setdefault(self.bucket_keys[i], []).append(i)
        batches = []
        for idxs in buckets.values():
            if self.shuffle:
                rng.shuffle(idxs)
            for s in range(0, len(idxs), self.batch_size):
                b = idxs[s: s + self.batch_size]
                if self.drop_last and len(b) < self.batch_size:
                    continue
                if b:
                    batches.append(b)
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        total = 0
        buckets: dict = {}
        for i in self._local:
            buckets.setdefault(self.bucket_keys[i], []).append(i)
        for idxs in buckets.values():
            if self.drop_last:
                total += len(idxs) // self.batch_size
            else:
                total += (len(idxs) + self.batch_size - 1) // self.batch_size
        return total
