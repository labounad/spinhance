"""
model.data.stacked_spectra
==========================
Spectra source for the PubChem 3M+ regime, where spectra are stored as a sequence
of **stacked shards** ``part_<k>.npy`` (each shape ``(n_k, P)``) rather than one
``.npy`` per molecule. Shard k holds the spectra for the contiguous block of
records ``[offset_k : offset_k + n_k]`` in the *same record order* as the
``spin_systems_*.json[.gz]`` file — so global molecule index i maps to
``(part, row)`` by a cumulative-offset lookup.

Parts are memory-mapped lazily (only the headers are read up front to size the
index), so the 196 GB of 90 MHz spectra never load into RAM at once — each
``__getitem__`` copies a single (P,) row out of the relevant mmap.
"""
from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path

import numpy as np

_PART_RE = re.compile(r"part_(\d+)\.npy$")


class StackedSpectra:
    def __init__(self, parts_dir, pattern="part_*.npy", max_open=32):
        self.parts_dir = Path(parts_dir)
        files = sorted(self.parts_dir.glob(pattern),
                       key=lambda p: int(_PART_RE.search(p.name).group(1)))
        if not files:
            raise FileNotFoundError(f"no {pattern} under {self.parts_dir}")
        self.files = files
        # read only the .npy headers to size each shard (cheap — no data read)
        counts = [int(np.load(f, mmap_mode="r").shape[0]) for f in files]
        self.offsets = np.cumsum([0] + counts)            # (n_parts+1,)
        self.total = int(self.offsets[-1])
        # Bounded LRU of open mmaps. Shuffled training touches all ~3200 shards;
        # caching one mmap per shard leaks file descriptors (OSError [Errno 24] Too
        # many open files) and resident pages. Keep only the last `max_open` open,
        # closing the rest — caps fds at max_open per DataLoader worker.
        self.max_open = max_open
        self._mmaps: "OrderedDict[int, np.ndarray]" = OrderedDict()

    def __len__(self):
        return self.total

    def _part(self, k: int) -> np.ndarray:
        m = self._mmaps.get(k)
        if m is not None:
            self._mmaps.move_to_end(k)
            return m
        m = np.load(self.files[k], mmap_mode="r")
        self._mmaps[k] = m
        if len(self._mmaps) > self.max_open:
            _, old = self._mmaps.popitem(last=False)      # evict least-recently-used
            try:
                old._mmap.close()                         # release the file descriptor
            except Exception:
                pass
        return m

    def __getitem__(self, i: int) -> np.ndarray:
        if not 0 <= i < self.total:
            raise IndexError(i)
        k = int(np.searchsorted(self.offsets, i, side="right") - 1)
        row = i - int(self.offsets[k])
        return np.array(self._part(k)[row], dtype=np.float32)     # writable copy out of mmap
