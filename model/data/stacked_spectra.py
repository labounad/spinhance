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

**Performance note (GPFS):** random single-row mmap faults across 3200 shards on a
network filesystem are *slow* (~20 s/step, GPU-starved). For runs whose needed
rows fit in RAM, call :meth:`preload` — it reads each shard **sequentially**
(fast on GPFS) once and serves rows from a RAM array thereafter. The array is
built in the main process before the DataLoader forks, so workers share it
copy-on-write (read-only) without duplicating it.
"""
from __future__ import annotations

import os
import re
from collections import OrderedDict, defaultdict
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
        self._ram = None                                  # (n_needed, P) preloaded array
        self._ram_index: dict[int, int] = {}              # global row -> local index in _ram

    def __len__(self):
        return self.total

    def preload(self, rows, dense_frac: float = 0.25, cache: str | None = None) -> None:
        """Load the given global rows into a RAM array, so ``__getitem__`` serves
        them from memory instead of random mmap faults. Call once in the main
        process before DataLoader fork.

        Per shard the read strategy is chosen by density: when the needed rows are
        a large fraction (≥ ``dense_frac``) of the shard, the whole shard is read
        sequentially (fast on GPFS, the right call for dense training subsets);
        when only a few rows are needed (the SPARSE case — e.g. a scattered held-out
        test subset where each shard contributes ~a dozen of ~2000 rows) the shard
        is mmap-ed and only the needed rows are faulted in. The sparse path reads
        ~``len(rows)`` rows total instead of the full corpus — e.g. a 20k-row
        held-out subset reads ~1.3 GB rather than ~196 GB of whole shards.

        Memory ≈ len(set(rows)) * P * 4 bytes (e.g. 500k * 16384 * 4 ≈ 33 GB)."""
        needed = sorted(set(int(r) for r in rows))
        if not needed:
            return
        if cache and Path(cache).exists():       # load the pre-materialized subset (one f32 read)
            z = np.load(cache)
            self._ram = z["spectra"]
            self._ram_index = {int(r): i for i, r in enumerate(z["rows"])}
            miss = [r for r in needed if r not in self._ram_index]
            if miss:
                raise ValueError(f"preload cache {cache} is missing {len(miss)} requested rows "
                                 f"(stale — delete it to rebuild)")
            print(f"[stacked] preload from cache {cache} "
                  f"({self._ram.shape[0]} rows, {self._ram.nbytes / 1e9:.1f} GB)", flush=True)
            return
        P = int(np.load(self.files[0], mmap_mode="r").shape[1])
        self._ram_index = {r: i for i, r in enumerate(needed)}
        self._ram = np.empty((len(needed), P), dtype=np.float32)
        by_shard: "defaultdict[int, list]" = defaultdict(list)
        for r in needed:
            k = int(np.searchsorted(self.offsets, r, side="right") - 1)
            by_shard[k].append(r)
        gb = self._ram.nbytes / 1e9
        print(f"[stacked] preloading {len(needed)} rows from {len(by_shard)} shards "
              f"into RAM (~{gb:.1f} GB)", flush=True)
        n_full = n_sparse = 0
        for n, (k, rs) in enumerate(sorted(by_shard.items())):
            base = int(self.offsets[k])
            shard_sz = int(self.offsets[k + 1] - base)
            if len(rs) >= dense_frac * shard_sz:          # dense → one sequential read
                part = np.load(self.files[k])
                for r in rs:
                    self._ram[self._ram_index[r]] = part[r - base]
                del part
                n_full += 1
            else:                                          # sparse → fault only needed rows
                mm = np.load(self.files[k], mmap_mode="r")
                for r in rs:
                    self._ram[self._ram_index[r]] = mm[r - base]
                try:
                    mm._mmap.close()
                except Exception:
                    pass
                n_sparse += 1
            if (n + 1) % 500 == 0:
                print(f"[stacked]   {n + 1}/{len(by_shard)} shards", flush=True)
        print(f"[stacked] preload done ({n_full} full-read, {n_sparse} sparse-mmap shards)",
              flush=True)
        if cache:                                 # persist for next run (atomic write)
            tmp = str(cache) + ".tmp.npz"
            np.savez(tmp, rows=np.asarray(needed, dtype=np.int64), spectra=self._ram)
            os.replace(tmp, cache)
            print(f"[stacked] wrote preload cache -> {cache} ({self._ram.nbytes / 1e9:.1f} GB)", flush=True)

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
        if self._ram is not None:                          # preloaded -> serve from RAM
            li = self._ram_index.get(i)
            if li is not None:
                return np.array(self._ram[li], dtype=np.float32)
        k = int(np.searchsorted(self.offsets, i, side="right") - 1)
        row = i - int(self.offsets[k])
        return np.array(self._part(k)[row], dtype=np.float32)     # writable copy out of mmap
