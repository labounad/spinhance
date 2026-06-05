"""
model.data.dataset
==================
PyTorch Dataset: one 90 MHz spectrum -> standardized, matrix-form target sample.
Thin torch layer over the torch-free transforms. Each __getitem__ returns a dict
of tensors; ``collate.collate_spin_batch`` stacks them into a ``SpinBatch``.

Targets are emitted in the matrix form the SpinBatch contract expects:
  couplings, coupling_mask : (G, G) symmetric (standardized magnitudes / {0,1})
built from the canonical-ordered upper-triangle encoding.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import torch
from torch.utils.data import Dataset

from model.data.splits import canonical_order
from model.data.standardization import DegeneracyVocab, Standardizer
from model.data.transforms import augment_spectrum, bucket_key, encode_target

#: max number of symmetry-orbit relabelings kept per sample for the symmetry-aware J-loss
SYM_PERMS_MAX = 8


def _orbit_perms(orbit, kmax=SYM_PERMS_MAX):
    """(G,) per-group symmetry-orbit ids -> (kmax, G) int64 array of within-orbit node
    relabelings (row 0 = identity, padded with identity). These are the canonical-sort's
    arbitrary tie-breaks among chemically-equivalent groups; the symmetry-aware J-loss
    minimizes the coupling term over them so a valid relabeling isn't penalized."""
    orbit = np.asarray(orbit)
    G = len(orbit)
    ident = np.arange(G, dtype=np.int64)
    classes = {}
    for i, o in enumerate(orbit):
        classes.setdefault(int(o), []).append(i)
    multi = [idx for idx in classes.values() if len(idx) > 1]
    perms = [ident]
    if multi:
        total = 1
        for idx in multi:
            total *= math.factorial(len(idx))
        if total <= 64:                                  # cheap to enumerate; else identity-only
            seen = {tuple(ident)}
            per_class = [list(itertools.permutations(idx)) for idx in multi]
            for combo in itertools.product(*per_class):
                p = ident.copy()
                for idx, order in zip(multi, combo):
                    p[np.asarray(idx)] = np.asarray(order)
                key = tuple(p)
                if key not in seen:
                    seen.add(key); perms.append(p)
                if len(perms) >= kmax:
                    break
    perms = perms[:kmax]
    while len(perms) < kmax:                             # pad with identity (harmless in the min)
        perms.append(ident)
    return np.stack(perms)                               # (kmax, G)

# Worker-level RNG seed — set once per worker by worker_init_fn
_WORKER_SEED: int = 0


def worker_init_fn(worker_id: int) -> None:
    global _WORKER_SEED
    _WORKER_SEED = torch.initial_seed() % (2 ** 31)


def _load_spectrum(rec, spectrum_field):
    """Spectrum from an in-memory array (rec[field]) or a .npy path (mmap)."""
    if spectrum_field in rec and rec[spectrum_field] is not None:
        return np.asarray(rec[spectrum_field], dtype=np.float32)
    path = rec.get(spectrum_field + "_path")
    if path is None:
        raise KeyError(f"record missing '{spectrum_field}' and '{spectrum_field}_path'")
    return np.load(path, mmap_mode="r").astype(np.float32)


def _pairs_to_matrix(vec, presence, G):
    """(E,) upper-tri magnitudes + presence -> (G,G) symmetric matrix and mask."""
    iu = np.triu_indices(G, 1)
    M = np.zeros((G, G), dtype=np.float32)
    mask = np.zeros((G, G), dtype=np.float32)
    M[iu] = vec; M[(iu[1], iu[0])] = vec
    mask[iu] = presence; mask[(iu[1], iu[0])] = presence
    return M, mask


class SpectrumMatrixDataset(Dataset):
    def __init__(self, records, vocab: DegeneracyVocab, standardizer: Standardizer,
                 spectrum_field="spec90", augment=False, ppm_from=0.0, ppm_to=12.0,
                 aug_kwargs=None, seed=0, region_tokens=False, region_max=48,
                 region_kwargs=None, spectra_source=None):
        self.records = list(records)
        self.vocab = vocab
        self.std = standardizer
        self.spectrum_field = spectrum_field
        # optional stacked-shard source (PubChem 3M+): record["row"] -> spectrum
        self.spectra_source = spectra_source
        self.augment = augment
        self.ppm_from, self.ppm_to = ppm_from, ppm_to
        self.aug_kwargs = aug_kwargs or {}
        self.seed = seed
        self.region_tokens = region_tokens
        self.region_max = region_max
        self.region_kwargs = region_kwargs or {}
        # Precompute canonical orders once (avoids an O(G log G) lexsort per epoch)
        self._orders = [canonical_order(r["shifts"], r["couplings"], r["degeneracy"])
                        for r in self.records]
        self.bucket_keys = [bucket_key(r["shifts"], r["couplings"], r["degeneracy"],
                                       order=self._orders[i])
                            for i, r in enumerate(self.records)]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        if self.spectra_source is not None:
            clean = np.asarray(self.spectra_source[r["row"]], dtype=np.float32)
        else:
            clean = _load_spectrum(r, self.spectrum_field)
        inp = clean
        if self.augment:
            rng = np.random.default_rng((_WORKER_SEED, self.seed, i))
            n_protons = int(sum(int(x) for x in r["degeneracy"]))
            inp = augment_spectrum(clean, self.ppm_from, self.ppm_to, rng=rng,
                                   n_protons=n_protons, **self.aug_kwargs)

        t = self.std.transform(
            encode_target(r["shifts"], r["couplings"], r["degeneracy"], self.vocab,
                          order=self._orders[i]))
        G = len(t["shifts"])
        c_mat, c_mask = _pairs_to_matrix(t["j_mag"], t["j_presence"], G)
        deg_values = self.vocab.from_index(t["deg_class"]).astype(np.int64)

        # Soft-equivalence target (G,G): two DISTINCT groups are soft-equivalent iff
        # they share a canonical symmetry orbit (chemically equivalent, not hard-
        # merged) — purely structural, NOT shift proximity. Built in the model's
        # canonical group order (same `order` as the shifts above). Zeros when the
        # record carries no equiv_orbit (legacy); the loss no-ops in that case.
        se_mat = np.zeros((G, G), dtype=np.float32)
        sym_perms = np.tile(np.arange(G, dtype=np.int64), (SYM_PERMS_MAX, 1))   # identity-only (no-op)
        orb = r.get("equiv_orbit")
        if orb is not None and len(orb) == G:
            oc = np.asarray(orb, dtype=np.int64)[self._orders[i]]
            eq = oc[:, None] == oc[None, :]
            np.fill_diagonal(eq, False)
            se_mat = eq.astype(np.float32)
            sym_perms = _orbit_perms(oc)                                        # (K,G) within-orbit relabelings

        item = {
            "spectrum":           torch.as_tensor(inp),                  # (P,)
            "spectrum_ref":       torch.as_tensor(clean),                # (P,)
            "shifts":             torch.from_numpy(t["shifts"]),         # (G,) standardized
            "couplings":          torch.from_numpy(c_mat),               # (G,G) standardized
            "coupling_mask":      torch.from_numpy(c_mask),              # (G,G) {0,1}
            "degeneracy_classes": torch.from_numpy(t["deg_class"]),      # (G,) long
            "degeneracy_values":  torch.from_numpy(deg_values),          # (G,) long
            "soft_equiv_target":  torch.from_numpy(se_mat),              # (G,G) {0,1}
            "sym_perms":          torch.from_numpy(sym_perms),           # (K,G) within-orbit relabelings
            "mol_id":             r["mol_id"],
            "smiles":             r.get("smiles"),
            "bucket_key":         self.bucket_keys[i],
        }
        if self.region_tokens:
            # extract from the (augmented) INPUT so regions match what the model sees
            from model.data.regions import extract_support_regions
            feats, mask = extract_support_regions(
                inp, self.ppm_from, self.ppm_to,
                max_regions=self.region_max, **self.region_kwargs)
            item["region_features"] = torch.from_numpy(feats)            # (R_max, F)
            item["region_mask"] = torch.from_numpy(mask)                 # (R_max,)
        return item
