"""
modelv2.data — data pipeline
============================
Everything that touches molecules before the model sees them, in one file:
splits, target encoding, standardization, the spectra RAM cache, the dataset,
collation, and augmentation.

Design contract (see DESIGN.md):

  * Target layout per molecule (G groups, canonical-ordered):
        shifts      (G,)            ppm        -> regression (standardized)
        j_mag       (n_pairs,)      Hz, triu   -> regression (standardized, masked)
        j_presence  (n_pairs,)      {0,1}      -> binary classification
        deg_class   (G,)            vocab idx  -> classification
    with  n_pairs = G*(G-1)/2  and the upper triangle taken row-major (i < j).

  * ONE canonical ordering and ONE upper-triangle index map, defined here and
    reused by target encoding, the heads, the losses, and the metrics. A
    mismatch silently corrupts J/presence learning, so it lives in exactly one
    place.

This module is import-safe **without torch** so that ``train.py --dry-run`` can
exercise the whole data path (load -> splits -> standardize -> encode -> mask)
in a torch-free environment. torch is imported lazily, only inside the dataset
collation and the on-GPU augmentation.
"""

from __future__ import annotations

import gzip
import io
import json
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np

__all__ = [
    "DEFAULT_DEG_VOCAB",
    "canonical_order",
    "reorder",
    "triu_index_map",
    "pairs_to_matrix",
    "matrix_to_pairs",
    "matrix_feature",
    "dedup_key",
    "DegeneracyVocab",
    "encode_target",
    "Standardizer",
    "class_balance",
    "make_splits",
    "record_to_arrays",
    "load_records",
    "renderable_mask",
    "SpectraCache",
    "SpinDataset",
    "collate",
    "augment_spectrum",
    "augment_spectrum_np",
]

# Degeneracy vocabulary: the integer proton-multiplicities the model can emit.
DEFAULT_DEG_VOCAB = (1, 2, 3, 4, 6, 9, 12, 18)


# =============================================================================
# Canonical ordering and the single upper-triangle index map
# =============================================================================

def canonical_order(shifts, couplings, degeneracy):
    """Permutation that sorts groups by shift desc, then degeneracy desc, then
    |J| row-sum desc. Deterministic — resolves the S_8 label arbitrariness into
    one ordering so a per-element loss is well defined.

    There is slight label noise for near-equal shifts; accepted at this stage
    (the eval-only Hungarian metric and the optional epsilon-band loss in
    train.py exist precisely to characterise / absorb it).
    """
    shifts = np.asarray(shifts, dtype=float)
    couplings = np.asarray(couplings, dtype=float)
    degeneracy = np.asarray(degeneracy, dtype=float)
    jrow = np.abs(couplings).sum(axis=1)
    # np.lexsort uses the LAST key as primary; negate every key for descending.
    return np.lexsort((-jrow, -degeneracy, -shifts))


def reorder(shifts, couplings, degeneracy, order):
    """Apply a permutation to (shifts, couplings, degeneracy)."""
    shifts = np.asarray(shifts, dtype=float)[order]
    degeneracy = np.asarray(degeneracy)[order]
    couplings = np.asarray(couplings, dtype=float)[np.ix_(order, order)]
    return shifts, couplings, degeneracy


# Cache the triu index arrays per G — used on every encode/decode.
_TRIU_CACHE: dict[int, tuple] = {}


def triu_index_map(G):
    """Return the (rows, cols) of the strict upper triangle in row-major order.

    This is THE coupling index map: ``j_mag[k]`` corresponds to the group pair
    ``(rows[k], cols[k])`` with rows[k] < cols[k]. Encoding, decoding, the loss
    masks, and the metrics all index couplings through this one function.
    """
    G = int(G)
    if G not in _TRIU_CACHE:
        _TRIU_CACHE[G] = np.triu_indices(G, 1)
    return _TRIU_CACHE[G]


def matrix_to_pairs(couplings, G=None):
    """(G, G) symmetric coupling matrix -> (n_pairs,) upper-triangle vector."""
    couplings = np.asarray(couplings, dtype=float)
    G = couplings.shape[-1] if G is None else G
    iu = triu_index_map(G)
    return couplings[iu[0], iu[1]]


def pairs_to_matrix(jpairs, G):
    """(..., n_pairs) upper-triangle vector -> (..., G, G) symmetric matrix."""
    jpairs = np.asarray(jpairs, dtype=float)
    iu = triu_index_map(G)
    out = np.zeros(jpairs.shape[:-1] + (G, G), dtype=float)
    out[..., iu[0], iu[1]] = jpairs
    out[..., iu[1], iu[0]] = jpairs
    return out


def matrix_feature(shifts, couplings, degeneracy):
    """Canonical 1-D invariant [shifts | triu J | deg] for near-dup detection."""
    order = canonical_order(shifts, couplings, degeneracy)
    s, c, d = reorder(shifts, couplings, degeneracy, order)
    return np.concatenate([s, matrix_to_pairs(c, len(s)), d.astype(float)])


def dedup_key(shifts, couplings, degeneracy, shift_tol=0.02, j_tol=0.5):
    """Hashable key collapsing near-identical matrices into one bucket.

    Rounds the canonical feature to the tolerance grid (shifts ppm, J Hz),
    degeneracy kept exact. O(1) per molecule.
    """
    order = canonical_order(shifts, couplings, degeneracy)
    s, c, d = reorder(shifts, couplings, degeneracy, order)
    s_q = np.round(s / shift_tol).astype(np.int64)
    j_q = np.round(matrix_to_pairs(c, len(s)) / j_tol).astype(np.int64)
    return (tuple(s_q.tolist()), tuple(j_q.tolist()), tuple(int(x) for x in d))


# =============================================================================
# Degeneracy vocabulary (classification target)
# =============================================================================

class DegeneracyVocab:
    """Bijection between integer degeneracies and contiguous class indices."""

    def __init__(self, vocab=DEFAULT_DEG_VOCAB):
        self.vocab = tuple(int(v) for v in vocab)
        self._to = {v: i for i, v in enumerate(self.vocab)}

    def __len__(self):
        return len(self.vocab)

    def __eq__(self, other):
        return isinstance(other, DegeneracyVocab) and other.vocab == self.vocab

    def contains(self, degeneracy) -> bool:
        return all(int(round(float(d))) in self._to
                   for d in np.asarray(degeneracy).ravel())

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
        flat = np.array(self.vocab, dtype=np.int64)[idx.ravel()]
        return flat.reshape(idx.shape)


# =============================================================================
# Encode one molecule into target components (canonical-ordered)
# =============================================================================

def _rec_arrays(rec):
    return (np.asarray(rec["shifts"], dtype=float),
            np.asarray(rec["couplings"], dtype=float),
            np.asarray(rec["degeneracy"]))


def encode_target(rec, vocab: DegeneracyVocab, order=None, j_zero_tol=1e-6):
    """Encode a record dict into the (unstandardized) target component arrays.

    Returns a dict with float32 ``shifts`` (G,), ``j_mag`` (n_pairs,),
    ``j_presence`` (n_pairs,) in {0,1}, int64 ``deg_class`` (G,), plus the
    canonical ``order`` actually used.
    """
    shifts, couplings, degeneracy = _rec_arrays(rec)
    if order is None:
        order = canonical_order(shifts, couplings, degeneracy)
    s, c, d = reorder(shifts, couplings, degeneracy, order)
    j_mag = matrix_to_pairs(c, len(s)).astype(float)
    j_presence = (np.abs(j_mag) > j_zero_tol).astype(np.float32)
    return dict(
        shifts=s.astype(np.float32),
        j_mag=j_mag.astype(np.float32),
        j_presence=j_presence,
        deg_class=vocab.to_index(d),
        order=np.asarray(order),
    )


# =============================================================================
# Standardizer (fit on TRAIN only; saved in the checkpoint; reused verbatim)
# =============================================================================

class Standardizer:
    """Z-score shifts (all groups) and coupling magnitudes (present pairs only).

    Denominators are floored so a zero-variance cell can never produce inf/nan
    (a hard numerical-stability requirement).
    """

    STD_FLOOR = 1e-6

    def __init__(self):
        self.shift_mean = self.shift_std = None
        self.j_mean = self.j_std = None

    # -- fit / apply -----------------------------------------------------------

    def fit(self, train_records, vocab: DegeneracyVocab):
        shifts_all, j_present = [], []
        for r in train_records:
            t = encode_target(r, vocab)
            shifts_all.append(t["shifts"])
            j_present.append(t["j_mag"][t["j_presence"] > 0])
        shifts_all = np.concatenate(shifts_all) if shifts_all else np.array([0.0])
        j_present = (np.concatenate(j_present)
                     if any(len(x) for x in j_present) else np.array([0.0]))
        self.shift_mean = float(shifts_all.mean())
        self.shift_std = float(max(shifts_all.std(), self.STD_FLOOR))
        self.j_mean = float(j_present.mean())
        self.j_std = float(max(j_present.std(), self.STD_FLOOR))
        return self

    def transform(self, t):
        out = dict(t)
        out["shifts"] = ((t["shifts"] - self.shift_mean) / self.shift_std).astype(np.float32)
        jm = (t["j_mag"] - self.j_mean) / self.j_std
        out["j_mag"] = (jm * t["j_presence"]).astype(np.float32)  # zero where absent
        return out

    def inverse_shifts(self, x):
        return np.asarray(x) * self.shift_std + self.shift_mean

    def inverse_j(self, x):
        return np.asarray(x) * self.j_std + self.j_mean

    # -- (de)serialisation for the checkpoint ---------------------------------

    def state_dict(self):
        return dict(shift_mean=self.shift_mean, shift_std=self.shift_std,
                    j_mean=self.j_mean, j_std=self.j_std)

    @classmethod
    def from_state(cls, d):
        s = cls()
        s.shift_mean = float(d["shift_mean"]); s.shift_std = float(d["shift_std"])
        s.j_mean = float(d["j_mean"]); s.j_std = float(d["j_std"])
        return s

    def assert_valid(self):
        for name in ("shift_std", "j_std"):
            v = getattr(self, name)
            assert v is not None and np.isfinite(v) and v > 0, \
                f"Standardizer.{name} must be finite and > 0 (got {v!r})"


# =============================================================================
# Class balancing (fit on TRAIN) — fixes majority-class collapse
# =============================================================================

def class_balance(train_records, vocab: DegeneracyVocab, power=0.5, cap=8.0):
    """Tempered inverse-frequency degeneracy CE weights + BCE presence pos_weight.

    Degeneracy is ~89% d=1, so plain 1/freq would zero the majority class under a
    ~760:1 imbalance; ``power=0.5`` (inverse-sqrt) with a cap keeps d=1 meaningful
    while up-weighting rare degeneracies. Couplings are ~70% absent, so the
    presence head gets ``pos_weight = #absent / #present``.

    All denominators are floored so a zero-count class cannot produce inf.
    """
    C = len(vocab)
    deg_counts = np.zeros(C)
    n_present = 0
    n_pairs_total = 0
    for r in train_records:
        t = encode_target(r, vocab)
        for c in t["deg_class"]:
            deg_counts[int(c)] += 1
        n_present += int(t["j_presence"].sum())
        n_pairs_total += int(t["j_presence"].size)

    present = deg_counts > 0
    deg_weights = np.zeros(C)
    if present.any():
        raw = (1.0 / np.maximum(deg_counts[present], 1.0)) ** power
        raw = raw / raw.mean()
        raw = np.clip(raw, 1.0 / cap, cap)
        raw = raw / raw.mean()
        deg_weights[present] = raw
    n_absent = n_pairs_total - n_present
    presence_pos_weight = float(n_absent / max(n_present, 1))
    return dict(deg_weights=deg_weights.astype(np.float32),
                presence_pos_weight=presence_pos_weight,
                deg_counts=deg_counts.astype(int))


# =============================================================================
# Molecule-level scaffold + near-duplicate split
# =============================================================================

class _UF:
    """Union-find with path halving."""

    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def _stratum_key(rec, n_density_bins=4):
    """(degeneracy multiset, coupling-density bin) — balanced across folds."""
    deg = tuple(sorted((int(x) for x in np.asarray(rec["degeneracy"])), reverse=True))
    c = np.asarray(rec["couplings"], dtype=float)
    G = c.shape[0]
    n_off = G * (G - 1) / 2
    iu = triu_index_map(G)
    density = (np.abs(c[iu[0], iu[1]]) > 0).sum() / max(n_off, 1)
    dbin = min(int(density * n_density_bins), n_density_bins - 1)
    return (deg, dbin)


def make_splits(records, ratios=(0.7, 0.2, 0.1), seed=0,
                shift_tol=0.02, j_tol=0.5, compute_scaffold=True):
    """Assign each molecule to ``train`` / ``val`` / ``test``.

    Molecules that are a
    near-duplicate (shift_tol, j_tol) matrix are unioned into one group; whole
    groups land in a single fold so no spin-system near-relative straddles the
    split. Groups are assigned with a stratified, size-aware greedy rule.

    Returns ``(assignment, report)`` where ``assignment`` maps mol_id -> fold.
    """
    recs = list(records)
    n = len(recs)
    fold_names = ("train", "val", "test")
    assert abs(sum(ratios) - 1.0) < 1e-9 and len(ratios) == 3

    scaffolds, keys = [], []
    for r in recs:
        if r.get("scaffold") is not None:
            scaffolds.append(r["scaffold"])
        else:
            scaffolds.append(None)
        keys.append(dedup_key(r["shifts"], r["couplings"], r["degeneracy"],
                              shift_tol, j_tol))

    uf = _UF(n)
    by_scaffold, by_key = defaultdict(list), defaultdict(list)
    for i in range(n):
        if scaffolds[i]:
            by_scaffold[scaffolds[i]].append(i)
        by_key[keys[i]].append(i)
    for members in list(by_scaffold.values()) + list(by_key.values()):
        for j in members[1:]:
            uf.union(members[0], j)

    groups = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    group_list = list(groups.values())

    rng = np.random.default_rng(seed)

    def group_stratum(g):
        cnt = defaultdict(int)
        for i in g:
            cnt[_stratum_key(recs[i])] += 1
        return max(sorted(cnt), key=lambda k: cnt[k])

    strata = defaultdict(list)
    for g in group_list:
        strata[group_stratum(g)].append(g)

    total = sum(len(g) for g in group_list)
    target = np.array(ratios) * total
    fold_count = np.zeros(3)
    assignment = {}
    for stratum in sorted(strata.keys()):
        gs = strata[stratum]
        order = sorted(range(len(gs)), key=lambda idx: (-len(gs[idx]), rng.random()))
        for idx in order:
            g = gs[idx]
            deficit = (target - fold_count) / np.maximum(target, 1)
            f = int(np.argmax(deficit))
            fold_count[f] += len(g)
            for i in g:
                assignment[recs[i]["mol_id"]] = fold_names[f]

    # leakage self-checks
    scaf_folds, key_folds = defaultdict(set), defaultdict(set)
    for i in range(n):
        mid = recs[i]["mol_id"]
        if scaffolds[i]:
            scaf_folds[scaffolds[i]].add(assignment[mid])
        key_folds[keys[i]].add(assignment[mid])
    report = dict(
        n_molecules=n, n_groups=len(group_list), n_strata=len(strata),
        counts={fn: int(fold_count[k]) for k, fn in enumerate(fold_names)},
        ratios={fn: fold_count[k] / max(total, 1) for k, fn in enumerate(fold_names)},
        target_ratios=dict(zip(fold_names, ratios)),
        scaffold_leaks=sum(1 for v in scaf_folds.values() if len(v) > 1),
        dup_matrix_leaks=sum(1 for v in key_folds.values() if len(v) > 1),
        used_scaffold=compute_scaffold and any(scaffolds),
        seed=seed,
    )
    return assignment, report


# =============================================================================
# Loading the spin-system ground truth
# =============================================================================

# Schema field names (single source of truth; matches the data README).
KEY_LABELS = "labels"
KEY_GROUPS = "spin_groups"     # [ [shift_ppm, degeneracy], ... ] aligned to labels
KEY_COUPLINGS = "couplings"    # [ [label_i, label_j, J_Hz], ... ]; absent => 0
ID_KEYS = ("chembl_id", "smiles", "inchikey")


def record_to_arrays(rec):
    """Convert one spin-system JSON record to (shifts, couplings, degeneracy).

    ``couplings`` is the symmetric (G, G) matrix in Hz (absent pairs = 0), built
    from the record's own label order (index-aligned with ``spin_groups``).
    """
    labels = list(rec[KEY_LABELS])
    groups = rec[KEY_GROUPS]
    if len(labels) != len(groups):
        raise ValueError(f"labels ({len(labels)}) / spin_groups ({len(groups)}) mismatch")
    index = {lab: i for i, lab in enumerate(labels)}
    G = len(labels)
    shifts = np.array([float(groups[i][0]) for i in range(G)], dtype=float)
    degeneracy = np.array([int(groups[i][1]) for i in range(G)], dtype=int)
    couplings = np.zeros((G, G), dtype=float)
    for a, b, j in rec.get(KEY_COUPLINGS, []):
        i, k = index[a], index[b]
        couplings[i, k] = couplings[k, i] = float(j)
    return shifts, couplings, degeneracy


def _iter_spin_systems(path):
    """Yield ``(index, record)`` from a spin-systems file.

    Accepts a plain ``.json`` (single array or JSONL), a gzipped ``.json.gz``,
    or a ``.tar.gz`` containing a single ``*.json`` member — the on-disk forms
    the pipeline ships in.
    """
    path = Path(path)
    name = path.name.lower()

    def _emit(text):
        text = text.lstrip()
        if text.startswith("version https://git-lfs"):
            raise ValueError(f"{path} is an unresolved git-LFS pointer; run `git lfs pull`.")
        # Robust to three on-disk forms: a single JSON array, several arrays
        # concatenated (the pubchem dump is chunked as ``...][...`` and even
        # ends with an empty ``[]``), and JSONL. ``raw_decode`` walks successive
        # top-level values; arrays are flattened and objects yielded directly,
        # with ONE running index across the whole file so ``mol_<idx>`` stays
        # aligned with the spectra enumeration.
        dec = json.JSONDecoder()
        i, n, idx = 0, len(text), 0
        ws = " \t\r\n"
        while True:
            while i < n and text[i] in ws:
                i += 1
            if i >= n:
                break
            obj, i = dec.raw_decode(text, i)
            if isinstance(obj, list):
                for rec in obj:
                    yield idx, rec
                    idx += 1
            elif isinstance(obj, dict):
                yield idx, obj
                idx += 1

    if name.endswith(".tar.gz") or name.endswith(".tgz") or name.endswith(".tar"):
        mode = "r:gz" if name.endswith((".tar.gz", ".tgz")) else "r:"
        with tarfile.open(path, mode) as tf:
            members = [m for m in tf.getmembers()
                       if m.isfile() and m.name.lower().endswith(".json")]
            if not members:
                raise ValueError(f"no .json member found inside {path}")
            members.sort(key=lambda m: m.name)
            f = tf.extractfile(members[0])
            yield from _emit(f.read().decode("utf-8"))
    elif name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            yield from _emit(f.read())
    else:
        yield from _emit(path.read_text())


def load_records(spin_systems_path, max_records=None):
    """Load the spin-system ground truth into ``records`` dicts.

    ``mol_id = mol_{idx:06d}`` is index-aligned with the spectra filenames, so it
    is the join key against :class:`SpectraCache`. No spectra are read here.
    """
    records = []
    for idx, rec in _iter_spin_systems(spin_systems_path):
        try:
            shifts, couplings, degeneracy = record_to_arrays(rec)
        except Exception:
            continue
        records.append({
            "mol_id": f"mol_{idx:06d}",
            "index": idx,
            "shifts": shifts,
            "couplings": couplings,
            "degeneracy": degeneracy,
            "smiles": rec.get("smiles"),
            "chembl_id": rec.get("chembl_id"),
            "inchikey": rec.get("inchikey"),
            "n_spins": int(degeneracy.sum()),
        })
        if max_records is not None and len(records) >= max_records:
            break
    return records


def renderable_mask(records, vocab: DegeneracyVocab, n_groups=8):
    """Boolean list: molecules usable by the fixed G-group model.

    A molecule is usable iff it has exactly ``n_groups`` groups, every degeneracy
    is in the vocabulary, and its shifts/couplings are finite. (modelv2 has no
    Stage-2 renderer, so this is the simple "can the fixed-shape model consume
    it" check that ``--dry-run`` validates against.)
    """
    out = []
    for r in records:
        s = np.asarray(r["shifts"], dtype=float)
        c = np.asarray(r["couplings"], dtype=float)
        ok = (len(s) == n_groups
              and np.isfinite(s).all() and np.isfinite(c).all()
              and vocab.contains(r["degeneracy"]))
        out.append(bool(ok))
    return out


# =============================================================================
# Spectra RAM cache — stream the archive once into a single fp16 array
# =============================================================================

class SpectraCache:
    """Hold every needed spectrum in one fp16 array in RAM, indexed by molecule identity.

    At construction ``simulation/data/spectra/90MHz.tar.gz`` is streamed once.
    ``index.csv`` inside the archive maps ``mol_<idx>`` filenames to molecule
    identity strings (first non-empty of chembl_id / smiles / inchikey, same
    priority as the simulation pipeline).  Spectra are stored and retrieved by
    that identity string — not by filename position — so the association is
    robust to any reordering of the source JSON.

    ``archive`` may also be a directory containing ``mol_*.npy`` under a
    ``<field>MHz/`` subfolder with ``index.csv`` at the directory root, which
    is handy for local development.

    Defined at module level (no closures) so it is picklable for DataLoader.
    """

    def __init__(self, records, archive, points=16384, spectrum_field="spec90",
                 verbose=True):
        self.points = int(points)
        self.spectrum_field = spectrum_field
        want = {_record_mol_id(r) for r in records if _record_mol_id(r)}
        self.row = {}  # identity_string -> row in self.data
        self.data = np.zeros((len(want), self.points), dtype=np.float16)
        self.ppm_axis = None
        self.n_loaded = 0
        self.n_missing = 0
        archive = Path(archive)
        if archive.is_dir():
            self._load_dir(archive, want, spectrum_field, verbose)
        else:
            self._load_tar(archive, want, verbose)
        self.n_missing = len(want) - self.n_loaded
        if verbose:
            print(f"[SpectraCache] loaded {self.n_loaded}/{len(want)} spectra "
                  f"({self.data.nbytes / 1e9:.2f} GB fp16); missing {self.n_missing}")

    @staticmethod
    def _mol_id_from_name(name):
        stem = Path(name).name
        if not stem.lower().endswith(".npy"):
            return None
        stem = stem[:-4]
        if not stem.startswith("mol_"):
            return None
        return stem

    def _store(self, id_str, arr, want):
        if not id_str or id_str not in want or id_str in self.row:
            return
        arr = np.asarray(arr, dtype=np.float32).ravel()
        if arr.shape[0] != self.points:
            arr = _resample(arr, self.points)
        r = len(self.row)
        self.row[id_str] = r
        self.data[r] = arr.astype(np.float16)
        self.n_loaded += 1

    def _load_tar(self, archive, want, verbose):
        import csv as _csv
        name = archive.name.lower()
        mode = "r:gz" if name.endswith((".tar.gz", ".tgz")) else (
            "r:" if name.endswith(".tar") else "r:*")
        # index.csv maps mol_<idx> filenames -> identity strings.
        # Canonical archive order: MANIFEST → ppm_axis → index.csv → mol_*.npy,
        # so the buffer below is normally empty. Spectra encountered before
        # index.csv is read are buffered by mol_id and resolved once it arrives.
        mol_id_to_id: dict[str, str] = {}
        pending: dict[str, bytes] = {}  # mol_id -> raw .npy bytes (pre-index)
        with tarfile.open(archive, mode) as tf:
            for m in tf:
                if not m.isfile():
                    continue
                base = Path(m.name).name
                if base == "ppm_axis.npy" and self.ppm_axis is None:
                    self.ppm_axis = np.load(io.BytesIO(tf.extractfile(m).read()))
                    continue
                if base == "index.csv":
                    raw = tf.extractfile(m).read().decode("utf-8")
                    for row in _csv.DictReader(io.StringIO(raw)):
                        mol_id_to_id[row["index"]] = row.get("id", "").strip()
                    for mid, data in pending.items():
                        self._store(mol_id_to_id.get(mid, ""),
                                    np.load(io.BytesIO(data)), want)
                    pending.clear()
                    continue
                mol_id = self._mol_id_from_name(m.name)
                if mol_id is None:
                    continue
                raw = tf.extractfile(m).read()
                if mol_id_to_id:
                    self._store(mol_id_to_id.get(mol_id, ""),
                                np.load(io.BytesIO(raw)), want)
                else:
                    pending[mol_id] = raw

    def _load_dir(self, root, want, spectrum_field, verbose):
        import csv as _csv
        field = "".join(ch for ch in spectrum_field if ch.isdigit())
        search = root / f"{field}MHz" if field and (root / f"{field}MHz").is_dir() else root
        ax = search / "ppm_axis.npy"
        if ax.exists():
            self.ppm_axis = np.load(ax)
        mol_id_to_id: dict[str, str] = {}
        for csv_path in [root / "index.csv", search / "index.csv"]:
            if csv_path.exists():
                with csv_path.open(newline="") as fh:
                    for row in _csv.DictReader(fh):
                        mol_id_to_id[row["index"]] = row.get("id", "").strip()
                break
        for p in sorted(search.glob("mol_*.npy")):
            mol_id = self._mol_id_from_name(p.name)
            if mol_id is None:
                continue
            self._store(mol_id_to_id.get(mol_id, ""), np.load(p), want)

    # -- access ----------------------------------------------------------------

    def __contains__(self, id_str):
        return id_str in self.row

    def get(self, id_str):
        """Return the fp32 spectrum for the molecule with this identity string."""
        r = self.row.get(id_str)
        if r is None:
            return np.zeros(self.points, dtype=np.float32)
        return self.data[r].astype(np.float32)


def _resample(arr, points):
    """Linear-interpolate a 1-D spectrum onto ``points`` samples (rare path)."""
    if arr.shape[0] == points:
        return arr
    xp = np.linspace(0.0, 1.0, arr.shape[0])
    x = np.linspace(0.0, 1.0, points)
    return np.interp(x, xp, arr)


def _record_mol_id(rec):
    """First non-empty value from (chembl_id, smiles, inchikey) — mirrors
    simulation.graph_io.molecule_id so the comparison is apples-to-apples."""
    for k in ID_KEYS:
        v = rec.get(k)
        if v:
            return str(v).strip()
    return ""


# =============================================================================
# Dataset + collation
# =============================================================================

class SpinDataset:
    """One item = one molecule: a raw (un-augmented) spectrum + encoded targets.

    Augmentation is intentionally NOT applied here — it is done on-GPU to the
    whole batch (see :func:`augment_spectrum`), which keeps ``num_workers=0`` and
    removes the CPU loader as a bottleneck. The dataset is duck-typed (no torch
    subclassing) so this module imports without torch; ``torch.utils.data.
    DataLoader`` consumes it fine.

    Spectrum source priority: the RAM ``cache`` -> an in-memory array on the
    record (``rec[spectrum_field]``) -> a ``.npy`` path (``rec[field+'_path']``).
    """

    def __init__(self, records, vocab: DegeneracyVocab, std: Standardizer,
                 cache: "SpectraCache | None" = None, spectrum_field="spec90",
                 points=16384):
        self.records = list(records)
        self.vocab = vocab
        self.std = std
        self.cache = cache
        self.spectrum_field = spectrum_field
        self.points = int(points)
        # Precompute canonical orders + encoded/standardized targets once.
        self._orders = [canonical_order(r["shifts"], r["couplings"], r["degeneracy"])
                        for r in self.records]
        self._targets = [std.transform(encode_target(r, vocab, order=o))
                         for r, o in zip(self.records, self._orders)]

    def __len__(self):
        return len(self.records)

    def _spectrum(self, r):
        if self.cache is not None:
            id_str = _record_mol_id(r)
            if id_str and id_str in self.cache:
                return self.cache.get(id_str)
        if r.get(self.spectrum_field) is not None:
            return np.asarray(r[self.spectrum_field], dtype=np.float32)
        path = r.get(self.spectrum_field + "_path")
        if path:
            return np.load(path, mmap_mode="r").astype(np.float32)
        return np.zeros(self.points, dtype=np.float32)

    def __getitem__(self, i):
        r = self.records[i]
        t = self._targets[i]
        return {
            "spectrum": np.asarray(self._spectrum(r), dtype=np.float32),
            "shifts": t["shifts"],
            "j_mag": t["j_mag"],
            "j_presence": t["j_presence"],
            "deg_class": t["deg_class"].astype(np.int64),
            "index": i,
        }


def collate(batch):
    """Stack a list of dataset items into a batch of torch tensors."""
    import torch
    out = {
        "spectrum": torch.from_numpy(np.stack([b["spectrum"] for b in batch])).float(),
        "shifts": torch.from_numpy(np.stack([b["shifts"] for b in batch])).float(),
        "j_mag": torch.from_numpy(np.stack([b["j_mag"] for b in batch])).float(),
        "j_presence": torch.from_numpy(np.stack([b["j_presence"] for b in batch])).float(),
        "deg_class": torch.from_numpy(np.stack([b["deg_class"] for b in batch])).long(),
        "index": torch.tensor([b["index"] for b in batch], dtype=torch.long),
    }
    return out


# =============================================================================
# Augmentation — vectorized over the batch, run on-GPU in the training step
# =============================================================================

def augment_spectrum(spec, ppm_from=0.0, ppm_to=12.0, *, noise_sigma_frac=0.01,
                     max_ref_shift_ppm=0.01, baseline_amp_frac=0.02,
                     broaden_sigma_pts=0.0, generator=None):
    """Augment a batch of normalized spectra **in torch**, vectorized over B.

    ``spec`` is a (B, P) tensor (the whole batch, already on the GPU). Returns a
    (B, P) tensor, each row clipped non-negative and renormalized to unit
    integral. All four physical perturbations from DESIGN.md are applied:
    sub-pixel referencing shift, low-frequency baseline drift, optional Gaussian
    broadening, and additive Gaussian noise.
    """
    import torch

    x = spec
    B, P = x.shape
    device, dtype = x.device, x.dtype
    dx = (ppm_to - ppm_from) / P
    peak = x.amax(dim=1, keepdim=True).clamp_min(1e-8)  # (B,1)

    def rand(*shape):
        return torch.rand(*shape, device=device, dtype=dtype, generator=generator)

    def randn(*shape):
        return torch.randn(*shape, device=device, dtype=dtype, generator=generator)

    # 1) sub-pixel referencing shift via batched linear interpolation
    if max_ref_shift_ppm and max_ref_shift_ppm > 0:
        shift_px = (rand(B, 1) * 2 - 1) * (max_ref_shift_ppm / dx)  # (B,1) pixels
        base = torch.arange(P, device=device, dtype=dtype).unsqueeze(0)  # (1,P)
        src = base - shift_px                                            # (B,P)
        lo = torch.floor(src)
        frac = (src - lo)
        lo = lo.long()
        hi = lo + 1
        valid_lo = (lo >= 0) & (lo < P)
        valid_hi = (hi >= 0) & (hi < P)
        lo_c = lo.clamp(0, P - 1)
        hi_c = hi.clamp(0, P - 1)
        x_lo = torch.gather(x, 1, lo_c) * valid_lo.to(dtype)
        x_hi = torch.gather(x, 1, hi_c) * valid_hi.to(dtype)
        x = x_lo * (1 - frac) + x_hi * frac

    # 2) optional Gaussian broadening (depthwise conv with a shared kernel)
    if broaden_sigma_pts and broaden_sigma_pts > 0:
        k = int(max(3, round(6 * broaden_sigma_pts)))
        t = torch.arange(-k, k + 1, device=device, dtype=dtype)
        g = torch.exp(-0.5 * (t / broaden_sigma_pts) ** 2)
        g = (g / g.sum()).view(1, 1, -1)
        x = torch.nn.functional.conv1d(x.unsqueeze(1), g, padding=k).squeeze(1)

    # 3) low-frequency sinusoidal baseline drift (kept non-negative)
    if baseline_amp_frac and baseline_amp_frac > 0:
        phase = rand(B, 1) * (2 * np.pi)
        freq = 0.5 + rand(B, 1) * 1.5
        grid = torch.linspace(0.0, float(np.pi), P, device=device, dtype=dtype).unsqueeze(0)
        base = baseline_amp_frac * peak * torch.sin(grid * freq + phase)
        x = x + (base - base.amin(dim=1, keepdim=True))

    # 4) additive Gaussian noise (fraction of per-spectrum peak height)
    if noise_sigma_frac and noise_sigma_frac > 0:
        x = x + randn(B, P) * (noise_sigma_frac * peak)

    x = x.clamp_min(0.0)
    area = x.sum(dim=1, keepdim=True) * dx
    return torch.where(area > 0, x / area, x)


def augment_spectrum_np(spec, ppm_from=0.0, ppm_to=12.0, rng=None,
                        noise_sigma_frac=0.01, max_ref_shift_ppm=0.01,
                        baseline_amp_frac=0.02, broaden_sigma_pts=0.0):
    """NumPy reference for a single spectrum — mirrors :func:`augment_spectrum`.

    Kept torch-free for unit tests and any CPU-only path.
    """
    rng = rng or np.random.default_rng()
    spec = np.asarray(spec, dtype=float).copy()
    P = len(spec)
    dx = (ppm_to - ppm_from) / P
    peak = spec.max() if spec.max() > 0 else 1.0

    if max_ref_shift_ppm > 0:
        shift_ppm = rng.uniform(-max_ref_shift_ppm, max_ref_shift_ppm)
        xg = np.arange(P)
        spec = np.interp(xg - shift_ppm / dx, xg, spec, left=0.0, right=0.0)
    if broaden_sigma_pts > 0:
        k = int(max(3, round(6 * broaden_sigma_pts)))
        t = np.arange(-k, k + 1)
        g = np.exp(-0.5 * (t / broaden_sigma_pts) ** 2)
        g /= g.sum()
        spec = np.convolve(spec, g, mode="same")
    if baseline_amp_frac > 0:
        phase = rng.uniform(0, 2 * np.pi)
        freq = rng.uniform(0.5, 2.0)
        base = baseline_amp_frac * peak * np.sin(np.linspace(0, freq * np.pi, P) + phase)
        spec = spec + (base - base.min())
    if noise_sigma_frac > 0:
        spec = spec + rng.normal(0, noise_sigma_frac * peak, P)
    spec = np.clip(spec, 0.0, None)
    area = spec.sum() * dx
    return (spec / area if area > 0 else spec).astype(np.float32)


# =============================================================================
# Torch-free self-test (run: python -m modelv2.data)
# =============================================================================

def _selftest():
    rng = np.random.default_rng(0)
    G = 8

    def mol(i):
        c = np.zeros((G, G))
        for a in range(G):
            for b in range(a + 1, G):
                if rng.random() < 0.4:
                    c[a, b] = c[b, a] = float(rng.uniform(1, 12))
        return dict(mol_id=f"mol_{i:06d}", smiles=None,
                    shifts=rng.uniform(0.5, 9, G),
                    couplings=c,
                    degeneracy=rng.choice([1, 2, 3, 6], size=G).astype(int))

    recs = [mol(i) for i in range(200)]
    vocab = DegeneracyVocab()

    # canonical order is a valid permutation and idempotent
    o = canonical_order(recs[0]["shifts"], recs[0]["couplings"], recs[0]["degeneracy"])
    assert sorted(o.tolist()) == list(range(G))
    s, c, d = reorder(recs[0]["shifts"], recs[0]["couplings"], recs[0]["degeneracy"], o)
    assert np.allclose(canonical_order(s, c, d), np.arange(G))

    # encode -> standardize round-trips through the shared index map
    std = Standardizer().fit(recs, vocab)
    std.assert_valid()
    t = std.transform(encode_target(recs[0], vocab))
    assert t["shifts"].shape == (G,) and t["j_mag"].shape == (G * (G - 1) // 2,)
    # recover physical shifts
    back = std.inverse_shifts(t["shifts"])
    s0, c0, d0 = reorder(recs[0]["shifts"], recs[0]["couplings"], recs[0]["degeneracy"], o)
    assert np.allclose(back, s0, atol=1e-4)

    # presence flags match the canonical coupling matrix exactly
    iu = triu_index_map(G)
    assert np.array_equal(t["j_presence"], (np.abs(c0[iu[0], iu[1]]) > 1e-6).astype(np.float32))

    # pairs_to_matrix is the inverse of matrix_to_pairs
    M = pairs_to_matrix(matrix_to_pairs(c0, G), G)
    assert np.allclose(M, c0)

    # class balance has no inf/nan and presence pos_weight is positive
    cb = class_balance(recs, vocab)
    assert np.isfinite(cb["deg_weights"]).all() and cb["presence_pos_weight"] > 0

    # splits: ratios in range, no near-dup matrix straddles folds
    assign, rep = make_splits(recs, seed=0, compute_scaffold=False)
    assert set(assign.values()) <= {"train", "val", "test"}
    assert rep["dup_matrix_leaks"] == 0
    assert abs(rep["ratios"]["train"] - 0.7) < 0.15

    # renderable mask: all synthetic mols are G=8 with in-vocab degeneracies
    assert all(renderable_mask(recs, vocab, n_groups=8))

    # numpy augmentation preserves unit integral and non-negativity
    spec = np.abs(rng.normal(size=1024)); spec /= spec.sum() * (12.0 / 1024)
    aug = augment_spectrum_np(spec, rng=rng)
    assert (aug >= 0).all() and abs(aug.sum() * (12.0 / 1024) - 1.0) < 1e-3

    # SpectraCache: chembl_id-based identity lookup via directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_root = Path(tmpdir)
        (cache_root / "90MHz").mkdir()
        (cache_root / "index.csv").write_text(
            "index,id\n" + "".join(f"mol_{i:06d},CHEMBL{i:04d}\n" for i in range(5))
        )
        for i in range(5):
            np.save(cache_root / "90MHz" / f"mol_{i:06d}.npy",
                    np.ones(16, dtype=np.float32) * i)
        id_recs = [dict(mol(i), chembl_id=f"CHEMBL{i:04d}") for i in range(5)]
        sc = SpectraCache(id_recs, cache_root, points=16, spectrum_field="spec90", verbose=False)
        assert sc.n_loaded == 5, f"expected 5 loaded, got {sc.n_loaded}"
        for i in range(5):
            s = sc.get(f"CHEMBL{i:04d}")
            assert abs(float(s.mean()) - i) < 0.01, f"CHEMBL{i:04d}: got {s.mean()}, want {i}"
        # unknown chembl_id returns zeros, not an error
        assert sc.get("CHEMBL9999").sum() == 0

    print("data.py self-test PASSED",
          {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rep["ratios"].items()})


if __name__ == "__main__":
    _selftest()
