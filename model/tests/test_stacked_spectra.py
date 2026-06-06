"""
Stacked-shard spectra source (PubChem 3M+ data path) tests.

StackedSpectra maps a global molecule index -> (part, row) across part_<k>.npy
shards; load_pubchem_records (gzip JSONL) carries record["row"] = that index; the
dataset fetches spectra via spectra_source[row]. We verify the index math, the
records<->parts alignment, gzip JSONL streaming + max_mol, and that the dataset
returns the spectrum belonging to the right molecule.
"""
import gzip
import json

import numpy as np

from model.data.dataset import SpectrumMatrixDataset
from model.data.records import load_pubchem_records
from model.data.stacked_spectra import StackedSpectra
from model.data.standardization import DegeneracyVocab, Standardizer
from model.schemas.constants import N_POINTS


def _write_parts(d, sizes, P=N_POINTS):
    """Write part_<k>.npy with distinct, identifiable rows; return the flat array."""
    rows, gidx = [], 0
    for k, n in enumerate(sizes):
        block = np.zeros((n, P), dtype=np.float32)
        for r in range(n):
            block[r, 0] = gidx          # tag row 0 with its global index
            gidx += 1
        np.save(d / f"part_{k}.npy", block)
        rows.append(block)
    return np.concatenate(rows, 0)


def _write_records(path, n):
    """A tiny JSONL.gz spin-systems file: n molecules, 2 groups each."""
    with gzip.open(path, "wt") as f:
        for i in range(n):
            rec = {"chembl_id": f"m{i}", "smiles": "CC",
                   "labels": ["A", "B"],
                   "spin_groups": [[2.0 + 0.001 * i, 1], [4.0, 2]],
                   "couplings": [["A", "B", 7.0]]}
            f.write(json.dumps(rec) + "\n")


def test_index_maps_global_to_part_row(tmp_path):
    flat = _write_parts(tmp_path, sizes=[3, 4, 2])     # 9 rows across 3 uneven parts
    ss = StackedSpectra(tmp_path)
    assert len(ss) == 9
    for i in range(9):
        assert ss[i][0] == i                            # row tag == global index
        assert np.array_equal(ss[i], flat[i])


def test_open_mmaps_capped_lru(tmp_path):
    """Shuffled access across many shards must NOT leak file descriptors — the
    LRU keeps at most max_open mmaps open (regression for OSError: Too many open
    files at 3200 shards)."""
    _write_parts(tmp_path, sizes=[2] * 20)              # 20 shards, 40 rows
    ss = StackedSpectra(tmp_path, max_open=5)
    for i in range(40):                                  # touch every shard
        _ = ss[i]
    assert len(ss._mmaps) <= 5


def test_preload_serves_from_ram(tmp_path):
    """preload() loads requested rows into RAM (sequential shard reads) and
    __getitem__ serves them, matching on-disk values; unrequested rows fall back
    to mmap."""
    flat = _write_parts(tmp_path, sizes=[3, 4, 3])      # 10 rows over 3 shards
    ss = StackedSpectra(tmp_path)
    want = [0, 1, 5, 9]
    ss.preload(want)
    assert ss._ram is not None and ss._ram.shape[0] == len(want)
    for i in want:
        assert ss[i][0] == i
        assert np.array_equal(ss[i], flat[i])
    assert np.array_equal(ss[3], flat[3])               # non-preloaded -> mmap fallback


def test_preload_cache_mmap_roundtrip(tmp_path):
    """A preload cache is written as a bare .npy (+ rows sidecar) and re-loaded via
    mmap on the next run, so co-located ranks share one copy. Values must survive
    the round-trip and the served array must be a memmap (shared), not a private copy."""
    flat = _write_parts(tmp_path, sizes=[3, 4, 3])
    cache = tmp_path / "preload.npy"
    want = [0, 1, 5, 9]
    ss = StackedSpectra(tmp_path)
    ss.preload(want, cache=str(cache))                  # build path -> writes cache
    assert cache.exists() and (tmp_path / "preload.npy.rows.npy").exists()

    ss2 = StackedSpectra(tmp_path)                      # fresh source -> load path (mmap)
    ss2.preload(want, cache=str(cache))
    assert isinstance(ss2._ram, np.memmap)             # shared via page cache, not a copy
    for i in want:
        assert np.array_equal(ss2[i], flat[i])
    # a cache missing a requested row is rejected as stale
    ss3 = StackedSpectra(tmp_path)
    try:
        ss3.preload([0, 1, 5, 9, 2], cache=str(cache))
        assert False, "expected stale-cache ValueError"
    except ValueError as e:
        assert "missing" in str(e)


def test_parts_sorted_numerically_not_lexically(tmp_path):
    # 12 parts: lexical sort would put part_10 before part_2
    _write_parts(tmp_path, sizes=[1] * 12)
    ss = StackedSpectra(tmp_path)
    assert len(ss) == 12
    for i in range(12):
        assert ss[i][0] == i


def test_preload_sparse_and_dense_paths(tmp_path):
    # 4 shards x 50 rows; row 0 of each tagged with its global index
    _write_parts(tmp_path, sizes=[50] * 4)
    # SPARSE: a few scattered rows -> per-shard mmap-only-needed path, correct values
    ss = StackedSpectra(tmp_path)
    want = [0, 50, 100, 150, 199]
    ss.preload(want)
    for r in want:
        assert ss[r][0] == r
    # DENSE: all of shard 2 (rows 100..149) -> whole-shard read path, still correct
    ss2 = StackedSpectra(tmp_path)
    ss2.preload(list(range(100, 150)))
    for r in (100, 125, 149):
        assert ss2[r][0] == r
    # non-preloaded row served via the mmap fallback
    ss3 = StackedSpectra(tmp_path)
    assert ss3[177][0] == 177


def test_records_gzip_jsonl_and_max_mol(tmp_path):
    rp = tmp_path / "spin_systems_pubchem.json.gz"
    _write_records(rp, 10)
    recs = load_pubchem_records(rp, max_mol=4)
    assert len(recs) == 4
    assert [r["row"] for r in recs] == [0, 1, 2, 3]
    assert recs[0]["mol_id"] == "mol_000000"
    assert recs[0]["shifts"].shape == (2,)


def test_out_of_vocab_degeneracy_filtered(tmp_path):
    """A molecule with an out-of-vocab degeneracy (e.g. 5) is skipped, and the
    surviving records keep their global ``row`` (so the spectrum mapping holds)."""
    rp = tmp_path / "recs.json.gz"
    with gzip.open(rp, "wt") as f:
        # idx 0,2 valid (deg 1,3); idx 1 has deg 5 (out of vocab) -> dropped
        for i, degs in enumerate([(1, 3), (1, 5), (3, 1)]):
            f.write(json.dumps({"chembl_id": f"m{i}", "labels": ["A", "B"],
                                "spin_groups": [[2.0, degs[0]], [4.0, degs[1]]],
                                "couplings": []}) + "\n")
    recs = load_pubchem_records(rp)
    assert [r["row"] for r in recs] == [0, 2]            # idx 1 filtered, rows preserved


def test_reservoir_sampling(tmp_path):
    """sample_n draws a uniform random subset (seeded, reproducible) whose records
    keep their global row; no duplicates."""
    rp = tmp_path / "recs.json.gz"
    _write_records(rp, 100)
    a = load_pubchem_records(rp, sample_n=10, sample_seed=0)
    b = load_pubchem_records(rp, sample_n=10, sample_seed=0)
    c = load_pubchem_records(rp, sample_n=10, sample_seed=1)
    assert len(a) == 10
    assert [r["row"] for r in a] == [r["row"] for r in b]    # deterministic per seed
    assert [r["row"] for r in a] != [r["row"] for r in c]    # seed changes the sample
    assert all(0 <= r["row"] < 100 for r in a)
    assert len(set(r["row"] for r in a)) == 10               # no duplicates


def test_dataset_pulls_correct_spectrum(tmp_path):
    parts_dir = tmp_path / "parts"; parts_dir.mkdir()
    flat = _write_parts(parts_dir, sizes=[3, 3])
    rp = tmp_path / "recs.json.gz"; _write_records(rp, 6)

    recs = load_pubchem_records(rp)
    ss = StackedSpectra(parts_dir)
    assert len(recs) == len(ss) == 6

    vocab = DegeneracyVocab()
    std = Standardizer().fit(recs, vocab)
    dsrc = SpectrumMatrixDataset(recs, vocab, std, augment=False, spectra_source=ss)
    # shuffle records to prove row-keying (not positional) drives the lookup
    import random; perm = list(range(6)); random.Random(0).shuffle(perm)
    shuffled = [recs[p] for p in perm]
    dshuf = SpectrumMatrixDataset(shuffled, vocab, std, augment=False, spectra_source=ss)
    for j, p in enumerate(perm):
        got = dshuf[j]["spectrum"].numpy()
        assert got[0] == p                               # fetched the molecule's own row
        assert np.array_equal(got, flat[p])
