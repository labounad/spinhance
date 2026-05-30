"""
export.py
=========
Pack a `spectra/` directory into a single compressed tarball for sharing /
handing to Task 4.

Two stages, each with a progress bar:
  1. **compress** — optionally sparsify each spectrum (drop points ≤ cutoff·max,
     renormalise to ∫=1, store as int32 indices + float32 values) and add to an
     uncompressed tar.
  2. **zip** — gzip the tar to ``.tar.gz``.

Sparse format (per spectrum, a ``.npz`` inside the tar): ``idx`` (nonzero point
indices), ``val`` (their intensities, renormalised so ∫=1), ``n`` (grid length),
``cutoff`` (the fraction used). Reconstruct with :func:`load_spectrum`:
``y = zeros(n); y[idx] = val``.
"""

from __future__ import annotations

import io
import gzip
import tarfile
from pathlib import Path

import numpy as np

# Re-export the canonical representation helpers (kept here for back-compat).
from simulation.spectrum_io import sparsify, load_spectrum  # noqa: F401

try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None

__all__ = ["export_spectra", "sparsify", "load_spectrum"]


def _bar(iterable, total, desc):
    if _tqdm is None:
        return iterable
    return _tqdm(iterable, total=total, desc=desc, unit="file")


def _sparse_bytes(y: np.ndarray, cutoff: float, renormalize: bool) -> bytes:
    idx, val = sparsify(y, cutoff, renormalize)
    buf = io.BytesIO()
    # uncompressed npz — the final gzip pass compresses once (avoids slow,
    # redundant double compression over thousands of files)
    np.savez(buf, idx=idx, val=val, n=np.int32(len(y)), cutoff=np.float32(cutoff))
    return buf.getvalue()


def _add_bytes(tar: tarfile.TarFile, arcname: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=arcname)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def export_spectra(spectra_dir, out, sparsify_data: bool = True,
                   cutoff: float = 0.001, renormalize: bool = True) -> dict:
    """Pack ``spectra_dir`` into a single ``.tar.gz`` at ``out``.

    Returns a small summary dict (counts and sizes).
    """
    spectra_dir = Path(spectra_dir)
    out = Path(out)
    if out.suffix != ".gz":
        out = out.with_name(out.name + (".tar.gz" if out.suffix != ".tar" else ".gz"))
    tmp_tar = out.with_suffix("")          # strip .gz → the intermediate .tar
    if tmp_tar.suffix != ".tar":
        tmp_tar = tmp_tar.with_suffix(".tar")

    mol_files = sorted(spectra_dir.rglob("mol_*.npy")) + sorted(spectra_dir.rglob("mol_*.npz"))
    if not mol_files:
        raise FileNotFoundError(f"No spectra (mol_*.np[yz]) under {spectra_dir}")
    aux = sorted(spectra_dir.rglob("ppm_axis.npy")) + sorted(spectra_dir.glob("index.csv"))

    manifest = (
        "SpinHance spectra export\n"
        f"dense->sparse on export: {sparsify_data}  cutoff: {cutoff} x max  "
        f"renormalized: {renormalize}\n"
        "Read any spectrum with simulation.spectrum_io.load_spectrum(path):\n"
        "  - mol_<i>.npy  = dense intensity array\n"
        "  - mol_<i>.npz with idx/val/n      = sparse (y=zeros(n); y[idx]=val)\n"
        "  - mol_<i>.npz with centers/amps/.. = peak list (convolved on load)\n"
        "ppm_axis.npy (per field) and index.csv (index -> chembl_id) are plain.\n"
    )

    # ── Stage 1: build uncompressed tar ───────────────────────────────────────
    try:
        with tarfile.open(tmp_tar, "w") as tar:
            _add_bytes(tar, "MANIFEST.txt", manifest.encode())
            for a in aux:
                tar.add(a, arcname=str(a.relative_to(spectra_dir)))
            for f in _bar(mol_files, len(mol_files),
                          "compressing" if sparsify_data else "packing"):
                arc = str(f.relative_to(spectra_dir))
                if f.suffix == ".npz":
                    tar.add(f, arcname=arc)            # already sparse / peaks
                elif sparsify_data:
                    data = _sparse_bytes(np.load(f), cutoff, renormalize)
                    _add_bytes(tar, arc[:-4] + ".npz", data)   # dense .npy → sparse
                else:
                    tar.add(f, arcname=arc)

        # ── Stage 2: gzip the tar with a byte-progress bar ────────────────────
        total = tmp_tar.stat().st_size
        chunk = 1 << 20
        with open(tmp_tar, "rb") as fin, gzip.open(out, "wb") as fout:
            bar = _tqdm(total=total, desc="zipping", unit="B",
                        unit_scale=True) if _tqdm else None
            while True:
                block = fin.read(chunk)
                if not block:
                    break
                fout.write(block)
                if bar:
                    bar.update(len(block))
            if bar:
                bar.close()
    finally:
        if tmp_tar.exists():
            tmp_tar.unlink()

    n_spec = len(mol_files)
    out_mb = out.stat().st_size / 1e6
    print(f"\nExported {n_spec} spectra → {out}  ({out_mb:.1f} MB"
          + (f", sparsified @ {cutoff}×max)" if sparsify_data else ", dense)"))
    return {"spectra": n_spec, "out": str(out), "size_mb": round(out_mb, 2),
            "sparsified": sparsify_data, "cutoff": cutoff}
