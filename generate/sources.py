"""generate/sources.py — pluggable compound-source readers.

The screening pipeline only ever needs a stream of ``(id, SMILES)`` records;
every supported database differs solely in how those fields are laid out on
disk.  This module isolates that format knowledge so :mod:`generate.pipeline`
stays source-agnostic and can screen ChEMBL, PubChem, or ZINC unchanged.

Every reader

* opens ``.gz`` files transparently,
* yields ``(source_id, smiles, inchikey)`` tuples, where *inchikey* is ``""``
  when the source does not ship one — the pipeline computes it for the (few)
  molecules that pass, so we never carry InChIKeys for the whole database,
* skips blank lines and the source's header row.

Supported sources
-----------------
``chembl``
    ChEMBL ``chembl_XX_chemreps.txt`` — tab-separated, header row, columns
    ``chembl_id, canonical_smiles, standard_inchi, standard_inchi_key``.
    InChIKey is read directly (column 3).

``pubchem``
    PubChem ``CID-SMILES.gz`` — tab-separated ``<CID>\\t<SMILES>``, no header,
    ~120 M rows.  Download from
    ``https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz``.
    ``source_id`` is ``"CID<n>"``; InChIKey computed downstream.

``zinc``
    ZINC ``.smi`` tranche files — whitespace-separated ``<SMILES> <ZINC_id>``,
    optional ``smiles zinc_id`` header.  ``source_id`` is the ZINC id.

``smiles``
    Generic fallback — whitespace-separated, first token SMILES, optional
    second token id (else ``mol_<line>``).
"""

from __future__ import annotations

import gzip
from collections.abc import Iterator
from pathlib import Path


def _open_text(path: Path):
    """Open *path* as UTF-8 text, transparently decompressing ``.gz``."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def _iter_chembl(path: Path) -> Iterator[tuple[str, str, str]]:
    with _open_text(path) as f:
        next(f, None)  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            yield parts[0], parts[1], parts[3]


def _iter_pubchem(path: Path) -> Iterator[tuple[str, str, str]]:
    with _open_text(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            cid, smiles = parts[0], parts[1]
            yield f"CID{cid}", smiles, ""


def _iter_zinc(path: Path) -> Iterator[tuple[str, str, str]]:
    with _open_text(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            smiles, zinc_id = parts[0], parts[1]
            if smiles.lower() == "smiles":  # header row
                continue
            yield zinc_id, smiles, ""


def _iter_smiles(path: Path) -> Iterator[tuple[str, str, str]]:
    with _open_text(path) as f:
        for i, line in enumerate(f):
            parts = line.split()
            if not parts:
                continue
            if i == 0 and parts[0].lower() == "smiles":  # header row
                continue
            smiles = parts[0]
            sid = parts[1] if len(parts) > 1 else f"mol_{i}"
            yield sid, smiles, ""


_READERS = {
    "chembl":  _iter_chembl,
    "pubchem": _iter_pubchem,
    "zinc":    _iter_zinc,
    "smiles":  _iter_smiles,
}

#: Sources that ship a header line a cheap byte-level line count must discount.
HAS_HEADER = {"chembl"}

SOURCES = tuple(_READERS)


def iter_compounds(
    path: Path, source: str = "chembl"
) -> Iterator[tuple[str, str, str]]:
    """Yield ``(source_id, smiles, inchikey)`` records from *path*.

    Parameters
    ----------
    path:
        Compound file for *source* (``.gz`` handled transparently).
    source:
        One of :data:`SOURCES`.

    Raises
    ------
    ValueError
        If *source* is not recognised.
    """
    try:
        reader = _READERS[source]
    except KeyError:
        raise ValueError(
            f"unknown source {source!r}; choose from {sorted(_READERS)}"
        ) from None
    yield from reader(path)
