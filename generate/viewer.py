"""generate/viewer.py — interactive GUI for triage of candidate molecules.

Gallery view  : 4×4 paginated grid of molecule thumbnails; click to open detail.
                Top bar includes a SMILES entry for direct molecule inspection.
Detail view   : 2-D structure with per-atom spin-group labels, spin-group table
                with HARD/SOFT tier indicators and per-class colour coding,
                and mouseover highlighting.

Usage
-----
::

    python generate/cli.py view
    python generate/cli.py view --file generate/data/candidates_final.csv --n 200
"""

from __future__ import annotations

import csv
import io
import random
import sys
from pathlib import Path
from tkinter import filedialog
import tkinter as tk
from tkinter import ttk
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# RDKit and spin_equivalence are NOT imported at module level — doing so would
# make `from generate.viewer import launch` block for 30-120 s on WSL2 while
# Windows Defender scans the .so files.  They are imported inside launch()
# and stored as module globals so all functions/classes can reference them
# by name at call time (Python resolves globals at call time, not definition time).

Chem            = None
RDLogger        = None
AllChem         = None
Draw            = None
rdMolDraw2D     = None
Image           = None
ImageTk         = None
classify_spin_groups = None
SpinGroup       = None

_REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Layout ────────────────────────────────────────────────────────────────────
_PAGE_SIZE  = 16
_GRID_COLS  = 4
_THUMB_W    = 215
_THUMB_H    = 170
_LABEL_H    = 20
_CELL_H     = _THUMB_H + _LABEL_H
_MOL_W      = 560
_MOL_H      = 430

# ── Tier display ──────────────────────────────────────────────────────────────
_TIER_TEXT  = {"HARD": "(H)", "SOFT": "(S)", "NONE": ""}
_TIER_FG    = {"HARD": "#1a3a8f", "SOFT": "#2a5a20", "NONE": "#444444"}

# Molecule highlight colours for HARD and NONE
_HARD_MOL_COLOR = (0.55, 0.72, 0.97)
_NONE_MOL_COLOR = (0.87, 0.87, 0.87)

# Palette for SOFT equivalence classes — each unique class shares a colour so
# it's immediately obvious which protons have the same chemical shift.
# Each entry: (tkinter_hex_bg, rdkit_mol_rgb)
_SOFT_PALETTE: list[tuple[str, tuple[float, float, float]]] = [
    ("#ffe0b2", (1.00, 0.82, 0.60)),   # orange
    ("#b3e5fc", (0.62, 0.87, 0.97)),   # sky blue
    ("#c8e6c9", (0.72, 0.88, 0.72)),   # green
    ("#f8bbd0", (0.97, 0.73, 0.82)),   # pink
    ("#e1bee7", (0.82, 0.72, 0.90)),   # lavender
    ("#fff9c4", (1.00, 0.96, 0.72)),   # yellow
    ("#b2dfdb", (0.67, 0.85, 0.83)),   # teal
    ("#ffccbc", (1.00, 0.78, 0.72)),   # salmon
]


# ── Soft colour helpers ───────────────────────────────────────────────────────

def _build_soft_color_map(
    groups: list[SpinGroup],
) -> dict[tuple[int, ...], tuple[str, tuple[float, float, float]]]:
    """Map each unique SOFT equivalence class to a (tk_hex, mol_rgb) colour pair."""
    seen: dict[tuple[int, ...], int] = {}
    for g in groups:
        if g.tier == "SOFT" and g.class_h_indices not in seen:
            seen[g.class_h_indices] = len(seen) % len(_SOFT_PALETTE)
    return {cls: _SOFT_PALETTE[idx] for cls, idx in seen.items()}


# ── Molecule rendering ────────────────────────────────────────────────────────

def _ensure_2d(mol: Chem.Mol) -> Chem.Mol:
    if mol.GetNumConformers() == 0:
        AllChem.Compute2DCoords(mol)
    return mol


def _draw_labeled_molecule(
    mol: Chem.Mol,
    groups: list[SpinGroup],
    *,
    soft_color_map: dict[tuple[int, ...], tuple[str, tuple[float, float, float]]],
    highlight_label: Optional[str] = None,
    size: tuple[int, int] = (_MOL_W, _MOL_H),
) -> tuple[Image.Image, dict[int, tuple[float, float]]]:
    """Render mol with spin-group atom notes and per-class colour highlights.

    * HARD  → steel blue
    * SOFT  → colour shared by all protons in the same equivalence class
    * NONE  → light grey
    * Hovered group → boosted; others dimmed
    """
    rw = Chem.RWMol(mol)
    _ensure_2d(rw)

    heavy_labels: dict[int, list[str]] = {}
    for g in groups:
        for h_idx in g.heavy_parent_indices:
            heavy_labels.setdefault(h_idx, []).append(g.label)
    for h_idx, labels in heavy_labels.items():
        rw.GetAtomWithIdx(h_idx).SetProp("atomNote", "/".join(labels))

    disp    = rw.GetMol()
    hl_atoms:  list[int]                              = []
    hl_colors: dict[int, tuple[float, float, float]]  = {}
    hl_radii:  dict[int, float]                       = {}
    assigned:  set[int]                               = set()

    for g in groups:
        is_hovered = (highlight_label == g.label)

        if g.tier == "HARD":
            base = _HARD_MOL_COLOR
        elif g.tier == "SOFT":
            base = soft_color_map.get(g.class_h_indices, _SOFT_PALETTE[0])[1]
        else:
            base = _NONE_MOL_COLOR

        if highlight_label is not None:
            if is_hovered:
                color  = tuple(max(0.0, c - 0.25) for c in base)
                radius = 0.45
            else:
                color  = tuple(min(1.0, c + (1.0 - c) * 0.5) for c in base)
                radius = 0.28
        else:
            color  = base
            radius = 0.38

        for h_idx in g.heavy_parent_indices:
            if h_idx in assigned:
                continue
            hl_atoms.append(h_idx)
            hl_colors[h_idx] = color   # type: ignore[assignment]
            hl_radii[h_idx]  = radius
            assigned.add(h_idx)

    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    drawer.drawOptions().addStereoAnnotation = True
    drawer.drawOptions().addAtomIndices      = False
    drawer.drawOptions().annotationFontScale = 0.55

    drawer.DrawMolecule(
        disp,
        highlightAtoms      = hl_atoms,
        highlightAtomColors = hl_colors,
        highlightAtomRadii  = hl_radii,
        highlightBonds      = [],
    )
    drawer.FinishDrawing()

    coords: dict[int, tuple[float, float]] = {}
    for atom in disp.GetAtoms():
        if atom.GetAtomicNum() != 1:
            try:
                pt = drawer.GetDrawCoords(atom.GetIdx())
                coords[atom.GetIdx()] = (pt.x, pt.y)
            except Exception:
                pass

    img = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")
    return img, coords


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_sample(path: Path, n: int, seed: int) -> list[dict]:
    csv.field_size_limit(10 * 1024 * 1024)   # handle large ChEMBL SMILES strings
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    random.seed(seed)
    return random.sample(rows, min(n, len(rows)))


def _smiles_row(smiles: str) -> dict:
    return {"smiles": smiles, "chembl_id": "manual", "inchikey": ""}


# ── Detail window ─────────────────────────────────────────────────────────────

class _DetailWindow(tk.Toplevel):
    """Single-molecule detail view: labelled 2-D structure + spin-group table."""

    def __init__(self, parent: tk.Tk, rows: list[dict], idx: int) -> None:
        super().__init__(parent)
        self.resizable(True, True)
        self._rows            = rows
        self._idx             = idx
        self._mol_photo:      Optional[ImageTk.PhotoImage] = None
        self._base_image:     Optional[Image.Image]        = None
        self._atom_coords:    dict[int, tuple[float, float]] = {}
        self._groups:         list[SpinGroup]              = []
        self._soft_color_map: dict                         = {}
        self._hovered_label:  Optional[str]               = None
        self._tree_tags:      dict[str, str]               = {}

        self._build_ui()
        self.after(0, self._refresh)   # defer so window appears immediately

    def _build_ui(self) -> None:
        self.title("Detail")

        top = ttk.Frame(self, padding=(8, 4))
        top.pack(fill=tk.X)
        self._id_var = tk.StringVar()
        ttk.Label(top, textvariable=self._id_var,
                  font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT)

        main = ttk.Frame(self, padding=6)
        main.pack(fill=tk.BOTH, expand=True)

        self._canvas = tk.Canvas(
            main, width=_MOL_W, height=_MOL_H, bg="white", highlightthickness=0
        )
        self._canvas.pack(side=tk.LEFT, padx=(0, 8))

        table_frame = ttk.LabelFrame(main, text="Spin groups", padding=4)
        table_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._tree = ttk.Treeview(
            table_frame, columns=("tier", "count"),
            show="tree headings", height=20, selectmode="none",
        )
        self._tree.heading("#0",    text="Group")
        self._tree.heading("tier",  text="Tier")
        self._tree.heading("count", text="H")
        self._tree.column("#0",    width=55, anchor=tk.CENTER)
        self._tree.column("tier",  width=45, anchor=tk.CENTER)
        self._tree.column("count", width=35, anchor=tk.CENTER)

        vsb = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._tree.pack(fill=tk.BOTH, expand=True)

        self._tree.bind("<Motion>", self._on_tree_motion)
        self._tree.bind("<Leave>",  self._on_tree_leave)

        # Static tier tags (HARD and NONE are fixed colours)
        self._tree.tag_configure(
            "hard", foreground=_TIER_FG["HARD"],
            background="#dde8ff", font=("TkDefaultFont", 9, "bold"),
        )
        self._tree.tag_configure(
            "none_tier", foreground=_TIER_FG["NONE"], background="#f5f5f5",
        )
        self._tree.tag_configure("hover", background="#ffe080")

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X)
        smi_frame = ttk.Frame(self, padding=(8, 4))
        smi_frame.pack(fill=tk.X)
        ttk.Label(smi_frame, text="SMILES",
                  font=("TkDefaultFont", 9, "bold")).pack(side=tk.LEFT)
        self._smiles_txt = tk.Text(
            smi_frame, height=2, font=("TkFixedFont", 8),
            wrap=tk.WORD, state=tk.DISABLED, relief=tk.FLAT, bg="#f0f0f0",
        )
        self._smiles_txt.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X)
        nav = ttk.Frame(self, padding=(6, 4))
        nav.pack(fill=tk.X)
        ttk.Button(nav, text="< Prev", command=self._prev).pack(side=tk.LEFT)
        self._counter_var = tk.StringVar()
        ttk.Label(nav, textvariable=self._counter_var, width=12,
                  anchor=tk.CENTER).pack(side=tk.LEFT, padx=6)
        ttk.Button(nav, text="Next >", command=self._next).pack(side=tk.LEFT)

    def _refresh(self) -> None:
        row = self._rows[self._idx]
        mol = Chem.MolFromSmiles(row["smiles"])

        self._id_var.set(row.get("chembl_id", "—"))
        self._counter_var.set(f"{self._idx + 1} / {len(self._rows)}")
        self.title(f"Detail  —  {row.get('chembl_id', '')}")

        self._smiles_txt.config(state=tk.NORMAL)
        self._smiles_txt.delete("1.0", tk.END)
        self._smiles_txt.insert("1.0", row["smiles"])
        self._smiles_txt.config(state=tk.DISABLED)

        self._groups         = []
        self._soft_color_map = {}
        self._hovered_label  = None
        self._tree_tags      = {}

        if mol is None:
            return

        _, groups = classify_spin_groups(mol)
        self._groups         = groups
        self._soft_color_map = _build_soft_color_map(groups)

        self._base_image, self._atom_coords = _draw_labeled_molecule(
            mol, groups, soft_color_map=self._soft_color_map,
        )
        self._show_image(self._base_image)

        # Populate table — one tag per unique SOFT class
        self._tree.delete(*self._tree.get_children())
        seen_classes: dict[tuple[int, ...], str] = {}
        for g in groups:
            if g.tier == "SOFT" and g.class_h_indices not in seen_classes:
                tag = f"soft_{len(seen_classes)}"
                tk_color, _ = self._soft_color_map.get(
                    g.class_h_indices, ("#e8f5e9", (0.7, 0.95, 0.7))
                )
                self._tree.tag_configure(
                    tag, background=tk_color, foreground=_TIER_FG["SOFT"],
                )
                seen_classes[g.class_h_indices] = tag

        for g in groups:
            if g.tier == "HARD":
                tag = "hard"
            elif g.tier == "SOFT":
                tag = seen_classes[g.class_h_indices]
            else:
                tag = "none_tier"
            self._tree_tags[g.label] = tag
            self._tree.insert(
                "", "end",
                text=g.label,
                values=(_TIER_TEXT[g.tier], len(g.h_indices)),
                tags=(tag,),
            )

    def _show_image(self, img: Image.Image) -> None:
        self._mol_photo = ImageTk.PhotoImage(img)
        self._canvas.config(width=img.width, height=img.height)
        self._canvas.create_image(0, 0, anchor=tk.NW, image=self._mol_photo)

    def _prev(self) -> None:
        self._idx = (self._idx - 1) % len(self._rows)
        self._refresh()

    def _next(self) -> None:
        self._idx = (self._idx + 1) % len(self._rows)
        self._refresh()

    def _on_tree_motion(self, event: tk.Event) -> None:
        item  = self._tree.identify_row(event.y)
        label = self._tree.item(item, "text") if item else None

        if label == self._hovered_label:
            return
        self._hovered_label = label

        for iid in self._tree.get_children():
            lbl  = self._tree.item(iid, "text")
            base = self._tree_tags.get(lbl, "none_tier")
            self._tree.item(iid, tags=(base,))
        if item:
            base = self._tree_tags.get(label, "none_tier")
            self._tree.item(item, tags=(base, "hover"))

        if self._base_image is None:
            return
        mol = Chem.MolFromSmiles(self._rows[self._idx]["smiles"])
        if mol is None:
            return
        img, _ = _draw_labeled_molecule(
            mol, self._groups,
            soft_color_map=self._soft_color_map,
            highlight_label=label,
        )
        self._show_image(img)

    def _on_tree_leave(self, event: tk.Event) -> None:
        self._hovered_label = None
        for iid in self._tree.get_children():
            lbl  = self._tree.item(iid, "text")
            base = self._tree_tags.get(lbl, "none_tier")
            self._tree.item(iid, tags=(base,))
        if self._base_image is not None:
            self._show_image(self._base_image)


# ── Gallery view ──────────────────────────────────────────────────────────────

class _GalleryView(tk.Tk):
    """Root window: file browser, SMILES entry, and 4×4 molecule grid."""

    def __init__(self, rows: list[dict], source_name: str, n: int, seed: int) -> None:
        super().__init__()
        self.title("SpinHance Candidate Viewer")
        self.resizable(False, False)
        ttk.Style(self).theme_use("clam")

        self._rows    = rows
        self._n       = n
        self._seed    = seed
        self._page    = 0
        self._n_pages = max(1, (len(rows) + _PAGE_SIZE - 1) // _PAGE_SIZE)
        self._photos:   list            = []
        self._canvases: list[tk.Canvas] = []

        self._build_ui()
        self._file_lbl.config(text=source_name)
        self.after(0, self._draw_page)   # defer so window appears before rendering

    def _build_ui(self) -> None:
        # Row 1: file label + open button
        top = ttk.Frame(self, padding=(6, 4))
        top.pack(fill=tk.X)
        self._file_lbl = ttk.Label(top, text="", font=("TkDefaultFont", 9, "italic"))
        self._file_lbl.pack(side=tk.LEFT)
        ttk.Button(top, text="Open file...", command=self._open_file).pack(side=tk.RIGHT)

        # Row 2: SMILES entry for direct lookup
        smi_bar = ttk.Frame(self, padding=(6, 2))
        smi_bar.pack(fill=tk.X)
        ttk.Label(smi_bar, text="SMILES:", font=("TkDefaultFont", 9)).pack(side=tk.LEFT)
        self._smiles_var = tk.StringVar()
        smi_entry = ttk.Entry(smi_bar, textvariable=self._smiles_var, width=58)
        smi_entry.pack(side=tk.LEFT, padx=(4, 4), fill=tk.X, expand=True)
        smi_entry.bind("<Return>", self._view_smiles)
        ttk.Button(smi_bar, text="View", command=self._view_smiles, width=6).pack(side=tk.LEFT)
        self._smiles_err = ttk.Label(smi_bar, text="", foreground="red",
                                      font=("TkDefaultFont", 8))
        self._smiles_err.pack(side=tk.LEFT, padx=(4, 0))

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X)

        grid_frame = ttk.Frame(self, padding=6)
        grid_frame.pack(fill=tk.BOTH, expand=True)

        for i in range(_PAGE_SIZE):
            r, c = divmod(i, _GRID_COLS)
            frame = ttk.Frame(grid_frame, relief=tk.GROOVE, borderwidth=1)
            frame.grid(row=r, column=c, padx=3, pady=3)
            cv = tk.Canvas(frame, width=_THUMB_W, height=_CELL_H,
                           bg="white", highlightthickness=0, cursor="hand2")
            cv.pack()
            self._canvases.append(cv)

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X)
        nav = ttk.Frame(self, padding=(6, 4))
        nav.pack(fill=tk.X)
        ttk.Button(nav, text="< Prev", command=self._prev_page).pack(side=tk.LEFT)
        self._page_var = tk.StringVar()
        ttk.Label(nav, textvariable=self._page_var, width=24,
                  anchor=tk.CENTER).pack(side=tk.LEFT, padx=6)
        ttk.Button(nav, text="Next >", command=self._next_page).pack(side=tk.LEFT)

    def _view_smiles(self, event: object = None) -> None:
        smi = self._smiles_var.get().strip()
        if not smi:
            return
        if Chem.MolFromSmiles(smi) is None:
            self._smiles_err.config(text="Invalid SMILES")
            return
        self._smiles_err.config(text="")
        _DetailWindow(self, [_smiles_row(smi)], 0)

    def _draw_page(self) -> None:
        new_photos: list = []
        start     = self._page * _PAGE_SIZE
        page_rows = self._rows[start : start + _PAGE_SIZE]

        for i, cv in enumerate(self._canvases):
            cv.unbind("<Button-1>")
            cv.unbind("<Enter>")
            cv.unbind("<Leave>")
            cv.delete("all")
            cv.config(bg="white")

            if i >= len(page_rows):
                continue

            row = page_rows[i]
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is None:
                continue

            try:
                thumb = Draw.MolToImage(_ensure_2d(mol), size=(_THUMB_W, _THUMB_H))
            except Exception:
                continue

            photo = ImageTk.PhotoImage(thumb)
            new_photos.append(photo)

            cv.create_image(0, 0, anchor=tk.NW, image=photo)
            cv.create_rectangle(0, _THUMB_H, _THUMB_W, _CELL_H,
                                 fill="#dde3ed", outline="")
            cv.create_text(_THUMB_W // 2, _THUMB_H + _LABEL_H // 2,
                           text=row.get("chembl_id", ""),
                           font=("TkDefaultFont", 8), fill="#222222")

            abs_idx = start + i
            cv.bind("<Button-1>", lambda e, idx=abs_idx: _DetailWindow(self, self._rows, idx))
            cv.bind("<Enter>",    lambda e, c=cv: c.config(bg="#cfe0f8"))
            cv.bind("<Leave>",    lambda e, c=cv: c.config(bg="white"))

        self._photos = new_photos
        self._page_var.set(
            f"Page {self._page + 1} / {self._n_pages}  ({len(self._rows)} molecules)"
        )

    def _prev_page(self) -> None:
        if self._page > 0:
            self._page -= 1
            self._draw_page()

    def _next_page(self) -> None:
        if self._page < self._n_pages - 1:
            self._page += 1
            self._draw_page()

    def _open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Open candidate CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=str(_REPO_ROOT / "generate" / "data"),
        )
        if not path:
            return
        p = Path(path)
        self._rows    = load_sample(p, self._n, self._seed)
        self._n_pages = max(1, (len(self._rows) + _PAGE_SIZE - 1) // _PAGE_SIZE)
        self._page    = 0
        self._file_lbl.config(text=p.name)
        self._draw_page()


# ── Public entry point ────────────────────────────────────────────────────────

def launch(path: Path, *, n: int = 80, seed: int = 42) -> None:
    """Launch the gallery viewer for *path*.

    All RDKit imports happen here so that ``import viewer`` is instant.
    """
    global Chem, RDLogger, AllChem, Draw, rdMolDraw2D
    global Image, ImageTk, classify_spin_groups, SpinGroup

    try:
        from PIL import Image as _Img, ImageTk as _ImgTk
        from rdkit import Chem as _Chem, RDLogger as _RL
        from rdkit.Chem import AllChem as _AllChem, Draw as _Draw
        from rdkit.Chem.Draw import rdMolDraw2D as _rmd
        from generate.spin_equivalence import (
            classify_spin_groups as _csg, SpinGroup as _SG,
        )
    except ImportError as exc:
        sys.exit(f"Missing dependency: {exc}")

    Chem             = _Chem
    RDLogger         = _RL
    AllChem          = _AllChem
    Draw             = _Draw
    rdMolDraw2D      = _rmd
    Image            = _Img
    ImageTk          = _ImgTk
    classify_spin_groups = _csg
    SpinGroup        = _SG
    RDLogger.DisableLog("rdApp.*")

    rows = load_sample(path, n, seed)
    _GalleryView(rows, path.name, n, seed).mainloop()
