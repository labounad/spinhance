"""generate/viewer.py — interactive GUI for triage of candidate molecules.

Gallery view  : 4×4 paginated grid of molecule thumbnails; click to open detail.
Detail view   : 2-D structure with per-atom spin-group labels, spin-group table
                with HARD/SOFT tier indicators, and mouseover highlighting.

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

try:
    from PIL import Image, ImageTk
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    RDLogger.DisableLog("rdApp.*")
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}")

from generate.spin_equivalence import classify_spin_groups, SpinGroup

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

# Tier display strings and colours
_TIER_TEXT  = {"HARD": "(H)", "SOFT": "(S)", "NONE": ""}
_TIER_FG    = {"HARD": "#1a3a8f", "SOFT": "#2a6e1f", "NONE": "#444444"}
_TIER_BG    = {"HARD": "#dde8ff", "SOFT": "#dff2d8", "NONE": "#f5f5f5"}


# ── Molecule rendering ────────────────────────────────────────────────────────

def _ensure_2d(mol: Chem.Mol) -> Chem.Mol:
    if mol.GetNumConformers() == 0:
        AllChem.Compute2DCoords(mol)
    return mol


def _draw_labeled_molecule(
    mol: Chem.Mol,
    groups: list[SpinGroup],
    *,
    highlight_label: Optional[str] = None,
    size: tuple[int, int] = (_MOL_W, _MOL_H),
) -> tuple[Image.Image, dict[int, tuple[float, float]]]:
    """Render mol with spin-group atom notes; optionally highlight one group.

    Returns the PIL image and a dict mapping heavy-atom index → pixel (x, y),
    used for hover-highlight overlays without a full redraw.
    """
    # Work on a copy so we don't mutate the caller's mol
    rw = Chem.RWMol(mol)
    _ensure_2d(rw)

    # Build heavy-atom → sorted label list
    heavy_labels: dict[int, list[str]] = {}
    for g in groups:
        for h_idx in g.heavy_parent_indices:
            heavy_labels.setdefault(h_idx, []).append(g.label)

    for h_idx, labels in heavy_labels.items():
        rw.GetAtomWithIdx(h_idx).SetProp("atomNote", "/".join(labels))

    disp = rw.GetMol()

    # Highlight atoms for the selected group
    hl_atoms: list[int] = []
    hl_colors: dict[int, tuple[float, float, float]] = {}
    if highlight_label is not None:
        grp = next((g for g in groups if g.label == highlight_label), None)
        if grp is not None:
            for h_idx in grp.heavy_parent_indices:
                hl_atoms.append(h_idx)
                if grp.tier == "HARD":
                    hl_colors[h_idx] = (0.13, 0.27, 0.87)   # blue
                elif grp.tier == "SOFT":
                    hl_colors[h_idx] = (0.17, 0.53, 0.13)   # green
                else:
                    hl_colors[h_idx] = (0.55, 0.55, 0.55)   # grey

    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    drawer.drawOptions().addStereoAnnotation  = True
    drawer.drawOptions().addAtomIndices       = False
    drawer.drawOptions().annotationFontScale  = 0.55

    if hl_atoms:
        drawer.DrawMolecule(
            disp,
            highlightAtoms      = hl_atoms,
            highlightAtomColors = hl_colors,
            highlightAtomRadii  = {i: 0.4 for i in hl_atoms},
            highlightBonds      = [],
        )
    else:
        drawer.DrawMolecule(disp)

    drawer.FinishDrawing()

    # Collect heavy-atom pixel positions for overlay highlights
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
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    random.seed(seed)
    return random.sample(rows, min(n, len(rows)))


# ── Detail window ─────────────────────────────────────────────────────────────

class _DetailWindow(tk.Toplevel):
    """Single-molecule detail view: labelled 2-D structure + spin-group table."""

    def __init__(self, parent: tk.Tk, rows: list[dict], idx: int) -> None:
        super().__init__(parent)
        self.resizable(True, True)
        self._rows         = rows
        self._idx          = idx
        self._mol_photo:   Optional[ImageTk.PhotoImage] = None
        self._base_image:  Optional[Image.Image]        = None
        self._atom_coords: dict[int, tuple[float, float]] = {}
        self._groups:      list[SpinGroup]              = []
        self._hovered_label: Optional[str]              = None

        self._build_ui()
        self._refresh()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.title("Detail")

        # ── top: ChEMBL ID ────────────────────────────────────────────────────
        top = ttk.Frame(self, padding=(8, 4))
        top.pack(fill=tk.X)
        self._id_var = tk.StringVar()
        ttk.Label(top, textvariable=self._id_var,
                  font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT)

        # ── main row: canvas | table ──────────────────────────────────────────
        main = ttk.Frame(self, padding=6)
        main.pack(fill=tk.BOTH, expand=True)

        # Molecule canvas
        self._canvas = tk.Canvas(
            main, width=_MOL_W, height=_MOL_H, bg="white", highlightthickness=0
        )
        self._canvas.pack(side=tk.LEFT, padx=(0, 8))

        # Spin-group table
        table_frame = ttk.LabelFrame(main, text="Spin groups", padding=4)
        table_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        cols = ("tier", "count")
        self._tree = ttk.Treeview(
            table_frame, columns=cols, show="tree headings", height=20, selectmode="none"
        )
        self._tree.heading("#0",    text="Group")
        self._tree.heading("tier",  text="Tier")
        self._tree.heading("count", text="H")
        self._tree.column("#0",    width=55,  anchor=tk.CENTER)
        self._tree.column("tier",  width=45,  anchor=tk.CENTER)
        self._tree.column("count", width=35,  anchor=tk.CENTER)

        vsb = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._tree.pack(fill=tk.BOTH, expand=True)

        self._tree.bind("<Motion>", self._on_tree_motion)
        self._tree.bind("<Leave>",  self._on_tree_leave)

        # Configure row colours per tier
        for tier, fg in _TIER_FG.items():
            self._tree.tag_configure(
                tier.lower(),
                foreground=fg,
                background=_TIER_BG[tier],
                font=("TkDefaultFont", 9, "bold" if tier == "HARD" else "normal"),
            )
        self._tree.tag_configure("hover", background="#ffe080")

        # ── SMILES row ────────────────────────────────────────────────────────
        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X)
        smi_frame = ttk.Frame(self, padding=(8, 4))
        smi_frame.pack(fill=tk.X)
        ttk.Label(smi_frame, text="SMILES", font=("TkDefaultFont", 9, "bold")).pack(side=tk.LEFT)
        self._smiles_txt = tk.Text(
            smi_frame, height=2, font=("TkFixedFont", 8),
            wrap=tk.WORD, state=tk.DISABLED, relief=tk.FLAT, bg="#f0f0f0"
        )
        self._smiles_txt.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))

        # ── navigation bar ────────────────────────────────────────────────────
        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X)
        nav = ttk.Frame(self, padding=(6, 4))
        nav.pack(fill=tk.X)
        ttk.Button(nav, text="< Prev", command=self._prev).pack(side=tk.LEFT)
        self._counter_var = tk.StringVar()
        ttk.Label(nav, textvariable=self._counter_var, width=12,
                  anchor=tk.CENTER).pack(side=tk.LEFT, padx=6)
        ttk.Button(nav, text="Next >", command=self._next).pack(side=tk.LEFT)

    # ── Data refresh ──────────────────────────────────────────────────────────

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

        self._groups      = []
        self._hovered_label = None

        if mol is None:
            return

        # Classify spin groups (embed + deuterium test)
        _, groups = classify_spin_groups(mol)
        self._groups = groups

        # Draw base molecule and cache
        self._base_image, self._atom_coords = _draw_labeled_molecule(mol, groups)
        self._show_image(self._base_image)

        # Populate table
        self._tree.delete(*self._tree.get_children())
        for g in groups:
            tier_text = _TIER_TEXT[g.tier]
            self._tree.insert(
                "", "end",
                text=g.label,
                values=(tier_text, len(g.h_indices)),
                tags=(g.tier.lower(),),
            )

    def _show_image(self, img: Image.Image) -> None:
        self._mol_photo = ImageTk.PhotoImage(img)
        self._canvas.config(width=img.width, height=img.height)
        self._canvas.create_image(0, 0, anchor=tk.NW, image=self._mol_photo)

    # ── Navigation ────────────────────────────────────────────────────────────

    def _prev(self) -> None:
        self._idx = (self._idx - 1) % len(self._rows)
        self._refresh()

    def _next(self) -> None:
        self._idx = (self._idx + 1) % len(self._rows)
        self._refresh()

    # ── Table mouseover ───────────────────────────────────────────────────────

    def _on_tree_motion(self, event: tk.Event) -> None:
        item = self._tree.identify_row(event.y)
        label = self._tree.item(item, "text") if item else None

        if label == self._hovered_label:
            return
        self._hovered_label = label

        # Clear old hover tag
        for iid in self._tree.get_children():
            tags = [t for t in self._tree.item(iid, "tags") if t != "hover"]
            self._tree.item(iid, tags=tags)

        if item:
            # Add hover tag to the selected row
            existing = list(self._tree.item(item, "tags"))
            self._tree.item(item, tags=existing + ["hover"])

        # Redraw molecule with or without highlight
        if self._base_image is None:
            return
        mol = Chem.MolFromSmiles(self._rows[self._idx]["smiles"])
        if mol is None:
            return
        img, _ = _draw_labeled_molecule(
            mol, self._groups, highlight_label=label
        )
        self._show_image(img)

    def _on_tree_leave(self, event: tk.Event) -> None:
        self._hovered_label = None
        for iid in self._tree.get_children():
            tags = [t for t in self._tree.item(iid, "tags") if t != "hover"]
            self._tree.item(iid, tags=tags)
        if self._base_image is not None:
            self._show_image(self._base_image)


# ── Gallery view ──────────────────────────────────────────────────────────────

class _GalleryView(tk.Tk):
    """Root window: paginated 4×4 grid of molecule thumbnails."""

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
        self._photos:   list              = []
        self._canvases: list[tk.Canvas]   = []

        self._build_ui()
        self._file_lbl.config(text=source_name)
        self._draw_page()

    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=(6, 4))
        top.pack(fill=tk.X)
        self._file_lbl = ttk.Label(top, text="", font=("TkDefaultFont", 9, "italic"))
        self._file_lbl.pack(side=tk.LEFT)
        ttk.Button(top, text="Open file...", command=self._open_file).pack(side=tk.RIGHT)

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

            thumb = Draw.MolToImage(_ensure_2d(mol), size=(_THUMB_W, _THUMB_H))
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
    """Launch the gallery viewer for *path*."""
    rows = load_sample(path, n, seed)
    _GalleryView(rows, path.name, n, seed).mainloop()
