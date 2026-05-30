"""generate/viewer.py — single-window interactive triage viewer.

Layout
------
A horizontal split puts the 4×4 molecule gallery on the left and the detail
panel on the right.  Clicking any thumbnail updates the detail panel in place —
no secondary windows.

Features
--------
* ChemDraw-style molecule rendering: 2× super-sampled with LANCZOS downscale,
  thin bonds, clean atom labels.
* Per-class SOFT colour coding so equivalence pairs are visually obvious.
* Mouseover the spin-group table → atom(s) highlighted on the structure.
* Mouseover the structure → nearest atom's row highlighted in the table.
* Fixed canvas size — hovering never causes a layout shift.
* SMILES entry bar at the top for direct ad-hoc molecule inspection.

Usage
-----
::

    python generate/cli.py view
    python generate/cli.py view --file generate/data/8spin.csv
"""

from __future__ import annotations

import csv
import io
import math
import random
import sys
from pathlib import Path
from tkinter import filedialog
import tkinter as tk
from tkinter import ttk
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# RDKit/PIL globals — populated lazily inside launch() so that
# `import viewer` is instant (avoids 30-120 s WSL2 .so scan on import).
Chem             = None
RDLogger         = None
AllChem          = None
Draw             = None
rdMolDraw2D      = None
Image            = None
ImageTk          = None
ImageDraw        = None
ImageFont        = None
classify_spin_groups = None
SpinGroup        = None

_REPO_ROOT = Path(__file__).resolve().parent.parent


# ── Dimensions ────────────────────────────────────────────────────────────────

_THUMB_W     = 175
_THUMB_H     = 145
_THUMB_LBL   = 18           # label strip below thumbnail
_CELL_H      = _THUMB_H + _THUMB_LBL
_GRID_COLS   = 4
_PAGE_SIZE   = 16

_MOL_W       = 520          # molecule canvas — NEVER changes after init
_MOL_H       = 420
_TABLE_W     = 250          # spin-group table width

_RENDER_SCALE = 2           # super-sample factor for ChemDraw quality


# ── Fonts & colours ───────────────────────────────────────────────────────────

import platform as _platform
_FF = "Helvetica Neue" if _platform.system() == "Darwin" else (
      "Segoe UI"        if _platform.system() == "Windows" else "Helvetica")

_BG          = "#f4f5f7"    # window background
_PANEL_BG    = "#ffffff"    # panel / card background
_BORDER      = "#d0d4db"    # separator / border colour
_ACCENT      = "#3a7bd5"    # primary accent (blue)
_ROW_ODD     = "#fafbfc"
_ROW_EVEN    = "#f0f2f5"
_HOVER_ROW   = "#fff3b0"    # warm yellow for table hover

_TIER_FG   = {"HARD": "#1a3a8f", "SOFT": "#2a5a20", "NONE": "#555555"}
_TIER_TEXT = {"HARD": "(H)", "SOFT": "(S)", "NONE": "—"}

_HARD_MOL  = (0.22, 0.48, 0.84)   # steel blue
_NONE_MOL  = (0.70, 0.72, 0.76)   # mid grey

_SOFT_PALETTE: list[tuple[str, tuple[float, float, float]]] = [
    ("#ffe0b2", (1.00, 0.80, 0.56)),
    ("#b3e5fc", (0.56, 0.84, 0.96)),
    ("#c8e6c9", (0.68, 0.88, 0.68)),
    ("#f8bbd0", (0.96, 0.70, 0.80)),
    ("#e1bee7", (0.80, 0.70, 0.90)),
    ("#fff9c4", (1.00, 0.96, 0.68)),
    ("#b2dfdb", (0.64, 0.84, 0.82)),
    ("#ffccbc", (1.00, 0.76, 0.68)),
]


# ── Style helpers ─────────────────────────────────────────────────────────────

def _configure_styles(root: tk.Tk) -> None:
    s = ttk.Style(root)
    s.theme_use("clam")
    base = {"font": (_FF, 10)}
    s.configure(".",          **base, background=_BG, foreground="#1a1a2e")
    s.configure("TFrame",     background=_BG)
    s.configure("TLabel",     background=_BG, foreground="#1a1a2e")
    s.configure("TButton",    font=(_FF, 10), padding=(10, 4))
    s.configure("TEntry",     font=(_FF, 10), fieldbackground="white",
                relief="flat")
    s.configure("TSeparator", background=_BORDER)
    s.configure("TLabelframe",       background=_PANEL_BG, relief="flat",
                borderwidth=1)
    s.configure("TLabelframe.Label", background=_PANEL_BG,
                font=(_FF, 9, "bold"), foreground="#555555")
    s.configure("Treeview",   font=(_FF, 9), rowheight=22,
                background=_PANEL_BG, fieldbackground=_PANEL_BG,
                borderwidth=0)
    s.configure("Treeview.Heading", font=(_FF, 9, "bold"),
                background=_BG, foreground="#555555", relief="flat")
    s.map("Treeview", background=[("selected", _HOVER_ROW)],
          foreground=[("selected", "#1a1a2e")])
    s.configure("Card.TFrame",  background=_PANEL_BG, relief="flat")
    s.configure("ID.TLabel",    background=_PANEL_BG,
                font=(_FF, 13, "bold"), foreground="#1a1a2e")
    s.configure("Sub.TLabel",   background=_PANEL_BG,
                font=(_FF, 9), foreground="#888888")
    s.configure("Nav.TButton",  font=(_FF, 10), padding=(14, 5))
    s.configure("Accent.TButton", font=(_FF, 10, "bold"), padding=(10, 4))


# ── Soft colour map ───────────────────────────────────────────────────────────

def _build_soft_color_map(groups):
    seen: dict[tuple, int] = {}
    for g in groups:
        if g.tier == "SOFT" and g.class_h_indices not in seen:
            seen[g.class_h_indices] = len(seen) % len(_SOFT_PALETTE)
    return {cls: _SOFT_PALETTE[i] for cls, i in seen.items()}


# ── ChemDraw-quality molecule rendering ──────────────────────────────────────

def _ensure_2d(mol):
    if mol.GetNumConformers() == 0:
        AllChem.Compute2DCoords(mol)
    return mol


def _draw_molecule(
    mol,
    groups,
    *,
    soft_color_map: dict,
    highlight_label: Optional[str] = None,
    size: tuple[int, int] = (_MOL_W, _MOL_H),
) -> tuple[object, dict[int, tuple[float, float]]]:
    """Render mol at 2× and downscale with LANCZOS for crisp, ChemDraw-quality output.

    Returns (PIL Image, atom_coords_1x) where atom_coords maps heavy-atom index
    to (x, y) pixel position in the *displayed* (1×) image for hit-testing.
    """
    W, H  = size
    S     = _RENDER_SCALE
    rw    = Chem.RWMol(mol)
    _ensure_2d(rw)

    # Atom notes
    heavy_labels: dict[int, list[str]] = {}
    for g in groups:
        for hi in g.heavy_parent_indices:
            heavy_labels.setdefault(hi, []).append(g.label)
    for hi, lbls in heavy_labels.items():
        rw.GetAtomWithIdx(hi).SetProp("atomNote", "/".join(lbls))

    disp = rw.GetMol()

    # Build highlight dicts
    hl_atoms:  list[int]                              = []
    hl_colors: dict[int, tuple[float, float, float]]  = {}
    hl_radii:  dict[int, float]                       = {}
    assigned:  set[int]                               = set()

    for g in groups:
        hovered = (highlight_label == g.label)
        if g.tier == "HARD":
            base = _HARD_MOL
        elif g.tier == "SOFT":
            base = soft_color_map.get(g.class_h_indices, _SOFT_PALETTE[0])[1]
        else:
            base = _NONE_MOL

        if highlight_label is not None:
            if hovered:
                col    = tuple(max(0.0, c - 0.22) for c in base)
                radius = 0.46
            else:
                col    = tuple(min(1.0, c + (1.0 - c) * 0.55) for c in base)
                radius = 0.28
        else:
            col    = base
            radius = 0.38

        for hi in g.heavy_parent_indices:
            if hi in assigned:
                continue
            hl_atoms.append(hi)
            hl_colors[hi] = col  # type: ignore[assignment]
            hl_radii[hi]  = radius
            assigned.add(hi)

    # Super-sampled drawer
    drawer = rdMolDraw2D.MolDraw2DCairo(W * S, H * S)
    opts   = drawer.drawOptions()
    opts.bondLineWidth        = 1.8
    opts.multipleBondOffset   = 0.20
    opts.padding              = 0.08
    opts.addAtomIndices       = False
    opts.addStereoAnnotation  = True
    opts.annotationFontScale  = 0.60

    drawer.DrawMolecule(
        disp,
        highlightAtoms      = hl_atoms,
        highlightAtomColors = hl_colors,
        highlightAtomRadii  = hl_radii,
        highlightBonds      = [],
    )
    drawer.FinishDrawing()

    # Collect atom coords at 2× then scale to 1×
    coords: dict[int, tuple[float, float]] = {}
    for atom in disp.GetAtoms():
        if atom.GetAtomicNum() != 1:
            try:
                pt = drawer.GetDrawCoords(atom.GetIdx())
                coords[atom.GetIdx()] = (pt.x / S, pt.y / S)
            except Exception:
                pass

    hires = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")
    img   = hires.resize(size, Image.LANCZOS)
    return img, coords


def _draw_thumbnail(mol, size: tuple[int, int]) -> Optional[object]:
    """Fast 2× thumbnail for the gallery."""
    try:
        _ensure_2d(mol)
        W, H  = size
        S     = _RENDER_SCALE
        drawer = rdMolDraw2D.MolDraw2DCairo(W * S, H * S)
        opts   = drawer.drawOptions()
        opts.bondLineWidth      = 1.6
        opts.addAtomIndices     = False
        opts.addStereoAnnotation = False
        opts.padding            = 0.08
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        hires = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")
        return hires.resize(size, Image.LANCZOS)
    except Exception:
        return None


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_sample(path: Path, n: int, seed: int) -> list[dict]:
    csv.field_size_limit(10 * 1024 * 1024)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    random.seed(seed)
    return random.sample(rows, min(n, len(rows)))


def _smiles_row(smiles: str) -> dict:
    return {"smiles": smiles, "chembl_id": "manual", "inchikey": ""}


# ── Main window ───────────────────────────────────────────────────────────────

class _MainWindow(tk.Tk):
    """Single-window viewer: gallery on the left, detail on the right."""

    def __init__(self, rows: list[dict], source_name: str, n: int, seed: int) -> None:
        super().__init__()
        self.title("SpinHance Viewer")
        self.configure(bg=_BG)
        self.resizable(True, True)
        _configure_styles(self)

        self._rows         = rows
        self._n            = n
        self._seed         = seed
        self._page         = 0
        self._n_pages      = max(1, (len(rows) + _PAGE_SIZE - 1) // _PAGE_SIZE)
        self._mol_idx:     int  = 0

        # Detail-panel state
        self._groups:         list  = []
        self._soft_color_map: dict  = {}
        self._atom_coords:    dict  = {}
        self._atom_to_label:  dict[int, str] = {}   # heavy_atom_idx → group label
        self._tree_tags:      dict[str, str]  = {}
        self._hover_label:    Optional[str]   = None
        self._base_image:     Optional[object] = None
        self._mol_photo:      Optional[object] = None

        self._photos:   list           = []   # gallery photo refs
        self._canvases: list[tk.Canvas] = []

        self._build_ui()
        self._file_lbl.config(text=source_name)
        self.after(0, self._init_load)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # ── Toolbar ───────────────────────────────────────────────────────────
        tb = ttk.Frame(self, padding=(10, 6))
        tb.pack(fill=tk.X)

        self._file_lbl = ttk.Label(tb, text="", style="Sub.TLabel")
        self._file_lbl.pack(side=tk.LEFT)

        ttk.Button(tb, text="Open file…",
                   command=self._open_file).pack(side=tk.RIGHT, padx=(6, 0))

        self._smiles_err = ttk.Label(tb, text="", foreground="#cc3333",
                                      font=(_FF, 9))
        self._smiles_err.pack(side=tk.RIGHT, padx=(6, 0))

        ttk.Button(tb, text="View", style="Accent.TButton",
                   command=self._view_smiles).pack(side=tk.RIGHT, padx=(4, 0))

        self._smiles_var = tk.StringVar()
        smi = ttk.Entry(tb, textvariable=self._smiles_var, width=50)
        smi.pack(side=tk.RIGHT, padx=(6, 4))
        smi.bind("<Return>", lambda e: self._view_smiles())

        ttk.Label(tb, text="SMILES:", font=(_FF, 9, "bold")).pack(side=tk.RIGHT)

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X)

        # ── Main horizontal split ─────────────────────────────────────────────
        pw = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # Left: gallery
        left = ttk.Frame(pw, padding=0)
        pw.add(left, weight=0)
        self._build_gallery(left)

        # Divider appearance
        ttk.Separator(pw, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y)

        # Right: detail
        right = ttk.Frame(pw, style="Card.TFrame", padding=0)
        pw.add(right, weight=1)
        self._build_detail(right)

    # ── Gallery ───────────────────────────────────────────────────────────────

    def _build_gallery(self, parent: ttk.Frame) -> None:
        grid = ttk.Frame(parent, padding=(8, 8, 8, 4))
        grid.pack(fill=tk.BOTH, expand=True)

        for i in range(_PAGE_SIZE):
            r, c = divmod(i, _GRID_COLS)
            cell = tk.Frame(grid, bg=_BG, highlightbackground=_BORDER,
                            highlightthickness=1)
            cell.grid(row=r, column=c, padx=3, pady=3)
            cv = tk.Canvas(cell, width=_THUMB_W, height=_CELL_H,
                           bg=_PANEL_BG, highlightthickness=0, cursor="hand2")
            cv.pack()
            self._canvases.append(cv)

        nav = ttk.Frame(parent, padding=(8, 0, 8, 8))
        nav.pack(fill=tk.X)
        ttk.Button(nav, text="◀", width=3,
                   command=self._prev_page).pack(side=tk.LEFT)
        self._page_var = tk.StringVar()
        ttk.Label(nav, textvariable=self._page_var,
                  font=(_FF, 9), width=20,
                  anchor=tk.CENTER).pack(side=tk.LEFT, expand=True)
        ttk.Button(nav, text="▶", width=3,
                   command=self._next_page).pack(side=tk.RIGHT)

    # ── Detail panel ──────────────────────────────────────────────────────────

    def _build_detail(self, parent: ttk.Frame) -> None:
        # Info bar
        info = ttk.Frame(parent, style="Card.TFrame", padding=(14, 10, 14, 6))
        info.pack(fill=tk.X)

        self._id_var     = tk.StringVar(value="Select a molecule")
        self._status_var = tk.StringVar(value="")
        ttk.Label(info, textvariable=self._id_var,
                  style="ID.TLabel").pack(side=tk.LEFT)
        ttk.Label(info, textvariable=self._status_var,
                  style="Sub.TLabel").pack(side=tk.RIGHT)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X)

        # Canvas + table side by side
        content = ttk.Frame(parent, style="Card.TFrame", padding=(10, 10, 10, 4))
        content.pack(fill=tk.BOTH, expand=True)
        content.columnconfigure(0, weight=0)   # canvas — fixed
        content.columnconfigure(1, weight=0)   # table  — fixed
        content.rowconfigure(0, weight=1)

        # Molecule canvas — fixed size, never resized
        cv_frame = tk.Frame(content, bg=_PANEL_BG,
                            highlightbackground=_BORDER, highlightthickness=1)
        cv_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 10))
        self._mol_canvas = tk.Canvas(cv_frame, width=_MOL_W, height=_MOL_H,
                                      bg="white", highlightthickness=0)
        self._mol_canvas.pack()
        self._mol_canvas.bind("<Motion>", self._on_canvas_motion)
        self._mol_canvas.bind("<Leave>",  self._on_canvas_leave)

        # Spin-group table
        tbl_frame = ttk.LabelFrame(content, text="Spin groups",
                                    width=_TABLE_W, padding=(4, 4))
        tbl_frame.grid(row=0, column=1, sticky=tk.NS)
        tbl_frame.pack_propagate(False)

        self._tree = ttk.Treeview(
            tbl_frame, columns=("tier", "h"),
            show="tree headings", selectmode="none",
        )
        self._tree.heading("#0",   text="Group")
        self._tree.heading("tier", text="Tier")
        self._tree.heading("h",    text="H")
        self._tree.column("#0",   width=58,  anchor=tk.CENTER, stretch=False)
        self._tree.column("tier", width=52,  anchor=tk.CENTER, stretch=False)
        self._tree.column("h",    width=38,  anchor=tk.CENTER, stretch=False)

        vsb = ttk.Scrollbar(tbl_frame, orient=tk.VERTICAL,
                            command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._tree.pack(fill=tk.BOTH, expand=True)

        self._tree.tag_configure("hard",      foreground=_TIER_FG["HARD"],
                                  background="#e8eeff",
                                  font=(_FF, 9, "bold"))
        self._tree.tag_configure("none_tier", foreground=_TIER_FG["NONE"],
                                  background=_ROW_ODD)
        self._tree.tag_configure("hover",     background=_HOVER_ROW)

        self._tree.bind("<Motion>", self._on_table_motion)
        self._tree.bind("<Leave>",  self._on_table_leave)

        # SMILES bar
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X)
        smi_bar = ttk.Frame(parent, style="Card.TFrame",
                             padding=(10, 5, 10, 5))
        smi_bar.pack(fill=tk.X)
        ttk.Label(smi_bar, text="SMILES",
                  font=(_FF, 9, "bold"),
                  style="Sub.TLabel").pack(side=tk.LEFT, padx=(0, 8))
        self._smiles_txt = tk.Text(
            smi_bar, height=2, font=("Courier New" if _platform.system() == "Windows"
                                      else "Courier", 9),
            wrap=tk.WORD, state=tk.DISABLED, relief=tk.FLAT,
            bg="#f4f5f7", borderwidth=0,
        )
        self._smiles_txt.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Mol navigation
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X)
        mol_nav = ttk.Frame(parent, style="Card.TFrame",
                             padding=(10, 6, 10, 8))
        mol_nav.pack(fill=tk.X)
        ttk.Button(mol_nav, text="◀  Prev",
                   style="Nav.TButton",
                   command=self._prev_mol).pack(side=tk.LEFT)
        self._mol_ctr = tk.StringVar(value="")
        ttk.Label(mol_nav, textvariable=self._mol_ctr,
                  font=(_FF, 10), anchor=tk.CENTER,
                  width=14).pack(side=tk.LEFT, expand=True)
        ttk.Button(mol_nav, text="Next  ▶",
                   style="Nav.TButton",
                   command=self._next_mol).pack(side=tk.RIGHT)

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_load(self) -> None:
        self._draw_gallery_page()
        if self._rows:
            self._show_mol(0)

    # ── Gallery ───────────────────────────────────────────────────────────────

    def _draw_gallery_page(self) -> None:
        new_photos: list = []
        start     = self._page * _PAGE_SIZE
        page_rows = self._rows[start : start + _PAGE_SIZE]

        for i, cv in enumerate(self._canvases):
            cv.unbind("<Button-1>")
            cv.unbind("<Enter>")
            cv.unbind("<Leave>")
            cv.delete("all")
            cv.config(bg=_PANEL_BG)

            if i >= len(page_rows):
                continue

            row = page_rows[i]
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is None:
                continue

            thumb = _draw_thumbnail(mol, (_THUMB_W, _THUMB_H))
            if thumb is None:
                continue

            photo = ImageTk.PhotoImage(thumb)
            new_photos.append(photo)

            cv.create_image(0, 0, anchor=tk.NW, image=photo)
            cv.create_rectangle(
                0, _THUMB_H, _THUMB_W, _CELL_H,
                fill="#e8ecf5", outline="",
            )
            cv.create_text(
                _THUMB_W // 2, _THUMB_H + _THUMB_LBL // 2,
                text=row.get("chembl_id", ""),
                font=(_FF, 8), fill="#555555",
            )

            abs_idx = start + i

            # Highlight the currently shown molecule
            if abs_idx == self._mol_idx:
                cv.create_rectangle(
                    1, 1, _THUMB_W - 1, _CELL_H - 1,
                    outline=_ACCENT, width=2, fill="",
                )

            cv.bind("<Button-1>",
                    lambda e, idx=abs_idx: self._show_mol(idx))
            cv.bind("<Enter>",
                    lambda e, c=cv, idx=abs_idx:
                        c.config(bg="#dde5f7") if idx != self._mol_idx else None)
            cv.bind("<Leave>",
                    lambda e, c=cv: c.config(bg=_PANEL_BG))

        self._photos = new_photos
        self._page_var.set(
            f"Page {self._page + 1} / {self._n_pages}"
            f"  •  {len(self._rows)} molecules"
        )

    def _prev_page(self) -> None:
        if self._page > 0:
            self._page -= 1
            self._draw_gallery_page()

    def _next_page(self) -> None:
        if self._page < self._n_pages - 1:
            self._page += 1
            self._draw_gallery_page()

    # ── Detail view ───────────────────────────────────────────────────────────

    def _show_mol(self, idx: int) -> None:
        old_page = self._mol_idx // _PAGE_SIZE
        self._mol_idx = idx
        new_page = idx // _PAGE_SIZE

        if new_page != self._page:
            self._page = new_page
        self._draw_gallery_page()     # re-draw to show selection ring

        row = self._rows[idx]
        self._id_var.set(row.get("chembl_id", "—"))
        self._mol_ctr.set(f"{idx + 1} / {len(self._rows)}")

        self._smiles_txt.config(state=tk.NORMAL)
        self._smiles_txt.delete("1.0", tk.END)
        self._smiles_txt.insert("1.0", row["smiles"])
        self._smiles_txt.config(state=tk.DISABLED)

        # Clear detail state
        self._groups         = []
        self._soft_color_map = {}
        self._atom_coords    = {}
        self._atom_to_label  = {}
        self._tree_tags      = {}
        self._hover_label    = None
        self._base_image     = None
        self._status_var.set("Computing…")
        self.update_idletasks()

        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            self._status_var.set("Invalid SMILES")
            return

        _, groups = classify_spin_groups(mol)
        scm       = _build_soft_color_map(groups)
        self._groups         = groups
        self._soft_color_map = scm

        # Map heavy atom → spin group label (for canvas hover)
        for g in groups:
            for hi in g.heavy_parent_indices:
                self._atom_to_label[hi] = g.label

        img, coords = _draw_molecule(mol, groups, soft_color_map=scm)
        self._base_image  = img
        self._atom_coords = coords
        self._show_image(img)

        self._populate_table(groups, scm)
        self._status_var.set(f"{len(groups)} spin groups")

    def _show_image(self, img) -> None:
        """Update the canvas image WITHOUT resizing the canvas."""
        self._mol_photo = ImageTk.PhotoImage(img)
        self._mol_canvas.create_image(0, 0, anchor=tk.NW, image=self._mol_photo)

    def _populate_table(self, groups: list, scm: dict) -> None:
        self._tree.delete(*self._tree.get_children())

        seen_cls: dict[tuple, str] = {}
        for g in groups:
            if g.tier == "SOFT" and g.class_h_indices not in seen_cls:
                tag  = f"soft_{len(seen_cls)}"
                tkcol, _ = scm.get(g.class_h_indices, ("#e8f5e9", None))
                self._tree.tag_configure(
                    tag, background=tkcol, foreground=_TIER_FG["SOFT"],
                )
                seen_cls[g.class_h_indices] = tag

        for g in groups:
            if g.tier == "HARD":
                tag = "hard"
            elif g.tier == "SOFT":
                tag = seen_cls[g.class_h_indices]
            else:
                tag = "none_tier"
            self._tree_tags[g.label] = tag
            self._tree.insert(
                "", "end",
                text=f"  {g.label}",
                values=(_TIER_TEXT[g.tier], len(g.h_indices)),
                tags=(tag,),
            )

    def _redraw_mol(self, highlight_label: Optional[str]) -> None:
        """Redraw with a different group highlighted — canvas size NEVER changes."""
        if self._base_image is None or not self._rows:
            return
        mol = Chem.MolFromSmiles(self._rows[self._mol_idx]["smiles"])
        if mol is None:
            return
        img, _ = _draw_molecule(
            mol, self._groups,
            soft_color_map=self._soft_color_map,
            highlight_label=highlight_label,
        )
        self._show_image(img)

    def _set_hover(self, label: Optional[str]) -> None:
        if label == self._hover_label:
            return
        self._hover_label = label

        # Table: restore all rows then apply hover to the matching one
        for iid in self._tree.get_children():
            lbl  = self._tree.item(iid, "text").strip()
            base = self._tree_tags.get(lbl, "none_tier")
            self._tree.item(iid, tags=(base,))
        if label:
            for iid in self._tree.get_children():
                if self._tree.item(iid, "text").strip() == label:
                    base = self._tree_tags.get(label, "none_tier")
                    self._tree.item(iid, tags=(base, "hover"))
                    self._tree.see(iid)
                    break

        self._redraw_mol(label)

    # ── Table mouseover ───────────────────────────────────────────────────────

    def _on_table_motion(self, event: tk.Event) -> None:
        item  = self._tree.identify_row(event.y)
        label = self._tree.item(item, "text").strip() if item else None
        self._set_hover(label)

    def _on_table_leave(self, event: tk.Event) -> None:
        self._set_hover(None)

    # ── Canvas mouseover → highlight table ───────────────────────────────────

    def _on_canvas_motion(self, event: tk.Event) -> None:
        if not self._atom_coords:
            return
        mx, my    = event.x, event.y
        best_d    = float("inf")
        best_lbl  = None
        threshold = 20.0

        for hi, (ax, ay) in self._atom_coords.items():
            d = math.hypot(mx - ax, my - ay)
            if d < best_d:
                best_d   = d
                if d < threshold:
                    best_lbl = self._atom_to_label.get(hi)

        self._set_hover(best_lbl)

    def _on_canvas_leave(self, event: tk.Event) -> None:
        self._set_hover(None)

    # ── Molecule navigation ───────────────────────────────────────────────────

    def _prev_mol(self) -> None:
        if self._rows:
            self._show_mol((self._mol_idx - 1) % len(self._rows))

    def _next_mol(self) -> None:
        if self._rows:
            self._show_mol((self._mol_idx + 1) % len(self._rows))

    # ── SMILES entry ──────────────────────────────────────────────────────────

    def _view_smiles(self) -> None:
        smi = self._smiles_var.get().strip()
        if not smi:
            return
        if Chem.MolFromSmiles(smi) is None:
            self._smiles_err.config(text="Invalid SMILES")
            return
        self._smiles_err.config(text="")
        # Prepend a temporary row and show it
        self._rows.insert(0, _smiles_row(smi))
        self._n_pages = max(1, (len(self._rows) + _PAGE_SIZE - 1) // _PAGE_SIZE)
        self._show_mol(0)

    # ── File open ─────────────────────────────────────────────────────────────

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
        self._mol_idx = 0
        self._file_lbl.config(text=p.name)
        self._draw_gallery_page()
        if self._rows:
            self._show_mol(0)


# ── Public entry point ────────────────────────────────────────────────────────

def launch(path: Path, *, n: int = 80, seed: int = 42) -> None:
    """Launch the single-window viewer for *path*."""
    global Chem, RDLogger, AllChem, Draw, rdMolDraw2D
    global Image, ImageTk, ImageDraw, ImageFont
    global classify_spin_groups, SpinGroup

    try:
        from PIL import Image as _I, ImageTk as _IT, ImageDraw as _ID, ImageFont as _IF
        from rdkit import Chem as _C, RDLogger as _RL
        from rdkit.Chem import AllChem as _A, Draw as _D
        from rdkit.Chem.Draw import rdMolDraw2D as _rmd
        from generate.spin_equivalence import (
            classify_spin_groups as _csg, SpinGroup as _SG,
        )
    except ImportError as exc:
        sys.exit(f"Missing dependency: {exc}")

    Chem  = _C;  RDLogger = _RL;  AllChem = _A;  Draw = _D
    rdMolDraw2D = _rmd
    Image = _I;  ImageTk  = _IT;  ImageDraw = _ID; ImageFont = _IF
    classify_spin_groups = _csg;  SpinGroup = _SG
    RDLogger.DisableLog("rdApp.*")

    rows = load_sample(path, n, seed)
    _MainWindow(rows, path.name, n, seed).mainloop()
