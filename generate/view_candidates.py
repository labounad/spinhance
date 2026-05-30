"""view_candidates.py — interactive GUI viewer for candidate molecule sets.

Gallery: 16 molecules per page, click any to open a detail view.
Detail: structure, stats, SMILES, and a collapsible deuterium sidebar
        with each D atom shown as a bold blue labelled circle.

Usage:
    python generate/view_candidates.py --file generate/data/candidates_8spin_round01.csv
    python generate/view_candidates.py --file generate/data/smiles_8spin_round02.csv --n 200
"""

import argparse
import csv
import io
import random
import sys
from pathlib import Path
from tkinter import filedialog
import tkinter as tk
from tkinter import ttk

try:
    from PIL import Image, ImageDraw, ImageFont, ImageTk
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    RDLogger.DisableLog("rdApp.*")
except ImportError as e:
    sys.exit(f"Missing dependency: {e}")

sys.path.insert(0, str(Path(__file__).parent))
from spin_group_filter import (
    strip_exchangeable_protons, analyze_spin_systems,
    _embed_3d, _signature,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# layout constants
PAGE_SIZE   = 16
GRID_COLS   = 4
THUMB_W     = 215
THUMB_H     = 170
LABEL_H     = 20          # strip below thumbnail for ChEMBL ID
CELL_H      = THUMB_H + LABEL_H
MOL_W       = 480
MOL_H       = 390
DEUT_W      = 240
DEUT_H      = 205
DEUT_COLS   = 2
DEUT_CELL_H = DEUT_H + 20   # molecule + label strip

# ── font helper ───────────────────────────────────────────────────────────────

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
]

def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _draw_text_centered(draw: ImageDraw.Draw, xy, text: str, font, fill):
    try:
        draw.text(xy, text, fill=fill, font=font, anchor="mm")
    except TypeError:
        w = len(text) * 6
        draw.text((xy[0] - w // 2, xy[1] - 7), text, fill=fill, font=font)


# ── molecule rendering ────────────────────────────────────────────────────────

def _ensure_2d(mol: Chem.Mol) -> Chem.Mol:
    if mol.GetNumConformers() == 0:
        AllChem.Compute2DCoords(mol)
    return mol


def draw_deuterated(mol: Chem.Mol, size=(DEUT_W, DEUT_H)) -> Image.Image:
    """Render a deuterated molecule with each D atom as a bold blue 'D' circle."""
    _ensure_2d(mol)

    d_idxs = [
        a.GetIdx() for a in mol.GetAtoms()
        if a.GetAtomicNum() == 1 and a.GetIsotope() == 2
    ]

    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    drawer.drawOptions().addStereoAnnotation = True
    drawer.DrawMolecule(
        mol,
        highlightAtoms=d_idxs,
        highlightAtomColors={idx: (0.1, 0.45, 0.95) for idx in d_idxs},
        highlightAtomRadii={idx: 0.45 for idx in d_idxs},
        highlightBonds=[],
    )
    drawer.FinishDrawing()

    d_pixels = []
    for idx in d_idxs:
        try:
            pt = drawer.GetDrawCoords(idx)
            d_pixels.append((pt.x, pt.y))
        except Exception:
            pass

    img   = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")
    paint = ImageDraw.Draw(img)
    font  = _font(14)

    for x, y in d_pixels:
        r = 12
        paint.ellipse([x - r, y - r, x + r, y + r], fill=(20, 90, 230, 220))
        _draw_text_centered(paint, (x, y), "D", font, "white")

    return img.convert("RGB")


# ── data / analysis helpers ───────────────────────────────────────────────────

def load_sample(path: Path, n: int, seed: int) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    random.seed(seed)
    return random.sample(rows, min(n, len(rows)))


def mol_stats(mol: Chem.Mol) -> tuple[int, int, int, list[int]]:
    n_c      = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
    mol_h, _ = _embed_3d(mol)
    mol_ch   = strip_exchangeable_protons(mol_h)
    n_h      = sum(1 for a in mol_ch.GetAtoms() if a.GetAtomicNum() == 1)
    n_spin, sizes = analyze_spin_systems(mol)
    return n_c, n_h, n_spin, sizes


def deuterated_representatives(mol: Chem.Mol) -> tuple[list[Chem.Mol], list[str]]:
    mol_h, use_3d = _embed_3d(mol)
    mol_h = strip_exchangeable_protons(mol_h)

    seen: dict[str, Chem.Mol] = {}
    counts: dict[str, int]    = {}

    for atom in mol_h.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        smi = _signature(mol_h, atom.GetIdx(), use_3d=use_3d)
        if smi not in seen:
            rw = Chem.RWMol(mol_h)
            rw.GetAtomWithIdx(atom.GetIdx()).SetIsotope(2)
            seen[smi] = rw.GetMol()
        counts[smi] = counts.get(smi, 0) + 1

    mols   = list(seen.values())
    labels = [f"Group {i+1}  ({counts[s]} H)" for i, s in enumerate(seen)]
    return mols, labels


def _build_deut_grid(mol: Chem.Mol) -> Image.Image:
    """Render all deuterated representatives into a labelled grid image."""
    d_mols, labels = deuterated_representatives(mol)
    n    = len(d_mols)
    cols = DEUT_COLS
    rows = (n + cols - 1) // cols
    gw   = DEUT_W * cols
    gh   = DEUT_CELL_H * rows

    grid  = Image.new("RGB", (gw, gh), (245, 245, 245))
    paint = ImageDraw.Draw(grid)
    font  = _font(11)

    for i, (d_mol, label) in enumerate(zip(d_mols, labels)):
        c, r  = i % cols, i // cols
        mol_img = draw_deuterated(d_mol, (DEUT_W, DEUT_H))
        grid.paste(mol_img, (c * DEUT_W, r * DEUT_CELL_H))

        lx = c * DEUT_W
        ly = r * DEUT_CELL_H + DEUT_H
        paint.rectangle([lx, ly, lx + DEUT_W, ly + 20], fill=(210, 215, 225))
        _draw_text_centered(paint, (lx + DEUT_W // 2, ly + 10), label, font, (40, 40, 40))

    return grid


# ── detail window ─────────────────────────────────────────────────────────────

class DetailWindow(tk.Toplevel):
    def __init__(self, parent: tk.Tk, rows: list[dict], idx: int):
        super().__init__(parent)
        self.resizable(False, False)
        self._rows = rows
        self._idx  = idx
        self._mol_photo  = None
        self._deut_photo = None
        self._deut_visible = False
        self._build_ui()
        self._refresh()

    def _build_ui(self):
        self._main = ttk.Frame(self, padding=8)
        self._main.pack(fill=tk.BOTH, expand=True)

        self._canvas = tk.Canvas(self._main, width=MOL_W, height=MOL_H, bg="white")
        self._canvas.pack(side=tk.LEFT)

        info = ttk.Frame(self._main, padding=(12, 0, 6, 0))
        info.pack(side=tk.LEFT, fill=tk.Y)

        def row_var(r, label):
            ttk.Label(info, text=label, font=("TkDefaultFont", 9, "bold")).grid(
                row=r, column=0, sticky=tk.W, pady=3)
            var = tk.StringVar(value="-")
            ttk.Label(info, textvariable=var).grid(row=r, column=1, sticky=tk.W, padx=8)
            return var

        self._id_var   = row_var(0, "ChEMBL ID")
        self._c_var    = row_var(1, "Carbons")
        self._h_var    = row_var(2, "C-H protons")
        self._spin_var = row_var(3, "Spin systems")

        ttk.Separator(info, orient=tk.HORIZONTAL).grid(row=4, columnspan=2, sticky=tk.EW, pady=8)

        ttk.Label(info, text="SMILES", font=("TkDefaultFont", 9, "bold")).grid(
            row=5, column=0, sticky=tk.NW)
        self._smiles_txt = tk.Text(
            info, width=32, height=6, font=("TkFixedFont", 8),
            wrap=tk.WORD, state=tk.DISABLED, relief=tk.FLAT, bg="#f0f0f0")
        self._smiles_txt.grid(row=5, column=1, sticky=tk.W, padx=8)

        ttk.Separator(info, orient=tk.HORIZONTAL).grid(row=6, columnspan=2, sticky=tk.EW, pady=8)

        self._deut_btn = ttk.Button(
            info, text="Show deuterated set", command=self._toggle_deut)
        self._deut_btn.grid(row=7, columnspan=2)

        # deuterium sidebar (hidden initially)
        self._deut_panel = ttk.LabelFrame(self._main, text="Deuterated Set", padding=4)
        dw = DEUT_W * DEUT_COLS + 20
        self._deut_cv = tk.Canvas(
            self._deut_panel, width=dw, bg="white", highlightthickness=0)
        self._deut_sb = ttk.Scrollbar(
            self._deut_panel, orient=tk.VERTICAL, command=self._deut_cv.yview)
        self._deut_cv.configure(yscrollcommand=self._deut_sb.set)
        self._deut_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._deut_cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X)
        nav = ttk.Frame(self, padding=(6, 4))
        nav.pack(fill=tk.X)
        ttk.Button(nav, text="< Prev", command=self._prev).pack(side=tk.LEFT)
        self._counter = tk.StringVar()
        ttk.Label(nav, textvariable=self._counter, width=12, anchor=tk.CENTER).pack(
            side=tk.LEFT, padx=6)
        ttk.Button(nav, text="Next >", command=self._next).pack(side=tk.LEFT)

    def _refresh(self):
        row = self._rows[self._idx]
        mol = Chem.MolFromSmiles(row["smiles"])

        if mol:
            img = Draw.MolToImage(_ensure_2d(mol), size=(MOL_W, MOL_H))
            self._mol_photo = ImageTk.PhotoImage(img)
            self._canvas.create_image(0, 0, anchor=tk.NW, image=self._mol_photo)

        self._id_var.set(row.get("chembl_id", "-"))

        self._smiles_txt.config(state=tk.NORMAL)
        self._smiles_txt.delete("1.0", tk.END)
        self._smiles_txt.insert("1.0", row["smiles"])
        self._smiles_txt.config(state=tk.DISABLED)

        if mol:
            n_c, n_h, n_spin, _ = mol_stats(mol)
            self._c_var.set(str(n_c))
            self._h_var.set(str(n_h))
            self._spin_var.set(str(n_spin))
        else:
            for v in (self._c_var, self._h_var, self._spin_var):
                v.set("(invalid)")

        self._counter.set(f"{self._idx + 1} / {len(self._rows)}")
        self.title(f"Detail  -  {row.get('chembl_id', '')}")

        if self._deut_visible and mol:
            self._render_deut(mol)

    def _prev(self):
        self._idx = (self._idx - 1) % len(self._rows)
        self._refresh()

    def _next(self):
        self._idx = (self._idx + 1) % len(self._rows)
        self._refresh()

    def _toggle_deut(self):
        if self._deut_visible:
            self._deut_panel.pack_forget()
            self._deut_visible = False
            self._deut_btn.config(text="Show deuterated set")
        else:
            mol = Chem.MolFromSmiles(self._rows[self._idx]["smiles"])
            if mol:
                self._render_deut(mol)
                self._deut_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 0))
                self._deut_visible = True
                self._deut_btn.config(text="Hide deuterated set")

    def _render_deut(self, mol: Chem.Mol):
        grid = _build_deut_grid(mol)
        self._deut_photo = ImageTk.PhotoImage(grid)
        self._deut_cv.delete("all")
        self._deut_cv.create_image(0, 0, anchor=tk.NW, image=self._deut_photo)
        self._deut_cv.configure(
            height=min(grid.height, MOL_H),
            scrollregion=(0, 0, grid.width, grid.height),
        )


# ── gallery view ──────────────────────────────────────────────────────────────

class GalleryView(tk.Tk):
    def __init__(self, rows: list[dict], source_name: str, n: int, seed: int):
        super().__init__()
        self.title("SpinHance Candidate Viewer")
        self.resizable(False, False)
        ttk.Style(self).theme_use("clam")

        self._rows    = rows
        self._n       = n
        self._seed    = seed
        self._page    = 0
        self._n_pages = max(1, (len(rows) + PAGE_SIZE - 1) // PAGE_SIZE)
        self._photos: list = []
        self._canvases: list[tk.Canvas] = []

        self._build_ui()
        self._file_lbl.config(text=source_name)
        self._draw_page()

    def _build_ui(self):
        top = ttk.Frame(self, padding=(6, 4))
        top.pack(fill=tk.X)
        self._file_lbl = ttk.Label(top, text="", font=("TkDefaultFont", 9, "italic"))
        self._file_lbl.pack(side=tk.LEFT)
        ttk.Button(top, text="Open file...", command=self._open_file).pack(side=tk.RIGHT)

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X)

        grid_frame = ttk.Frame(self, padding=6)
        grid_frame.pack(fill=tk.BOTH, expand=True)

        for i in range(PAGE_SIZE):
            r, c = divmod(i, GRID_COLS)
            frame = ttk.Frame(grid_frame, relief=tk.GROOVE, borderwidth=1)
            frame.grid(row=r, column=c, padx=3, pady=3)

            cv = tk.Canvas(frame, width=THUMB_W, height=CELL_H,
                           bg="white", highlightthickness=0, cursor="hand2")
            cv.pack()
            self._canvases.append(cv)

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X)
        nav = ttk.Frame(self, padding=(6, 4))
        nav.pack(fill=tk.X)
        ttk.Button(nav, text="< Prev", command=self._prev_page).pack(side=tk.LEFT)
        self._page_var = tk.StringVar()
        ttk.Label(nav, textvariable=self._page_var, width=22, anchor=tk.CENTER).pack(
            side=tk.LEFT, padx=6)
        ttk.Button(nav, text="Next >", command=self._next_page).pack(side=tk.LEFT)

    def _draw_page(self):
        new_photos: list = []
        start     = self._page * PAGE_SIZE
        page_rows = self._rows[start : start + PAGE_SIZE]

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

            thumb = Draw.MolToImage(_ensure_2d(mol), size=(THUMB_W, THUMB_H))
            photo = ImageTk.PhotoImage(thumb)
            new_photos.append(photo)

            cv.create_image(0, 0, anchor=tk.NW, image=photo)
            cv.create_rectangle(0, THUMB_H, THUMB_W, CELL_H, fill="#dde3ed", outline="")
            cv.create_text(THUMB_W // 2, THUMB_H + LABEL_H // 2,
                           text=row.get("chembl_id", ""),
                           font=("TkDefaultFont", 8), fill="#222222")

            abs_idx = start + i
            cv.bind("<Button-1>", lambda e, idx=abs_idx: DetailWindow(self, self._rows, idx))
            cv.bind("<Enter>",    lambda e, c=cv: c.config(bg="#cfe0f8"))
            cv.bind("<Leave>",    lambda e, c=cv: c.config(bg="white"))

        self._photos = new_photos
        self._page_var.set(
            f"Page {self._page + 1} / {self._n_pages}  "
            f"({len(self._rows)} molecules)"
        )

    def _prev_page(self):
        if self._page > 0:
            self._page -= 1
            self._draw_page()

    def _next_page(self):
        if self._page < self._n_pages - 1:
            self._page += 1
            self._draw_page()

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Open candidate CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=str(REPO_ROOT / "generate" / "data"),
        )
        if not path:
            return
        p = Path(path)
        self._rows    = load_sample(p, self._n, self._seed)
        self._n_pages = max(1, (len(self._rows) + PAGE_SIZE - 1) // PAGE_SIZE)
        self._page    = 0
        self._file_lbl.config(text=p.name)
        self._draw_page()


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file", type=Path, required=True, help="path to any round CSV")
    p.add_argument("--n",    type=int,  default=80,   help="molecules to sample (default 80)")
    p.add_argument("--seed", type=int,  default=42)
    args = p.parse_args()

    if not args.file.exists():
        sys.exit(f"File not found: {args.file}")

    rows = load_sample(args.file, args.n, args.seed)
    GalleryView(rows, args.file.name, args.n, args.seed).mainloop()


if __name__ == "__main__":
    main()
