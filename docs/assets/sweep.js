/* Spinhance — scroll-driven field-sweep hero
   Loads docs/data/field_sweep.json (a STICK spectrum per molecule per field) and
   broadens the sticks into smooth Lorentzians on a high-res grid, client-side, so
   resolution is independent of stored data size. The bold current-field trace is
   driven by scroll; a faint static "fan" of all fields sits behind it. */
(() => {
  "use strict";

  /* ---------- theme ---------- */
  const root = document.documentElement;
  const btn = document.getElementById("themeBtn");
  const saved = localStorage.getItem("spinhance-theme");
  if (saved) root.setAttribute("data-theme", saved);
  const syncBtn = () => { btn.textContent = root.getAttribute("data-theme") === "dark" ? "☀️" : "🌙"; };
  syncBtn();
  btn.addEventListener("click", () => {
    const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    localStorage.setItem("spinhance-theme", next);
    syncBtn();
    colors = readColors();
    renderFan(); draw();
  });

  const cssVar = (n) => getComputedStyle(root).getPropertyValue(n).trim();
  const readColors = () => ({
    trace: cssVar("--trace"), fan: cssVar("--fan"), grid: cssVar("--grid"),
    faint: cssVar("--ink-faint"), accent: cssVar("--accent"),
    dark: root.getAttribute("data-theme") === "dark",
  });
  let colors = readColors();

  /* ---------- elements ---------- */
  const canvas = document.getElementById("spectrum");
  const ctx = canvas.getContext("2d");
  const hero = document.getElementById("top");
  const fieldVal = document.getElementById("fieldVal");
  const barFill = document.getElementById("barFill");
  const molTag = document.getElementById("molTag");
  const scrollHint = document.getElementById("scrollHint");

  const GRID = 4096;          // broadening resolution (independent of pixels)
  const BASE = 0.75, AMP = 0.56;  // baseline at 75% height; peaks rise 56% of height
  const FADE_MIN = 0.14;          // faint persistent backdrop after the sweep completes
  const FADE_VH = 0.85;           // fraction of a viewport over which it fades out

  let W = 0, H = 0, dpr = 1;
  const fan = document.createElement("canvas");
  const fctx = fan.getContext("2d");

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = canvas.clientWidth; H = canvas.clientHeight;
    for (const c of [canvas, fan]) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    fctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  /* ---------- data ---------- */
  let meta = null, mol = null, curves = [], winLo = 0, winHi = 1;

  const b64f32 = (s) => { const b = atob(s), n = b.length / 4, u = new Uint8Array(b.length);
    for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new Float32Array(u.buffer, 0, n); };
  const b64u16 = (s) => { const b = atob(s), n = b.length / 2, out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = (b.charCodeAt(2*i) | (b.charCodeAt(2*i+1) << 8)) / 65535; return out; };

  /* broaden one frame's sticks into a normalized Lorentzian curve on the grid */
  function broaden(centers, amps, hwhm) {
    const y = new Float32Array(GRID);
    const dppm = (winHi - winLo) / (GRID - 1);
    const cutoff = Math.max(30 * hwhm, dppm * 3);
    for (let i = 0; i < centers.length; i++) {
      const c = centers[i], a = amps[i];
      let k0 = Math.floor((c - cutoff - winLo) / dppm), k1 = Math.ceil((c + cutoff - winLo) / dppm);
      if (k0 < 0) k0 = 0; if (k1 > GRID - 1) k1 = GRID - 1;
      for (let k = k0; k <= k1; k++) {
        const d = (winLo + k * dppm - c) / hwhm;
        y[k] += a / (1 + d * d);
      }
    }
    let m = 0; for (let k = 0; k < GRID; k++) if (y[k] > m) m = y[k];
    if (m > 0) for (let k = 0; k < GRID; k++) y[k] /= m;
    return y;
  }

  function chooseMolecule(data) {
    meta = data.meta;
    mol = data.molecules[Math.floor(Math.random() * data.molecules.length)];
    [winLo, winHi] = mol.win;
    curves = mol.frames.map((fr, idx) => {
      const hwhm = (meta.linewidth_hz / 2) / meta.fields_mhz[idx];
      return broaden(b64f32(fr.c), b64u16(fr.a), hwhm);
    });
    molTag.innerHTML = `<b>${mol.chembl_id || mol.id || "molecule"}</b> &nbsp;` +
      `<span class="mono smi" id="smilesCopy" title="Click to copy SMILES">${mol.smiles || ""}</span>` +
      `<span class="copied" id="copiedMsg" style="opacity:0">✓ copied</span>`;
    buildMatrix();
    // hand the molecule to the 3D viewer module
    window.__heroMol = { smiles: mol.smiles, id: mol.chembl_id || mol.id, xyz: mol.xyz };
    window.dispatchEvent(new CustomEvent("spinhance:molecule"));
  }

  // click SMILES -> copy to clipboard AND jump to "The representation"
  molTag.addEventListener("click", (e) => {
    const t = e.target.closest(".smi");
    if (!t || !mol) return;
    navigator.clipboard.writeText(mol.smiles || "").then(() => {
      const m = document.getElementById("copiedMsg");
      if (m) { m.style.opacity = "1"; clearTimeout(molTag._ct); molTag._ct = setTimeout(() => m.style.opacity = "0", 1300); }
    }).catch(() => {});
    const rep = document.getElementById("rep");
    if (rep) rep.scrollIntoView({ behavior: "smooth", block: "start" });
  });

  function curvePath(c, g, amp, baseY) {
    // k=0 -> ppm=winLo (low ppm) -> right side; reversed axis = high ppm on left
    g.beginPath();
    for (let k = 0; k < GRID; k++) {
      const x = (1 - k / (GRID - 1)) * W;     // winLo(low ppm) at right, winHi(high ppm) at left
      const y = baseY - c[k] * amp;
      k === 0 ? g.moveTo(x, y) : g.lineTo(x, y);
    }
    g.stroke();
  }

  function drawAxis(g, baseY) {
    g.strokeStyle = colors.grid; g.lineWidth = 1;
    g.beginPath(); g.moveTo(0, baseY); g.lineTo(W, baseY); g.stroke();
    g.fillStyle = colors.faint; g.font = "12px ui-monospace,Menlo,monospace"; g.textAlign = "center";
    const sp = (winHi - winLo) > 4 ? 1 : (winHi - winLo) > 2 ? 0.5 : 0.25;
    const first = Math.ceil(winLo / sp) * sp;
    for (let p = first; p <= winHi + 1e-6; p += sp) {
      const x = ((winHi - p) / (winHi - winLo)) * W;
      g.beginPath(); g.moveTo(x, baseY); g.lineTo(x, baseY + 6); g.strokeStyle = colors.grid; g.stroke();
      g.fillText(p.toFixed(sp < 1 ? (sp < 0.5 ? 2 : 1) : 0), x, baseY + 21);
    }
    g.textAlign = "left"; g.fillText("δ (ppm)", 12, baseY + 21);
  }

  /* fan + axis are static per molecule/size/theme -> render once to offscreen */
  function renderFan() {
    if (!meta) return;
    const baseY = H * BASE, amp = H * AMP;
    fctx.clearRect(0, 0, W, H);
    drawAxis(fctx, baseY);
    fctx.strokeStyle = colors.fan; fctx.lineWidth = 1.1; fctx.lineJoin = "round";
    for (const c of curves) curvePath(c, fctx, amp, baseY);
  }

  const clamp01 = (x) => Math.min(1, Math.max(0, x));

  function draw() {
    if (!meta) return;
    const baseY = H * BASE, amp = H * AMP;

    // The hero stage is sticky-pinned for `sweepDist` of scroll: across that span the
    // field sweeps 90->600 (text/bar/opacity locked). Only AFTER it does the stage
    // release (scroll up) and the spectrum fade to a faint persistent backdrop.
    const sweepDist = Math.max(1, hero.offsetHeight - window.innerHeight);
    const p = clamp01(window.scrollY / sweepDist);   // sweep progress while pinned
    const over = window.scrollY - sweepDist;         // px scrolled past 600 MHz
    const op = over <= 0 ? 1
      : Math.max(FADE_MIN, 1 - (1 - FADE_MIN) * (over / (window.innerHeight * FADE_VH)));
    canvas.style.opacity = op.toFixed(3);

    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(fan, 0, 0, W, H);

    const fpos = p * (curves.length - 1);
    const lo = Math.floor(fpos), hi = Math.min(curves.length - 1, lo + 1), t = fpos - lo;
    const a = curves[lo], b = curves[hi];
    const cur = new Float32Array(GRID);
    for (let k = 0; k < GRID; k++) cur[k] = a[k] * (1 - t) + b[k] * t;

    ctx.strokeStyle = colors.trace; ctx.lineWidth = 2.6; ctx.lineJoin = "round"; ctx.lineCap = "round";
    ctx.shadowColor = colors.accent; ctx.shadowBlur = colors.dark ? 20 : 6;
    curvePath(cur, ctx, amp, baseY);
    ctx.shadowBlur = 0;

    const fields = meta.fields_mhz;
    fieldVal.textContent = Math.round(fields[lo] * (1 - t) + fields[hi] * t);
    barFill.style.width = (p * 100).toFixed(1) + "%";
    scrollHint.style.opacity = p > 0.015 ? "0" : "";
  }

  /* ---------- shift+J matrix for the hero molecule ---------- */
  function buildMatrix() {
    const host = document.getElementById("matrixHost");
    if (!host || !mol) return;
    const n = mol.n_groups, J = mol.couplings || [], labels = "ABCDEFGH".slice(0, n).split("");
    let html = "<table class='mx'><tr><th></th>";
    labels.forEach(l => html += `<th>${l}</th>`); html += "<th>n</th></tr>";
    for (let i = 0; i < n; i++) {
      html += `<tr><th>${labels[i]}</th>`;
      for (let j = 0; j < n; j++) {
        if (i === j) html += `<td class="diag">${mol.shifts[i].toFixed(2)}</td>`;
        else { const v = J[i] ? J[i][j] : 0; html += `<td>${v ? v.toFixed(1) : "·"}</td>`; }
      }
      html += `<td class="deg">${mol.degeneracy[i]}</td></tr>`;
    }
    host.innerHTML = html + "</table>";
    const note = document.getElementById("repNote");
    if (note) note.innerHTML =
      `Diagonal = chemical shifts δ (ppm) of <b>${mol.chembl_id}</b>; right column = proton degeneracy <i>n</i>. ` +
      `Off-diagonal couplings <i>J</i> (Hz) drive the second-order behaviour you see above.`;
  }

  /* ---------- boot ---------- */
  let ticking = false;
  const onScroll = () => { if (!ticking) { ticking = true; requestAnimationFrame(() => { draw(); ticking = false; }); } };

  fetch("data/field_sweep.json").then(r => r.json()).then(data => {
    resize(); chooseMolecule(data); renderFan(); draw();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", () => { resize(); renderFan(); draw(); });
  }).catch(err => { console.error("field_sweep.json failed", err); molTag.textContent = "spectra failed to load"; });
})();
