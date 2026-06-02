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
  let meta = null, mol = null, winLo = 0, winHi = 1;
  let frameCurves = [];      // raw broadened curve per field frame (for the static fan)
  let morphs = [];           // matched line pairs between adjacent frames (for the trace)
  let globalScale = 1;       // single height scale across all frames (no per-frame re-pinning)
  let bufC = null, bufA = null, bufCur = null;   // reused scratch buffers (no per-scroll alloc)

  const b64f32 = (s) => { const b = atob(s), n = b.length / 4, u = new Uint8Array(b.length);
    for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new Float32Array(u.buffer, 0, n); };
  const b64u16 = (s) => { const b = atob(s), n = b.length / 2, out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = (b.charCodeAt(2*i) | (b.charCodeAt(2*i+1) << 8)) / 65535; return out; };

  /* Broaden a stick list into a pseudo-Voigt curve on the grid, RAW (no per-frame
     normalization — heights are made consistent across fields by a single global
     scale, see chooseMolecule). Real lines are Voigt (Lorentzian T2 (x) Gaussian
     B0-inhomogeneity); ETA is the Lorentzian fraction (~0.8, matches the simulator).
     `broadenInto` writes into a reused buffer so the scroll path allocates nothing. */
  const ETA = 0.8, LN2 = Math.log(2);
  function broadenInto(centers, amps, len, hwhm, out) {
    out.fill(0);
    const dppm = (winHi - winLo) / (GRID - 1);
    const cutoff = Math.max(30 * hwhm, dppm * 3);
    for (let i = 0; i < len; i++) {
      const a = amps[i];
      if (a <= 0) continue;
      const c = centers[i];
      let k0 = Math.floor((c - cutoff - winLo) / dppm), k1 = Math.ceil((c + cutoff - winLo) / dppm);
      if (k0 < 0) k0 = 0; if (k1 > GRID - 1) k1 = GRID - 1;
      for (let k = k0; k <= k1; k++) {
        const d = (winLo + k * dppm - c) / hwhm;
        const d2 = d * d;
        out[k] += a * (ETA / (1 + d2) + (1 - ETA) * Math.exp(-LN2 * d2));  // pseudo-Voigt
      }
    }
    return out;
  }
  const broadenSticks = (c, a, hwhm) => broadenInto(c, a, c.length, hwhm, new Float32Array(GRID));

  /* Decode one frame's base64 sticks and sort by ppm (needed for the line matcher). */
  function decodeSorted(fr) {
    const c = b64f32(fr.c), a = b64u16(fr.a), n = c.length;
    const idx = Array.from({ length: n }, (_, k) => k).sort((p, q) => c[p] - c[q]);
    const cs = new Float32Array(n), as = new Float32Array(n);
    for (let k = 0; k < n; k++) { cs[k] = c[idx[k]]; as[k] = a[idx[k]]; }
    return { c: cs, a: as };
  }

  /* Match lines between two adjacent frames so the sweep MORPHS peaks (centers +
     amps interpolate, peaks translate) instead of cross-fading two finished
     curves — which made peaks sag/bounce mid-interpolation. Unmatched lines fade
     in place (birth/death), so the line count can change.

     GREEDY-NEAREST bipartite match (not an in-order two-pointer): collect all
     candidate pairs within TOL, assign by increasing distance with each line used
     once. This is robust to merges/splits — when two lines merge, the survivor
     keeps its near match and the other simply dies, instead of the two-pointer
     mis-pairing every downstream line by one position (the cascade that made a
     whole cluster of aromatic peaks slide ~0.03 ppm and jump in height). */
  const MATCH_TOL = 0.02;   // ppm cap on a match; beyond this a line is a birth/death
  function matchPair(lc, la, hc, ha) {
    const triples = [];     // candidate [distance, i, j] pairs within TOL (built once at load)
    for (let i = 0; i < lc.length; i++)
      for (let j = 0; j < hc.length; j++) {
        const dij = Math.abs(lc[i] - hc[j]);
        if (dij <= MATCH_TOL) triples.push([dij, i, j]);
      }
    triples.sort((p, q) => p[0] - q[0]);
    const usedI = new Uint8Array(lc.length), usedJ = new Uint8Array(hc.length);
    const cLo = [], aLo = [], cHi = [], aHi = [];
    for (const [, i, j] of triples) {
      if (usedI[i] || usedJ[j]) continue;
      usedI[i] = 1; usedJ[j] = 1;
      cLo.push(lc[i]); aLo.push(la[i]); cHi.push(hc[j]); aHi.push(ha[j]);   // matched
    }
    for (let i = 0; i < lc.length; i++) if (!usedI[i]) { cLo.push(lc[i]); aLo.push(la[i]); cHi.push(lc[i]); aHi.push(0); }  // death
    for (let j = 0; j < hc.length; j++) if (!usedJ[j]) { cLo.push(hc[j]); aLo.push(0); cHi.push(hc[j]); aHi.push(ha[j]); }  // birth
    return { cLo: Float32Array.from(cLo), aLo: Float32Array.from(aLo),
             cHi: Float32Array.from(cHi), aHi: Float32Array.from(aHi) };
  }

  function chooseMolecule(data) {
    meta = data.meta;
    mol = data.molecules[Math.floor(Math.random() * data.molecules.length)];
    [winLo, winHi] = mol.win;
    // decode + sort sticks per frame, broaden each frame raw (for the fan), and
    // precompute the line matching between adjacent frames (for the morph).
    const sticks = mol.frames.map(decodeSorted);
    frameCurves = sticks.map((s, idx) =>
      broadenSticks(s.c, s.a, (meta.linewidth_hz / 2) / meta.fields_mhz[idx]));
    // one global height scale (max over all frames) -> heights vary smoothly &
    // physically instead of every frame re-pinning its tallest peak to 1.
    globalScale = 0;
    for (const cv of frameCurves) for (let k = 0; k < GRID; k++) if (cv[k] > globalScale) globalScale = cv[k];
    if (!(globalScale > 0)) globalScale = 1;
    morphs = []; let maxLen = 0;
    for (let i = 0; i < sticks.length - 1; i++) {
      const m = matchPair(sticks[i].c, sticks[i].a, sticks[i + 1].c, sticks[i + 1].a);
      morphs.push(m); if (m.cLo.length > maxLen) maxLen = m.cLo.length;
    }
    bufC = new Float32Array(maxLen); bufA = new Float32Array(maxLen); bufCur = new Float32Array(GRID);
    molTag.innerHTML = `<b>${mol.chembl_id || mol.id || "molecule"}</b> &nbsp;` +
      `<span class="mono smi" id="smilesCopy" title="Click to copy SMILES">${mol.smiles || ""}</span>` +
      `<span class="copied" id="copiedMsg" style="opacity:0">✓ copied</span>`;
    buildMatrix();
    // hand the molecule to the 3D viewer and spin graph modules
    window.__heroMol = {
      smiles: mol.smiles, id: mol.chembl_id || mol.id, xyz: mol.xyz,
      shifts: mol.shifts, couplings: mol.couplings,
      degeneracy: mol.degeneracy, n_groups: mol.n_groups,
    };
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
    g.textAlign = "left"; g.fillText("δ (ppm)", 12, baseY + 34);
  }

  /* fan + axis are static per molecule/size/theme -> render once to offscreen */
  function renderFan() {
    if (!meta) return;
    const baseY = H * BASE, amp = H * AMP;
    fctx.clearRect(0, 0, W, H);
    drawAxis(fctx, baseY);
    fctx.strokeStyle = colors.fan; fctx.lineWidth = 1.1; fctx.lineJoin = "round";
    for (const c of frameCurves) curvePath(c, fctx, amp / globalScale, baseY);
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

    const n = frameCurves.length;
    const fpos = p * (n - 1);
    const lo = Math.floor(fpos), hi = Math.min(n - 1, lo + 1), t = fpos - lo;

    // Morph in PEAK space: interpolate matched line centers+amps, broaden ONCE at
    // the continuous current field. Peaks translate/merge smoothly (no cross-fade
    // sag). At t=0/1 this reproduces a frame exactly, so it's seamless with the fan.
    let cur;
    if (lo < morphs.length) {
      const m = morphs[lo], L = m.cLo.length, s = 1 - t;
      for (let i = 0; i < L; i++) { bufC[i] = m.cLo[i] * s + m.cHi[i] * t; bufA[i] = m.aLo[i] * s + m.aHi[i] * t; }
      const fcur = meta.fields_mhz[lo] * s + meta.fields_mhz[hi] * t;
      cur = broadenInto(bufC, bufA, L, (meta.linewidth_hz / 2) / fcur, bufCur);
    } else {
      cur = frameCurves[lo];      // exactly on the last frame
    }

    ctx.strokeStyle = colors.trace; ctx.lineWidth = 2.6; ctx.lineJoin = "round"; ctx.lineCap = "round";
    ctx.shadowColor = colors.accent; ctx.shadowBlur = colors.dark ? 20 : 6;
    curvePath(cur, ctx, amp / globalScale, baseY);
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

    // find max |J| for symmetric color scaling (mirrors GUI's zmin/zmax)
    let maxJ = 0;
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++)
        if (i !== j && J[i] && Math.abs(J[i][j]) > maxJ) maxJ = Math.abs(J[i][j]);

    // RdBu-like diverging color: blue for +J, red for -J, transparent near 0
    function jBg(v) {
      if (!maxJ || Math.abs(v) < 0.3) return "";
      const t = Math.abs(v) / maxJ;
      const a = (0.12 + t * 0.50).toFixed(2);
      return v > 0 ? `rgba(59,130,246,${a})` : `rgba(220,38,38,${a})`;
    }
    // shift diagonal: blue tint scaled to the shift's position in [0,12] ppm
    function shiftBg(s) {
      const t = Math.max(0, Math.min(1, s / 10));
      return `rgba(34,193,195,${(0.08 + t * 0.22).toFixed(2)})`;
    }

    let html = "<table class='mx'><tr><th></th>";
    labels.forEach(l => html += `<th>${l}</th>`); html += "<th>n</th></tr>";
    for (let i = 0; i < n; i++) {
      html += `<tr><th>${labels[i]}</th>`;
      for (let j2 = 0; j2 < n; j2++) {
        if (i === j2) {
          const bg = shiftBg(mol.shifts[i]);
          html += `<td class="diag" style="background:${bg}">${mol.shifts[i].toFixed(2)}</td>`;
        } else {
          const v = J[i] ? J[i][j2] : 0;
          const bg = jBg(v);
          const style = bg ? ` style="background:${bg}"` : "";
          const cls = Math.abs(v) < 0.3 ? " class='zero'" : "";
          html += `<td${cls}${style}>${Math.abs(v) >= 0.3 ? v.toFixed(1) : "·"}</td>`;
        }
      }
      html += `<td class="deg">${mol.degeneracy[i]}</td></tr>`;
    }
    host.innerHTML = html + "</table>";
    const note = document.getElementById("repNote");
    if (note) note.innerHTML =
      `Diagonal = chemical shifts δ (ppm) of <b>${mol.chembl_id}</b>; right column = proton degeneracy <i>n</i>. ` +
      `Off-diagonal: <span style="color:rgba(59,130,246,0.85)">blue = positive J</span>, ` +
      `<span style="color:rgba(220,38,38,0.85)">red = negative J</span> (Hz). ` +
      `Only |J| &gt; 0.3 Hz shown.`;
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
