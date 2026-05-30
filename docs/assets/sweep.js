/* SpinHance — scroll-driven field-sweep hero
   Reads docs/data/field_sweep.json (base64 uint16 spectra per molecule across a
   geometric 90->600 MHz sweep) and renders a bold spectral trace + a faint fan
   of all fields onto a canvas, with the spectrometer field driven by scroll. */
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
    draw();
  });

  const cssVar = (n) => getComputedStyle(root).getPropertyValue(n).trim();
  const readColors = () => ({
    trace: cssVar("--trace"),
    fan: cssVar("--fan"),
    grid: cssVar("--grid"),
    faint: cssVar("--ink-faint"),
    accent: cssVar("--accent"),
    accent2: cssVar("--accent-2"),
    dark: root.getAttribute("data-theme") === "dark",
  });
  let colors = readColors();

  /* ---------- canvas ---------- */
  const canvas = document.getElementById("spectrum");
  const ctx = canvas.getContext("2d");
  const hero = document.getElementById("top");
  const fieldVal = document.getElementById("fieldVal");
  const barFill = document.getElementById("barFill");
  const molTag = document.getElementById("molTag");
  const scrollHint = document.getElementById("scrollHint");

  let W = 0, H = 0, dpr = 1;
  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = canvas.clientWidth; H = canvas.clientHeight;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  /* ---------- data ---------- */
  let meta = null, mol = null, frames = [];   // frames: Float32Array[] (0..1)
  let winLo = 0, winHi = 12, iLo = 0, iHi = 1; // display ppm window + index range

  function decodeFrame(b64) {
    const bin = atob(b64);
    const n = bin.length / 2;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const v = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
      out[i] = v / 65535;
    }
    return out;
  }

  function chooseMolecule(data) {
    meta = data.meta;
    mol = data.molecules[Math.floor(Math.random() * data.molecules.length)];
    frames = mol.frames.map(decodeFrame);
    // each stored frame's points span exactly the molecule's data-driven window
    [winLo, winHi] = mol.win;
    iLo = 0; iHi = meta.disp_points - 1;
    molTag.innerHTML = `<b>${mol.chembl_id || mol.id || "molecule"}</b> · <span class="mono">${(mol.smiles || "").slice(0, 40)}</span>`;
    buildMatrix(data);
  }

  /* map a sample index -> x (NMR: high ppm on the left) */
  function xOf(i) {
    const ppm = winLo + (i / (meta.disp_points - 1)) * (winHi - winLo);
    return ((winHi - ppm) / (winHi - winLo)) * W;
  }

  /* ---------- scroll progress ---------- */
  function progress() {
    const r = hero.getBoundingClientRect();
    const total = hero.offsetHeight - window.innerHeight;
    return total > 0 ? Math.min(1, Math.max(0, -r.top / total)) : 0;
  }

  /* ---------- draw ---------- */
  function tracePath(arr, amp, baseY) {
    ctx.beginPath();
    let started = false;
    for (let i = iLo; i <= iHi; i++) {
      const x = xOf(i), y = baseY - arr[i] * amp;
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  function drawAxis(baseY) {
    ctx.strokeStyle = colors.grid; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, baseY); ctx.lineTo(W, baseY); ctx.stroke();
    ctx.fillStyle = colors.faint;
    ctx.font = "12px ui-monospace,Menlo,monospace";
    ctx.textAlign = "center";
    const step = (winHi - winLo) > 6 ? 2 : (winHi - winLo) > 3 ? 1 : 0.5;
    const first = Math.ceil(winLo / step) * step;
    for (let p = first; p <= winHi + 1e-6; p += step) {
      const x = ((winHi - p) / (winHi - winLo)) * W;
      ctx.beginPath(); ctx.moveTo(x, baseY); ctx.lineTo(x, baseY + 6);
      ctx.strokeStyle = colors.grid; ctx.stroke();
      ctx.fillText(p.toFixed(step < 1 ? 1 : 0), x, baseY + 22);
    }
    ctx.textAlign = "left";
    ctx.fillText("δ  (ppm)", 14, baseY + 22);
  }

  function draw() {
    if (!meta) return;
    ctx.clearRect(0, 0, W, H);
    const baseY = H * 0.86, amp = H * 0.62;

    drawAxis(baseY);

    // faint fan: every precomputed field
    ctx.strokeStyle = colors.fan; ctx.lineWidth = 1.1;
    for (let f = 0; f < frames.length; f++) tracePath(frames[f], amp, baseY);

    // current field (interpolated between the two nearest frames)
    const p = progress();
    const fpos = p * (frames.length - 1);
    const lo = Math.floor(fpos), hi = Math.min(frames.length - 1, lo + 1), t = fpos - lo;
    const a = frames[lo], b = frames[hi];
    const cur = new Float32Array(meta.disp_points);
    for (let i = iLo; i <= iHi; i++) cur[i] = a[i] * (1 - t) + b[i] * t;

    ctx.strokeStyle = colors.trace;
    ctx.lineWidth = 2.7; ctx.lineJoin = "round"; ctx.lineCap = "round";
    ctx.shadowColor = colors.accent; ctx.shadowBlur = colors.dark ? 22 : 8;
    tracePath(cur, amp, baseY);
    ctx.shadowBlur = 0;

    // readouts
    const fields = meta.fields_mhz;
    const mhz = fields[lo] * (1 - t) + fields[hi] * t;
    fieldVal.textContent = Math.round(mhz);
    barFill.style.width = (p * 100).toFixed(1) + "%";
    scrollHint.style.opacity = p > 0.02 ? "0" : "";
  }

  /* ---------- shift+J matrix table for the hero molecule ---------- */
  function buildMatrix(data) {
    const host = document.getElementById("matrixHost");
    if (!host || !mol) return;
    const n = mol.n_groups;
    const J = mol.couplings || [];
    const labels = "ABCDEFGH".slice(0, n).split("");
    let html = "<table class='mx'><tr><th></th>";
    labels.forEach(l => html += `<th>${l}</th>`);
    html += "<th>n</th></tr>";
    for (let i = 0; i < n; i++) {
      html += `<tr><th>${labels[i]}</th>`;
      for (let j = 0; j < n; j++) {
        if (i === j) html += `<td class="diag">${mol.shifts[i].toFixed(2)}</td>`;
        else {
          const v = J[i] ? J[i][j] : 0;
          html += `<td>${v ? v.toFixed(1) : "·"}</td>`;
        }
      }
      html += `<td class="deg">${mol.degeneracy[i]}</td></tr>`;
    }
    html += "</table>";
    host.innerHTML = html;
    const note = document.getElementById("repNote");
    if (note) note.innerHTML =
      `Diagonal = chemical shifts δ (ppm) of <b>${mol.chembl_id}</b>; right column = proton degeneracy <i>n</i>. ` +
      `Off-diagonal couplings <i>J</i> (Hz) drive the second-order behaviour you see above.`;
  }

  /* ---------- boot ---------- */
  let ticking = false;
  function onScroll() { if (!ticking) { ticking = true; requestAnimationFrame(() => { draw(); ticking = false; }); } }

  fetch("data/field_sweep.json")
    .then(r => r.json())
    .then(data => {
      document.getElementById("statMol").textContent = data.molecules.length >= 30 ? "1,072" : data.molecules.length;
      resize();
      chooseMolecule(data);
      draw();
      window.addEventListener("scroll", onScroll, { passive: true });
      window.addEventListener("resize", () => { resize(); draw(); });
    })
    .catch(err => {
      console.error("field_sweep.json failed to load", err);
      molTag.textContent = "spectra failed to load";
    });
})();
