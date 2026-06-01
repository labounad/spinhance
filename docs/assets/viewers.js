/* viewers.js — interactive "Where we are" explorers (vanilla, no deps).
   Three viewers: (1) learning-curve chart, (2) architecture comparison table,
   (3) held-out test-molecule prediction explorer. Reads docs/data/*.json. */
(function () {
  "use strict";
  const $ = (s, r) => (r || document).querySelector(s);
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const C = () => ({
    ink: css("--ink") || "#1a1a1a", soft: css("--ink-soft") || "#555",
    faint: css("--ink-faint") || "#999", line: css("--line") || "#e5e5e5",
    accent: css("--accent") || "#5b8def", accent3: css("--accent-3") || "#22b8a6",
    panel: css("--panel") || "#fff",
  });
  const SES = {  // colour per model in both viewers
    "022": "#9aa0a6", "025": "var-accent", "026": "var-accent3",
    "light025": "#e0a44d", "light026": "#c46be0", "xl025": "#e06b6b", "xl026": "#6bd0e0",
  };
  const colOf = (k) => { const c = C(); const v = SES[k] || c.soft;
    return v === "var-accent" ? c.accent : v === "var-accent3" ? c.accent3 : v; };

  // ---- generic responsive line plot ---------------------------------------
  function linePlot(canvas, series, opts) {
    opts = opts || {};
    const c = C(), dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth, H = canvas.clientHeight;
    canvas.width = W * dpr; canvas.height = H * dpr;
    const ctx = canvas.getContext("2d"); ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);
    const padL = opts.noY ? 16 : 52, padR = 14, padT = 12, padB = 30;
    const xs = series.flatMap(s => s.pts.map(p => p[0]));
    const ys = series.flatMap(s => s.pts.map(p => p[1]));
    if (!xs.length) return;
    let x0 = opts.x0 != null ? opts.x0 : Math.min(...xs), x1 = opts.x1 != null ? opts.x1 : Math.max(...xs);
    let y0 = opts.y0 != null ? opts.y0 : Math.min(...ys), y1 = opts.y1 != null ? opts.y1 : Math.max(...ys);
    if (y0 === y1) { y1 = y0 + 1; }
    const pad = (y1 - y0) * 0.08; y0 -= pad; y1 += pad;
    // invertX: high→low ppm (standard NMR view)
    const X = v => { const f = (v - x0) / (x1 - x0 || 1); return padL + (opts.invertX ? 1 - f : f) * (W - padL - padR); };
    const Y = v => H - padB - (v - y0) / (y1 - y0 || 1) * (H - padT - padB);
    // grid + y ticks (skipped for noY, e.g. NMR spectra where intensity is arbitrary)
    ctx.strokeStyle = c.line; ctx.fillStyle = c.faint; ctx.font = "11px system-ui,sans-serif"; ctx.lineWidth = 1;
    if (!opts.noY) for (let i = 0; i <= 4; i++) {
      const yv = y0 + (y1 - y0) * i / 4, yy = Y(yv);
      ctx.globalAlpha = .5; ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(W - padR, yy); ctx.stroke(); ctx.globalAlpha = 1;
      ctx.textAlign = "right"; ctx.fillText(yv.toFixed(opts.yd != null ? opts.yd : 2), padL - 6, yy + 3);
    }
    // x ticks
    ctx.textAlign = "center";
    for (let i = 0; i <= 4; i++) { const xv = x0 + (x1 - x0) * i / 4; ctx.fillText(Math.round(xv), X(xv), H - padB + 16); }
    ctx.fillStyle = c.soft; ctx.fillText(opts.xlabel || "epoch", (padL + W - padR) / 2, H - 4);
    // series
    series.forEach(s => {
      ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 2.2; ctx.beginPath();
      s.pts.forEach((p, i) => { const xx = X(p[0]), yy = Y(p[1]); i ? ctx.lineTo(xx, yy) : ctx.moveTo(xx, yy); });
      ctx.stroke();
      if (s.mark != null) { ctx.fillStyle = s.color; const p = s.pts[s.mark]; ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 3.5, 0, 7); ctx.fill(); }
    });
  }

  // ====================== 1. LEARNING CURVES ===============================
  function initCurves() {
    const host = $("#lcViewer"); if (!host) return;
    fetch("data/learning_curves.json").then(r => r.json()).then(data => {
      const metrics = [["shift", "shift MAE (ppm)", 3], ["j", "J MAE (Hz)", 2], ["f1", "presence F1", 2], ["deg", "deg balanced-acc", 2]];
      let metric = "shift";
      const active = new Set(Object.keys(data));
      const canvas = $("#lcCanvas", host);
      function draw() {
        const md = metrics.find(m => m[0] === metric);
        const series = Object.keys(data).filter(k => active.has(k)).map(k => ({
          color: colOf(k), pts: data[k].series.map(p => [p.epoch, p[metric]]),
        }));
        linePlot(canvas, series, { xlabel: "epoch", yd: md[2] >= 3 ? 3 : 2 });
      }
      // metric buttons
      const mbar = $("#lcMetrics", host);
      mbar.innerHTML = metrics.map(m => `<button data-m="${m[0]}"${m[0] === metric ? ' class="on"' : ''}>${m[1]}</button>`).join("");
      mbar.querySelectorAll("button").forEach(b => b.onclick = () => {
        metric = b.dataset.m; mbar.querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b)); draw();
      });
      // session legend toggles
      const leg = $("#lcLegend", host);
      leg.innerHTML = Object.keys(data).map(k =>
        `<button data-k="${k}" class="on" style="--c:${colOf(k)}"><i></i>${data[k].label}</button>`).join("");
      leg.querySelectorAll("button").forEach(b => b.onclick = () => {
        const k = b.dataset.k; active.has(k) ? active.delete(k) : active.add(k);
        b.classList.toggle("on", active.has(k)); draw();
      });
      draw(); window.addEventListener("resize", draw);
      host._draw = draw;
    }).catch(e => { host.innerHTML = "<p class='muted'>learning curves unavailable</p>"; console.error(e); });
  }

  // ====================== 2. COMPARISON TABLE ==============================
  const RUNS = [
    { m: "CNN baseline", arch: "ResNet-1D + typed heads", data: "64k ChEMBL", p: "5.0M", shift: "0.279", j: "1.80", f1: "0.807", deg: "0.732", st: "floor" },
    { m: "022", arch: "spingraph + surrogate-spectral", data: "64k ChEMBL", p: "10M", shift: "0.064", j: "0.91", f1: "0.916", deg: "0.928", st: "superseded" },
    { m: "025", arch: "spingraph, shift-2×, WSD LR", data: "64k ChEMBL", p: "10M", shift: "0.037", j: "0.59", f1: "0.940", deg: "0.945", st: "production" },
    { m: "026", arch: "025 + peak channel + soft-equiv", data: "64k ChEMBL", p: "10M", shift: "0.037", j: "0.65", f1: "0.940", deg: "0.960", st: "done" },
    { m: "light·025", arch: "025 recipe", data: "500k PubChem", p: "10M", shift: "—", j: "—", f1: "—", deg: "—", st: "running" },
    { m: "light·026", arch: "026 recipe", data: "500k PubChem", p: "10M", shift: "—", j: "—", f1: "—", deg: "—", st: "running" },
    { m: "xl·025", arch: "025 recipe, xl", data: "3.2M PubChem", p: "57M", shift: "—", j: "—", f1: "—", deg: "—", st: "running" },
    { m: "xl·026", arch: "026 recipe, xl", data: "3.2M PubChem", p: "57M", shift: "—", j: "—", f1: "—", deg: "—", st: "running" },
  ];
  function initTable() {
    const host = $("#cmpTable"); if (!host) return;
    const cols = [["m", "model"], ["arch", "architecture / recipe"], ["data", "data"], ["p", "params"],
      ["shift", "shift↓"], ["j", "J↓"], ["f1", "F1↑"], ["deg", "deg↑"], ["st", "status"]];
    let sortK = null, asc = true;
    const numK = new Set(["shift", "j", "f1", "deg"]);
    function render() {
      let rows = RUNS.slice();
      if (sortK) rows.sort((a, b) => {
        let x = a[sortK], y = b[sortK];
        if (numK.has(sortK)) { x = parseFloat(x) || 1e9; y = parseFloat(y) || 1e9; }
        return (x < y ? -1 : x > y ? 1 : 0) * (asc ? 1 : -1);
      });
      host.innerHTML = `<table class="cmp"><thead><tr>${cols.map(c =>
        `<th data-k="${c[0]}"${c[0] === sortK ? ` class="srt ${asc ? 'a' : 'd'}"` : ''}>${c[1]}</th>`).join("")}</tr></thead><tbody>${
        rows.map(r => `<tr class="st-${r.st}">${cols.map(c =>
          `<td${numK.has(c[0]) ? ' class="num"' : ''}>${r[c[0]]}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
      host.querySelectorAll("th").forEach(th => th.onclick = () => {
        const k = th.dataset.k; if (sortK === k) asc = !asc; else { sortK = k; asc = !numK.has(k); } render();
      });
    }
    render();
  }

  // ====================== 3. TEST-MOLECULE EXPLORER ========================
  function initExplorer() {
    const host = $("#txViewer"); if (!host) return;
    fetch("data/test_explorer.json").then(r => r.json()).then(data => {
      const ppm = data.ppm, mols = data.molecules; let idx = 0;
      const spec = $("#txSpec", host), sel = $("#txSel", host), meta = $("#txMeta", host), mat = $("#txMatrix", host);
      const el3d = $("#tx3d", host), load3d = $("#tx3dLoad", host); let v3d = null;
      sel.innerHTML = mols.map((m, i) => `<option value="${i}">${m.id} · ${m.n_spins}H</option>`).join("");
      function ensure3d(cb) {
        if (window.$3Dmol) return cb(window.$3Dmol);
        let n = 0; (function chk() { if (window.$3Dmol) cb(window.$3Dmol); else if (++n < 120) setTimeout(chk, 50); })();
      }
      function render3d(m) {
        if (!m || !m.xyz) { if (load3d) { load3d.textContent = "no 3D structure"; load3d.style.opacity = "1"; } if (v3d) { v3d.removeAllModels(); v3d.render(); } return; }
        ensure3d(($3) => {
          try {
            if (!v3d) v3d = $3.createViewer(el3d, { backgroundAlpha: 0 });
            v3d.removeAllModels();
            v3d.addModel(m.xyz, "xyz");                      // 3Dmol infers bonds by distance
            v3d.setStyle({}, { stick: { radius: 0.13 }, sphere: { scale: 0.24 } });
            v3d.zoomTo(); v3d.resize(); v3d.render(); v3d.spin("y", 0.6);
            if (load3d) load3d.style.opacity = "0";
          } catch (e) { if (load3d) { load3d.textContent = "3D unavailable"; load3d.style.opacity = "1"; } console.error(e); }
        });
      }
      function drawSpec() {
        const m = mols[idx], c = C();
        linePlot(spec, [
          { color: c.faint, width: 1.6, pts: m.input.map((v, i) => [ppm[i], v]) },
          { color: c.accent, width: 2, pts: m.rendered.map((v, i) => [ppm[i], v]) },
        ], { xlabel: "ppm", x0: 0, x1: 12, y0: 0, invertX: true, noY: true });
      }
      function drawMatrix() {
        const m = mols[idx]; const G = 8;
        const cell = (t, p, unit) => { const d = Math.abs((+t) - (+p)); const bad = unit === "ppm" ? d > 0.1 : d > 1.5;
          return `<td class="num">${t}</td><td class="num pred${bad ? ' off' : ''}">${p}</td>`; };
        let rows = "";
        for (let i = 0; i < G; i++)
          rows += `<tr><td class="gi">${i + 1}</td>${cell(m.true_shift[i].toFixed(2), m.pred_shift[i].toFixed(2), "ppm")}${cell(m.true_deg[i], m.pred_deg[i], "n")}</tr>`;
        // J heatmaps (true vs pred): 64 cells flattened into an 8-col grid
        const maxJ = Math.max(1, ...m.true_J.flat().map(Math.abs), ...m.pred_J.flat().map(Math.abs));
        const grid = (M) => {
          let cells = "";
          M.forEach(row => row.forEach(v => {
            const a = Math.min(1, Math.abs(v) / maxJ);
            cells += '<i style="opacity:' + a.toFixed(2) + '" title="' + v + ' Hz"></i>';
          }));
          return '<div class="jgrid">' + cells + '</div>';
        };
        mat.innerHTML =
          `<table class="nodes"><thead><tr><th>#</th><th>δ true</th><th>δ pred</th><th>n true</th><th>n pred</th></tr></thead><tbody>${rows}</tbody></table>
           <div class="jwrap"><div><div class="jlbl">J — target</div>${grid(m.true_J)}</div><div><div class="jlbl">J — predicted</div>${grid(m.pred_J)}</div></div>`;
      }
      function show() {
        const m = mols[idx]; sel.value = idx;
        meta.innerHTML = `<span class="mono">${m.smiles || m.id}</span> · ${m.n_spins} protons ·
          shift MAE <b>${m.shift_mae.toFixed(3)}</b> ppm · J MAE <b>${m.j_mae.toFixed(2)}</b> Hz`;
        drawSpec(); drawMatrix(); render3d(m);
      }
      sel.onchange = () => { idx = +sel.value; show(); };
      $("#txPrev", host).onclick = () => { idx = (idx - 1 + mols.length) % mols.length; show(); };
      $("#txNext", host).onclick = () => { idx = (idx + 1) % mols.length; show(); };
      show(); window.addEventListener("resize", drawSpec);
    }).catch(e => { host.innerHTML = "<p class='muted'>test explorer data unavailable</p>"; console.error(e); });
  }

  function boot() { initCurves(); initTable(); initExplorer(); }
  if (document.readyState !== "loading") boot(); else document.addEventListener("DOMContentLoaded", boot);
})();
