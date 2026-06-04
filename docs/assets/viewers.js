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
  const SES = {  // explicit colour per fleet key; others fall back to PALETTE
    "baseline": "#9aa0a6", "64k": "var-accent", "500k": "var-accent3", "3M": "#b07bff",
  };
  // deterministic distinct colours for arbitrary keys (e.g. "64k_027", "500k_027")
  const PALETTE = ["#5b8cff", "#34e3c4", "#b07bff", "#e0a44d", "#7ee06b", "#e06b6b", "#6bd0e0", "#c46be0", "#f5a623", "#e3d34e"];
  const hashIdx = (k) => { let h = 0; for (let i = 0; i < k.length; i++) h = (h * 31 + k.charCodeAt(i)) >>> 0; return h % PALETTE.length; };
  const colOf = (k) => { const c = C(); const v = SES[k];
    if (!v) return PALETTE[hashIdx(k)];
    return v === "var-accent" ? c.accent : v === "var-accent3" ? c.accent3 : v; };

  // Stroke a series either as straight segments or, when `smooth`, as a monotone cubic
  // (Fritsch–Carlson) spline: smooth AND shape-preserving — no overshoot above peaks or
  // dips below baseline (unlike Catmull-Rom/bezier ringing). Assumes x ascending.
  function strokeSeries(ctx, pts, X, Y, smooth) {
    const n = pts.length;
    if (!smooth || n < 3) {
      pts.forEach((p, i) => { const xx = X(p[0]), yy = Y(p[1]); i ? ctx.lineTo(xx, yy) : ctx.moveTo(xx, yy); });
      return;
    }
    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
    const dx = [], sl = [];
    for (let i = 0; i < n - 1; i++) { dx[i] = xs[i + 1] - xs[i]; sl[i] = dx[i] ? (ys[i + 1] - ys[i]) / dx[i] : 0; }
    const m = new Array(n);
    m[0] = sl[0]; m[n - 1] = sl[n - 2];
    for (let i = 1; i < n - 1; i++) {
      if (sl[i - 1] * sl[i] <= 0) m[i] = 0;          // local extremum -> flat tangent (kills overshoot)
      else { const w1 = 2 * dx[i] + dx[i - 1], w2 = dx[i] + 2 * dx[i - 1]; m[i] = (w1 + w2) / (w1 / sl[i - 1] + w2 / sl[i]); }
    }
    ctx.moveTo(X(xs[0]), Y(ys[0]));
    for (let i = 0; i < n - 1; i++) {                // Hermite -> cubic bezier control points
      const c1x = xs[i] + dx[i] / 3, c1y = ys[i] + m[i] * dx[i] / 3;
      const c2x = xs[i + 1] - dx[i] / 3, c2y = ys[i + 1] - m[i + 1] * dx[i] / 3;
      ctx.bezierCurveTo(X(c1x), Y(c1y), X(c2x), Y(c2y), X(xs[i + 1]), Y(ys[i + 1]));
    }
  }

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
    for (let i = 0; i <= 4; i++) { const xv = x0 + (x1 - x0) * i / 4; ctx.fillText(opts.xdp != null ? xv.toFixed(opts.xdp) : Math.round(xv), X(xv), H - padB + 16); }
    ctx.fillStyle = c.soft; ctx.fillText(opts.xlabel || "epoch", (padL + W - padR) / 2, H - 4);
    // series (s.alpha gives a per-line opacity so overlapping traces stay readable)
    series.forEach(s => {
      ctx.globalAlpha = s.alpha != null ? s.alpha : 1;
      ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 2.2; ctx.beginPath();
      strokeSeries(ctx, s.pts, X, Y, opts.smooth);   // smooth = monotone-cubic spline (opt-in)
      ctx.stroke();
      if (s.mark != null) { ctx.fillStyle = s.color; const p = s.pts[s.mark]; ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 3.5, 0, 7); ctx.fill(); }
      ctx.globalAlpha = 1;
    });
  }

  // ====================== 1. LEARNING CURVES ===============================
  function initCurves() {
    const host = $("#lcViewer"); if (!host) return;
    fetch("data/learning_curves.json").then(r => r.json()).then(data => {
      const metrics = [["shift", "shift MAE (ppm)", 3], ["j", "J MAE (Hz)", 2], ["f1", "presence F1", 2], ["deg", "deg balanced-acc", 2]];
      let metric = "shift";
      const active = new Set(Object.keys(data));
      // distinct colour per curve: explicit SES if defined, else palette by index
      const cmap = {}; Object.keys(data).forEach((k, i) => { cmap[k] = SES[k] ? colOf(k) : PALETTE[i % PALETTE.length]; });
      const canvas = $("#lcCanvas", host);
      function draw() {
        const md = metrics.find(m => m[0] === metric);
        const series = Object.keys(data).filter(k => active.has(k)).map(k => ({
          color: cmap[k], pts: data[k].series.map(p => [p.epoch, p[metric]]),
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
        `<button data-k="${k}" class="on" title="click: show only this (kept pins stay) · double-click: pin/unpin" style="--c:${cmap[k]}"><i></i>${data[k].label}</button>`).join("");
      // Selection = pinned set (double-click toggles, persistent) ∪ the last single-
      // clicked "solo". Single click loads one but leaves pins alone; with nothing
      // selected, all curves show. `active` (used by draw) is recomputed from these.
      const pinned = new Set();
      let solo = null;
      const recompute = () => {
        active.clear();
        pinned.forEach(k => active.add(k));
        if (solo) active.add(solo);
        if (active.size === 0) Object.keys(data).forEach(k => active.add(k));
      };
      const syncLegend = () => leg.querySelectorAll("button").forEach(b => {
        b.classList.toggle("on", active.has(b.dataset.k));
        b.classList.toggle("pinned", pinned.has(b.dataset.k));
      });
      let pend = null;
      leg.querySelectorAll("button").forEach(b => b.onclick = () => {
        const k = b.dataset.k;
        if (pend && pend.k === k) {                       // 2nd click on same button = double -> pin/unpin
          clearTimeout(pend.timer); pend = null;
          pinned.has(k) ? pinned.delete(k) : pinned.add(k);
          recompute(); syncLegend(); draw();
        } else {
          if (pend) clearTimeout(pend.timer);
          pend = { k, timer: setTimeout(() => {            // single click -> load only this; pins persist
            pend = null; solo = k; recompute(); syncLegend(); draw();
          }, 200) };
        }
      });
      draw(); window.addEventListener("resize", draw);
      host._draw = draw;
    }).catch(e => { host.innerHTML = "<p class='muted'>learning curves unavailable</p>"; console.error(e); });
  }

  // ====================== 2. COMPARISON TABLE (validation) =================
  // Data-driven from data/test_eval.json (the corrected-data fleet). Validation metrics;
  // auto-fills as tiers finish (running tiers show "—"). The held-out TEST view is below.
  const RECIPE_DESC = { "025": "matrix (shift 2×)", "026": "+ peak-channel + soft-equiv",
    "027": "+ focal loss", "028": "+ cum-integral channel", "029": "026 + focal",
    "030": "super (026+027+028)" };
  const FLEET_ORDER = ["64k_025", "64k_026", "64k_027", "64k_028", "64k_029", "64k_030",
    "500k_026", "3M_026"];
  function initTable() {
    const host = $("#cmpTable"); if (!host) return;
    fetch("data/test_eval.json").then(r => r.json()).then(d => {
      const fmt = (v, n) => (v == null ? "—" : (+v).toFixed(n));
      const RUNS = [{ m: "CNN baseline", arch: "ResNet-1D + typed heads", data: "ChEMBL", p: "5.0M",
                      shift: "—", j: "—", f1: "—", deg: "—", st: "floor" }];
      FLEET_ORDER.forEach(k => { const e = d[k]; if (!e) return;
        const fin = e.state === "finished" && e.val;
        RUNS.push({ m: e.tier + " · " + e.recipe,
          arch: "spingraph_decoder · " + (RECIPE_DESC[e.recipe] || e.recipe),
          data: e.tier + " PubChem", p: e.params,
          shift: fin ? fmt(e.val.shift_mae_ppm, 3) : "—", j: fin ? fmt(e.val.j_mae_hz, 2) : "—",
          f1: fin ? fmt(e.val.presence_f1, 3) : "—", deg: fin ? fmt(e.val.deg_acc_balanced, 3) : "—",
          st: fin ? "trained" : "running" }); });
      const cols = [["m", "model"], ["arch", "architecture / recipe"], ["data", "data"], ["p", "params"],
        ["shift", "shift↓"], ["j", "J↓"], ["f1", "F1↑"], ["deg", "deg↑"], ["st", "status"]];
      let sortK = null, asc = true;
      const numK = new Set(["shift", "j", "f1", "deg"]);
      const lowerBetter = new Set(["shift", "j"]);   // others (f1, deg) higher = better
      const best = {};
      numK.forEach(k => {
        const vals = RUNS.map(r => parseFloat(r[k])).filter(v => !isNaN(v));
        if (vals.length) best[k] = (lowerBetter.has(k) ? Math.min : Math.max).apply(null, vals);
      });
      function render() {
        let rows = RUNS.slice();
        if (sortK) rows.sort((a, b) => {
          let x = a[sortK], y = b[sortK];
          if (numK.has(sortK)) { x = parseFloat(x) || 1e9; y = parseFloat(y) || 1e9; }
          return (x < y ? -1 : x > y ? 1 : 0) * (asc ? 1 : -1);
        });
        host.innerHTML = `<table class="cmp"><thead><tr>${cols.map(c =>
          `<th data-k="${c[0]}"${c[0] === sortK ? ` class="srt ${asc ? 'a' : 'd'}"` : ''}>${c[1]}</th>`).join("")}</tr></thead><tbody>${
          rows.map(r => `<tr class="st-${r.st}">${cols.map(c => {
            const isBest = numK.has(c[0]) && parseFloat(r[c[0]]) === best[c[0]];
            return `<td${numK.has(c[0]) ? ` class="num${isBest ? ' best' : ''}"` : ''}>${r[c[0]]}</td>`;
          }).join("")}</tr>`).join("")}</tbody></table>`;
        host.querySelectorAll("th").forEach(th => th.onclick = () => {
          const k = th.dataset.k; if (sortK === k) asc = !asc; else { sortK = k; asc = !numK.has(k); } render();
        });
      }
      render();
    }).catch(e => { host.innerHTML = "<p class='muted'>comparison unavailable</p>"; console.error(e); });
  }

  // ====================== 3. TEST-MOLECULE EXPLORER ========================
  function initExplorer() {
    const host = $("#txViewer"); if (!host) return;
    fetch("data/test_explorer.json").then(r => r.json()).then(data => {
      const mols = data.molecules; let idx = 0;
      // per-molecule ppm axis: each molecule stores its own [x0,x1] window (higher resolution
      // than a shared 0-12 axis). ppmOf(m) reconstructs the axis from the window + sample count.
      const winOf = (m) => [m.x0 != null ? m.x0 : 0, m.x1 != null ? m.x1 : 12];
      // Each spectrum ships its own adaptive mesh x (m.ix for the input, preds[k].rx for a
      // model render) — points clustered at peaks, sparse on baseline; the line plot
      // interpolates linearly between them. Old data without a mesh falls back to a uniform axis.
      const ppmOf = (m) => { const [a, b] = winOf(m), n = m.input.length;
        return m.input.map((_, i) => a + i * (b - a) / (n - 1)); };
      if (!mols || !mols.length) {          // held-out explorer not generated yet (tier still training)
        host.innerHTML = '<p class="muted">' + (data.note ||
          "The held-out test-molecule explorer is generated once a tier finishes training — coming shortly.") + '</p>';
        return;
      }
      // per-model predictions + per-model test-split membership
      const models = data.models || [{ key: "_", label: "model" }];
      let mkey = (models.find(x => x.key === "64k_026") || models[0]).key;   // default: 64k·026 (production recipe)
      const labelOf = (k) => (models.find(x => x.key === k) || {}).label || k;
      const inTest = () => true;   // every molecule is in the shared global held-out set
      const predOf = (m) => (m.preds ? m.preds[mkey] : m);
      const spec = $("#txSpec", host), sel = $("#txSel", host), meta = $("#txMeta", host), mat = $("#txMatrix", host);
      const modSel = $("#txModel", host), statusEl = $("#txStatus", host);
      const vis = { target: true, pred: true, refined: true };   // per-trace visibility (legend toggles)
      const LINE_ALPHA = 0.78;                                    // slight transparency so overlaps read clearly
      const el3d = $("#tx3d", host), load3d = $("#tx3dLoad", host), spinCb = $("#txSpin", host); let v3d = null;
      function applySpin() { if (v3d) v3d.spin(spinCb && spinCb.checked ? "y" : false, 0.6); }
      if (spinCb) spinCb.addEventListener("change", applySpin);
      if (modSel) { modSel.innerHTML = models.map(mm => `<option value="${mm.key}">${mm.label}</option>`).join(""); modSel.value = mkey; }
      const visibleIdx = () => mols.map((m, i) => i);
      function rebuildMolOptions() {
        const vis = visibleIdx();
        sel.innerHTML = vis.map(i => { const m = mols[i];
          return `<option value="${i}">${m.id} · ${m.n_spins}H</option>`; }).join("");
        if (!vis.includes(idx)) idx = vis.length ? vis[0] : 0;
        sel.value = idx;
      }
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
            v3d.zoomTo(); v3d.resize(); v3d.render(); applySpin();
            if (load3d) load3d.style.opacity = "0";
          } catch (e) { if (load3d) { load3d.textContent = "3D unavailable"; load3d.style.opacity = "1"; } console.error(e); }
        });
      }
      // Interactive zoom state for the spectrum plot: x-range (ppm) + y-stretch factor.
      let xr0 = 0, xr1 = 12, yZoom = 1;
      const PADL = 16, PADR = 14, PADT = 12, PADB = 30;   // mirror linePlot's noY paddings
      const showRef = (P) => vis.refined && P.refined && !P.ref_skipped;   // refined available + toggled on
      function drawSpec() {
        const m = mols[idx], P = predOf(m), c = C();
        const ix = m.ix || ppmOf(m), rx = (P && P.rx) || ppmOf(m);   // per-spectrum adaptive mesh x
        // target-spectrum colour encodes test membership: teal = held-out test for this model, amber = out-of-distribution
        const tgt = inTest(m) ? (css("--accent-2") || "#34e3c4") : "#f5a623";
        const ref = showRef(P);
        // y-scale spans only the visible traces, so a soloed line still fills the plot
        const dmax = Math.max(1e-6, ...(vis.target ? m.input : []), ...(vis.pred ? P.rendered : []),
                              ...(ref ? P.refined : []));
        const series = [];
        if (vis.target) series.push({ color: tgt, width: 1.8, alpha: LINE_ALPHA, pts: m.input.map((v, i) => [ix[i], v]) });
        if (vis.pred) series.push({ color: c.accent, width: 2, alpha: LINE_ALPHA, pts: P.rendered.map((v, i) => [rx[i], v]) });
        if (ref) { const fx = P.fx || ppmOf(m);                       // violet, drawn on top
          series.push({ color: "#b07bff", width: 2, alpha: LINE_ALPHA, pts: P.refined.map((v, i) => [fx[i], v]) }); }
        linePlot(spec, series, { xlabel: "ppm", x0: xr0, x1: xr1, y0: 0, y1: dmax / yZoom,
             invertX: true, noY: true, smooth: true, xdp: (xr1 - xr0) < 6 ? 1 : 0 });
      }

      // --- plot interactions: scroll = stretch Y, click-drag horizontally = zoom X, dbl-click = reset ---
      const evX = (e) => e.clientX - spec.getBoundingClientRect().left;       // client x within the canvas
      const pxToPpm = (px) => {                                              // invert linePlot's invertX mapping
        const plotW = spec.clientWidth - PADL - PADR;
        const f = Math.max(0, Math.min(1, 1 - (px - PADL) / (plotW || 1)));
        return xr0 + f * (xr1 - xr0);
      };
      spec.addEventListener("wheel", (e) => {                                // scroll -> stretch the intensity axis
        e.preventDefault();
        // gentle, proportional to scroll delta (handles wheel notches + trackpads); was too sensitive
        yZoom = Math.max(1, Math.min(60, yZoom * Math.exp(-e.deltaY * 0.0006)));
        drawSpec();
      }, { passive: false });
      let dragX0 = null;
      spec.addEventListener("mousedown", (e) => { dragX0 = evX(e); });
      spec.addEventListener("mousemove", (e) => {                            // live selection band while dragging
        if (dragX0 == null) return;
        drawSpec();
        const x = evX(e), ctx = spec.getContext("2d");
        ctx.save(); ctx.fillStyle = "rgba(91,140,255,0.18)";
        ctx.fillRect(Math.min(dragX0, x), PADT, Math.abs(x - dragX0), spec.clientHeight - PADT - PADB);
        ctx.restore();
      });
      const endDrag = (e) => {
        if (dragX0 == null) return;
        const x = evX(e);
        if (Math.abs(x - dragX0) > 4) {                                      // ignore tiny drags (treat as clicks)
          let a = pxToPpm(dragX0), b = pxToPpm(x);
          xr0 = Math.min(a, b); xr1 = Math.max(a, b);
          if (xr1 - xr0 < 0.05) xr1 = xr0 + 0.05;                            // floor the zoom width
        }
        dragX0 = null; drawSpec();
      };
      spec.addEventListener("mouseup", endDrag);
      spec.addEventListener("mouseleave", endDrag);
      spec.addEventListener("dblclick", () => { [xr0, xr1] = winOf(mols[idx]); yZoom = 1; drawSpec(); });
      spec.style.cursor = "ew-resize";
      { const lg = $(".tx-legend", host);
        if (lg) lg.insertAdjacentHTML("beforeend",
          '<span style="color:var(--ink-faint);font-size:11px">scroll: stretch Y · drag: zoom X · dbl-click: reset</span>'); }
      function drawMatrix() {
        const m = mols[idx], P = predOf(m); const G = 8;
        const cell = (t, p, unit) => { const d = Math.abs((+t) - (+p)); const bad = unit === "ppm" ? d > 0.1 : d > 1.5;
          return `<td class="num">${t}</td><td class="num pred${bad ? ' off' : ''}">${p}</td>`; };
        let rows = "";
        for (let i = 0; i < G; i++)
          rows += `<tr><td class="gi">${i + 1}</td>${cell(m.true_shift[i].toFixed(2), P.pred_shift[i].toFixed(2), "ppm")}${cell(m.true_deg[i], P.pred_deg[i], "n")}</tr>`;
        // J heatmaps (true vs pred): 64 cells flattened into an 8-col grid
        const maxJ = Math.max(1, ...m.true_J.flat().map(Math.abs), ...P.pred_J.flat().map(Math.abs));
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
           <div class="jwrap"><div><div class="jlbl">J — target</div>${grid(m.true_J)}</div><div><div class="jlbl">J — predicted</div>${grid(P.pred_J)}</div></div>`;
      }
      function setMeta() {                         // reflects the current refined-toggle state
        const m = mols[idx], P = predOf(m);
        const hasRef = showRef(P) && P.ref_shift_mae != null;
        const shiftTxt = hasRef
          ? `shift MAE <b>${P.shift_mae.toFixed(3)}</b> → <b style="color:#b07bff">${P.ref_shift_mae.toFixed(3)}</b> ppm <span class="mono" style="color:var(--ink-faint)">(refined)</span>`
          : `shift MAE <b>${P.shift_mae.toFixed(3)}</b> ppm`;
        meta.innerHTML = `<span class="mono">${m.smiles || m.id}</span> · ${m.n_spins} protons · ${shiftTxt} · J MAE <b>${P.j_mae.toFixed(2)}</b> Hz`;
      }
      function show() {
        const m = mols[idx], P = predOf(m); sel.value = idx;
        [xr0, xr1] = winOf(m); yZoom = 1;          // reset zoom to this molecule's window on navigation
        if (statusEl) {
          statusEl.className = "tx-status is-test";
          statusEl.innerHTML = `● held-out test molecule — no model trained on it · predictions from <b>${labelOf(mkey)}</b>`;
        }
        setMeta(); drawSpec(); drawMatrix(); render3d(m);
      }
      function step(d) { const vis = visibleIdx(); let p = vis.indexOf(idx); if (p < 0) p = 0;
        p = (p + d + vis.length) % vis.length; idx = vis[p]; show(); }
      sel.onchange = () => { idx = +sel.value; show(); };
      if (modSel) modSel.onchange = () => { mkey = modSel.value; rebuildMolOptions(); show(); };
      // legend = per-trace toggles (target / prediction / refined); update plot + meta in place
      host.querySelectorAll(".tx-leg").forEach(b => b.onclick = () => {
        const s = b.dataset.s; vis[s] = !vis[s];
        b.classList.toggle("off", !vis[s]); drawSpec(); setMeta();
      });
      $("#txPrev", host).onclick = () => step(-1);
      $("#txNext", host).onclick = () => step(1);
      idx = 0;
      rebuildMolOptions(); show(); window.addEventListener("resize", drawSpec);
    }).catch(e => { host.innerHTML = "<p class='muted'>test explorer data unavailable</p>"; console.error(e); });
  }

  // ====================== 4. MOLECULE BROWSER (3Dmol.js) ===================
  function initSpinViewer() {
    var host      = $("#svViewer");     if (!host) return;
    var tileCont  = $("#svTiles",      host);
    var molBox    = $("#svMolBox",     host);
    var headId    = $("#svId",         host);
    var headSub   = $("#svSub",        host);
    var tableBody = $("#svTableBody",  host);
    var smilesVal = $("#svSmiles",     host);
    var mols = [], current = null, hoverLabel = null;
    var overlay, ctx;

    // All molecules display at this fixed scale (px per native SVG pixel).
    // ChemDraw exports all molecules at the same bond length in SVG px, so
    // a single PIXEL_SCALE makes every structure appear at the same bond size.
    var PIXEL_SCALE = 2.5;

    function svgUri(svg) {
      return "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg);
    }

    // ---- Canvas overlay (spotlight effect) ----
    // All coordinates are in PIXEL space (canvas buffer px = molBox display px).
    // Atom positions (p[0], p[1]) are fractions of the ORIGINAL SVG viewBox;
    // we convert using imgW/imgH (displayed image dimensions) + offset for centering.

    function atomPx(p) {
      // Returns {x, y} in canvas pixel space, using the clamped display dimensions
      if (!current || !current._dispW) return {x:0, y:0};
      var boxW = overlay.width, boxH = overlay.height;
      var offX = (boxW - current._dispW) / 2;
      var offY = (boxH - current._dispH) / 2;
      return {x: offX + p[0] * current._dispW, y: offY + p[1] * current._dispH};   // RDKit GetDrawCoords are exact atom centres (no ChemDraw label-anchor offset)
    }

    // Diastereotopic CH2 protons are two groups on the SAME carbon, so their
    // circles overlap. Each such group carries a unit `off` vector (opposite signs
    // for the pair); nudge its circle a few px along it so the two are separable —
    // mousing a row in the table then lights up its own, offset, spot.
    var OFFSET_PX = 7;
    function offPx(p, g) {
      var pt = atomPx(p);
      if (g && g.off && (g.off[0] || g.off[1])) { pt.x += g.off[0] * OFFSET_PX; pt.y += g.off[1] * OFFSET_PX; }
      return pt;
    }

    function drawOverlay(activeLabel) {
      if (!overlay || !current) return;
      var W = overlay.width, H = overlay.height;
      ctx.clearRect(0, 0, W, H);
      var r = 10;   // fixed pixel radius for glow circles

      if (!activeLabel) {
        current.groups.forEach(function(g) {
          g.atoms.forEach(function(p) {
            var pt = atomPx(p);   // static view: circles sit on the atom (methylene split is hover-only)
            ctx.beginPath(); ctx.arc(pt.x, pt.y, r, 0, Math.PI*2);
            ctx.fillStyle = g.color + "40"; ctx.fill();
            ctx.strokeStyle = g.color; ctx.lineWidth = 1.5;
            ctx.globalAlpha = 0.65; ctx.stroke(); ctx.globalAlpha = 1;
          });
        });
        return;
      }

      // Spotlight: veil colour matches the mol-box background (dark or light mode)
      var isDark = document.documentElement.getAttribute("data-theme") === "dark";
      var veilColor = isDark ? "rgba(0,0,0,0.55)" : "rgba(255,255,255,0.65)";
      var revealOpaque = isDark ? "rgba(0,0,0,1)" : "rgba(255,255,255,1)";
      var revealMid    = isDark ? "rgba(0,0,0,0.82)" : "rgba(255,255,255,0.82)";
      var revealEdge   = isDark ? "rgba(0,0,0,0)" : "rgba(255,255,255,0)";
      ctx.fillStyle = veilColor; ctx.fillRect(0, 0, W, H);
      var ag = current.groups.find(function(g){ return g.label === activeLabel; });
      if (!ag) return;

      ctx.globalCompositeOperation = "destination-out";
      ag.atoms.forEach(function(p) {
        var pt = offPx(p, ag);
        var gr = ctx.createRadialGradient(pt.x, pt.y, 0, pt.x, pt.y, r*2.8);
        gr.addColorStop(0,   revealOpaque);
        gr.addColorStop(0.5, revealMid);
        gr.addColorStop(1,   revealEdge);
        ctx.fillStyle = gr; ctx.beginPath(); ctx.arc(pt.x, pt.y, r*2.8, 0, Math.PI*2); ctx.fill();
      });
      ctx.globalCompositeOperation = "source-over";

      ag.atoms.forEach(function(p) {
        var pt = offPx(p, ag);
        ctx.beginPath(); ctx.arc(pt.x, pt.y, r, 0, Math.PI*2);
        ctx.strokeStyle = ag.color; ctx.lineWidth = 2.5;
        ctx.shadowColor = ag.color; ctx.shadowBlur = 10;
        ctx.stroke(); ctx.shadowBlur = 0;
      });
    }

    function resizeOverlay() {
      // Canvas covers the full molBox; buffer = display size (no stretching)
      var w = molBox.clientWidth  || 500;
      var h = molBox.clientHeight || 300;
      overlay.width  = w;
      overlay.height = h;
      drawOverlay(hoverLabel);
    }

    function setHover(label) {
      if (label === hoverLabel) return;
      hoverLabel = label;
      tableBody.querySelectorAll("tr").forEach(function(tr) {
        tr.classList.toggle("sv-hover", tr.dataset.label === label);
      });
      drawOverlay(label);
    }

    // ---- Table ----
    function buildTable() {
      tableBody.innerHTML = "";
      current.groups.forEach(function(g) {
        var tr = document.createElement("tr"); tr.dataset.label = g.label;
        var tdD = document.createElement("td"); tdD.className = "sg-dot-cell";
        var dot = document.createElement("span"); dot.className = "sv-dot";
        dot.style.background = g.color; tdD.appendChild(dot);
        var tdL = document.createElement("td"); tdL.className = "sg-label";
        tdL.style.color = g.color; tdL.textContent = g.label;
        var tdT = document.createElement("td"); tdT.className = "sg-type";
        tdT.textContent = g.tier_label;
        var tdC = document.createElement("td"); tdC.className = "sg-count";
        tdC.textContent = g.h_count;
        tr.append(tdD, tdL, tdT, tdC);
        tr.addEventListener("mouseenter", function() { setHover(g.label); });
        tr.addEventListener("mouseleave", function() { setHover(null); });
        tableBody.appendChild(tr);
      });
    }

    // ---- Select ----
    function select(idx, scroll) {
      tileCont.querySelectorAll(".sv-tile").forEach(function(t, i) {
        t.classList.toggle("sv-active", i === idx);
        // only scroll on a user click — never on the initial select(0), which
        // would otherwise yank the whole page down to this viewer on load/refresh
        if (i === idx && scroll) t.scrollIntoView({block:"nearest", inline:"nearest"});
      });
      current = mols[idx]; hoverLabel = null;
      headId.textContent  = current.id;
      headSub.textContent = current.formula + " · " + current.n_protons + " protons";
      smilesVal.textContent = current.smiles;
      buildTable();

      // Replace molecule image
      var oldImg = molBox.querySelector("img");
      if (oldImg) oldImg.remove();

      if (current.svg) {
        var img = document.createElement("img");
        img.className = "sv-mol-img"; img.alt = current.id;
        // Fixed scale for uniform bond sizes; clamp to box minus 24px padding each side
        var PAD = 24;
        var maxW = molBox.clientWidth  - PAD * 2;
        var maxH = molBox.clientHeight - PAD * 2;
        var imgW = current.svg_w * PIXEL_SCALE;
        var imgH = current.svg_h * PIXEL_SCALE;
        // If the natural size overflows, scale down uniformly (preserves bond proportions
        // for this molecule only; other similarly-sized molecules are unaffected)
        var clamp = Math.min(1, maxW / imgW, maxH / imgH);
        imgW = Math.round(imgW * clamp);
        imgH = Math.round(imgH * clamp);
        current._dispW = imgW;
        current._dispH = imgH;
        img.style.width  = imgW + "px";
        img.style.height = imgH + "px";
        img.src = svgUri(current.svg);
        molBox.insertBefore(img, overlay);
        requestAnimationFrame(resizeOverlay);
      }
    }

    // ---- Gallery ----
    function buildGallery(data) {
      mols = data.molecules;
      tileCont.innerHTML = "";
      mols.forEach(function(mol, i) {
        var dotColor = (mol.groups.find(function(g){return g.tier==="HARD";}) ||
                        mol.groups.find(function(g){return g.tier==="SOFT";}) ||
                        mol.groups[0]).color;
        var tile = document.createElement("div"); tile.className = "sv-tile";
        var dot  = document.createElement("span"); dot.className = "sv-tile-dot";
        dot.style.background = dotColor;
        var body = document.createElement("div"); body.className = "sv-tile-body";
        var idEl = document.createElement("div"); idEl.className = "sv-tile-id";
        idEl.textContent = mol.id;
        var sub  = document.createElement("div"); sub.className = "sv-tile-sub";
        sub.textContent = mol.formula + " · " + mol.n_protons + "H";
        body.append(idEl, sub); tile.append(dot, body);
        tile.addEventListener("click", function() { select(i, true); });
        tileCont.appendChild(tile);
      });
      if (mols.length) select(0);
    }

    // ---- Gallery height sync ----
    function syncGalleryHeight() {
      var card   = host.querySelector(".sv-detail-card");
      var galCol = host.querySelector(".sv-gallery-col");
      if (!card || !galCol) return;
      var bodyH = card.offsetHeight - 42;
      galCol.style.maxHeight = Math.round(bodyH * 0.67) + 40 + "px";
      galCol.style.alignSelf  = "start";
      galCol.style.marginTop  = "42px";
    }

    // ---- Init ----
    // Build overlay canvas once, keep it
    overlay = document.createElement("canvas");
    overlay.className = "sv-overlay";
    ctx = overlay.getContext("2d");
    molBox.appendChild(overlay);

    overlay.addEventListener("mousemove", function(e) {
      if (!current) return;
      var rect = overlay.getBoundingClientRect();
      var mx = e.clientX - rect.left;   // pixel coords in canvas space
      var my = e.clientY - rect.top;
      var best = 20, bestLbl = null;    // 20px hit threshold
      current.groups.forEach(function(g) {
        g.atoms.forEach(function(p) {
          var pt = offPx(p, g);
          var d = Math.hypot(mx - pt.x, my - pt.y);
          if (d < best) { best = d; bestLbl = g.label; }
        });
      });
      setHover(bestLbl);
    });
    overlay.addEventListener("mouseleave", function() { setHover(null); });
    window.addEventListener("resize", resizeOverlay);

    fetch("data/spin_viewer.json")
      .then(function(r) { return r.json(); })
      .then(function(data) {
        buildGallery(data);
        requestAnimationFrame(syncGalleryHeight);
        window.addEventListener("resize", syncGalleryHeight);
      })
      .catch(function(e) {
        tileCont.innerHTML = "<p class='sv-empty'>Molecule browser unavailable.</p>";
        console.error(e);
      });
  }

  // ============= 5. HELD-OUT TEST EVAL (2-D: columns axis + ‹ › fixed axis) =
  // The fleet is a recipe × size grid. Pick which axis is the COLUMNS:
  //   · by recipe → columns = 025…029; ‹ › steps the fixed model SIZE (64k/500k/3M)
  //   · by model size → columns = 64k/500k/3M; ‹ › steps the fixed RECIPE (025…029)
  // So you can compare recipes at each size, or sizes at each recipe.
  function initTestEval() {
    const host = $("#testEval"); if (!host) return;
    fetch("data/test_eval.json").then(r => r.json()).then(d => {
      const meta = d._meta || {};
      const done = (k) => d[k] && d[k].state === "finished" && d[k].test;
      if (!Object.keys(d).some(k => k !== "_meta" && done(k))) {     // nothing finished yet
        host.innerHTML = '<p class="muted" style="font-size:13px">' +
          (meta.note || "Held-out test evaluation is computed once a model finishes training.") + '</p>';
        return;
      }
      const RECIPES = ["025", "026", "027", "028", "029", "030"];
      const SIZES = [["64k", "10M"], ["500k", "57M"], ["3M", "137M"]];
      const rows = [["shift_mae_ppm", "shift MAE (ppm) ↓", true], ["j_mae_hz", "J MAE (Hz) ↓", true],
        ["presence_f1", "presence F1 ↑", false], ["deg_acc_balanced", "deg balanced-acc ↑", false]];
      let mode = "recipe", si = 0, ri = 1;   // columns=recipes, fixed size=64k / fixed recipe=026
      host.innerHTML =
        '<style>.te-bar{display:flex;gap:6px;align-items:center;margin-bottom:12px;flex-wrap:wrap}' +
        '.te-bar button{padding:4px 11px;border-radius:7px;border:1px solid var(--line-strong);' +
        'background:var(--panel);color:var(--ink-soft);font:600 12px/1.2 inherit;cursor:pointer}' +
        '.te-bar button.on{border-color:var(--accent);color:var(--ink)}' +
        '.te-nav{display:inline-flex;gap:6px;align-items:center;margin-left:10px}' +
        '.te-nav .te-fixed{font:600 12px inherit;color:var(--ink-soft);min-width:118px;text-align:center}</style>' +
        '<div class="te-bar"></div><div class="te-body"></div>';
      const bar = $(".te-bar", host), body = $(".te-body", host);
      function cols() {
        if (mode === "recipe") return RECIPES.map(r => ({ k: `${SIZES[si][0]}_${r}`, lbl: r }))
          .filter(c => d[c.k]);
        return SIZES.map(([t, p]) => ({ k: `${t}_${RECIPES[ri]}`, lbl: `${p} · ${t}` })).filter(c => d[c.k]);
      }
      function render() {
        const cs = cols();
        let h = '<table class="cmp"><thead><tr><th>metric</th>';
        cs.forEach(c => { const e = d[c.k];
          const tag = e.state !== "finished" ? "· training" : (!e.test ? "· eval pending" : "");
          h += '<th class="num">' + c.lbl + (tag ?
            ' <span style="color:var(--ink-faint);font-weight:400">' + tag + '</span>' : '') + '</th>'; });
        h += '</tr></thead><tbody>';
        rows.forEach(([mk, lbl, lower]) => {
          const vals = cs.map(c => done(c.k) ? d[c.k].test[mk] : null);
          const fin = vals.filter(v => v != null);
          const best = fin.length ? (lower ? Math.min : Math.max).apply(null, fin) : null;
          h += '<tr><td>' + lbl + '</td>' + vals.map(v =>
            '<td class="num' + (v != null && v === best ? ' best' : '') + '">' +
            (v == null ? '—' : (+v).toFixed(4)) + '</td>').join("") + '</tr>';
        });
        h += '</tbody></table>';
        const fixed = mode === "recipe" ? `size: ${SIZES[si][0]} · ${SIZES[si][1]}` : `recipe: ${RECIPES[ri]}`;
        const blurb = mode === "recipe" ? "recipes at a fixed model size — ‹ › changes the size"
                                        : "model sizes at a fixed recipe — ‹ › changes the recipe";
        body.innerHTML = h + '<p class="muted" style="font-size:12px;margin-top:8px">' + blurb + '. ' +
          (meta.note || "") + ' Test eval: ' + (meta.test_n_eval || "?") + ' of ' +
          (meta.test_n_total || "?") + ' held-out molecules. <b>Test ≈ validation → no overfitting.</b></p>';
        bar.innerHTML =
          '<button data-m="recipe"' + (mode === "recipe" ? ' class="on"' : '') + '>by recipe</button>' +
          '<button data-m="size"' + (mode === "size" ? ' class="on"' : '') + '>by model size</button>' +
          '<span class="te-nav"><button data-nav="-1" title="previous">‹</button>' +
          '<span class="te-fixed">' + fixed + '</span>' +
          '<button data-nav="1" title="next">›</button></span>';
        bar.querySelector('[data-m="recipe"]').onclick = () => { mode = "recipe"; render(); };
        bar.querySelector('[data-m="size"]').onclick = () => { mode = "size"; render(); };
        bar.querySelectorAll("[data-nav]").forEach(b => b.onclick = () => {
          const s = +b.dataset.nav;
          if (mode === "recipe") si = (si + s + SIZES.length) % SIZES.length;
          else ri = (ri + s + RECIPES.length) % RECIPES.length;
          render();
        });
      }
      render();
    }).catch(e => { host.innerHTML = "<p class='muted'>test eval unavailable</p>"; console.error(e); });
  }

  function boot() { initCurves(); initTable(); initTestEval(); initExplorer(); initSpinViewer(); }
  if (document.readyState !== "loading") boot(); else document.addEventListener("DOMContentLoaded", boot);
})();
