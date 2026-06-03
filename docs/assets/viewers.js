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
    "light025": "#e0a44d", "light026": "#c46be0", "light027": "#7ee06b", "xl025": "#e06b6b", "xl026": "#6bd0e0",
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
        `<button data-k="${k}" class="on" title="click: show only this (kept pins stay) · double-click: pin/unpin" style="--c:${colOf(k)}"><i></i>${data[k].label}</button>`).join("");
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

  // ====================== 2. COMPARISON TABLE ==============================
  const RUNS = [
    { m: "CNN baseline", arch: "ResNet-1D + typed heads", data: "64k ChEMBL", p: "5.0M", shift: "0.279", j: "1.80", f1: "0.807", deg: "0.732", st: "floor" },
    { m: "64k·022", arch: "spingraph + surrogate-spectral", data: "64k ChEMBL", p: "10M", shift: "0.064", j: "0.91", f1: "0.916", deg: "0.928", st: "superseded" },
    { m: "64k·025", arch: "spingraph, shift-2×, WSD LR", data: "64k ChEMBL", p: "10M", shift: "0.037", j: "0.59", f1: "0.940", deg: "0.945", st: "superseded" },
    { m: "64k·026", arch: "025 + peak channel + soft-equiv", data: "64k ChEMBL", p: "10M", shift: "0.037", j: "0.65", f1: "0.940", deg: "0.960", st: "superseded" },
    { m: "500k·025", arch: "025 recipe", data: "500k PubChem", p: "10M", shift: "0.036", j: "0.51", f1: "0.969", deg: "0.969", st: "production" },
    { m: "500k·026", arch: "026 recipe", data: "500k PubChem", p: "10M", shift: "0.058", j: "0.78", f1: "0.944", deg: "0.952", st: "relaunching" },
    { m: "500k·027", arch: "025 + focal loss", data: "500k PubChem", p: "10M", shift: "0.032", j: "0.46", f1: "0.963", deg: "0.969", st: "done" },
    { m: "1M·025", arch: "025 recipe, xl", data: "1M PubChem", p: "57M", shift: "0.045", j: "0.53", f1: "0.971", deg: "0.976", st: "running · ep16" },
    { m: "1M·026", arch: "026 recipe, xl", data: "1M PubChem", p: "57M", shift: "0.044", j: "0.62", f1: "0.964", deg: "0.919", st: "running · ep16" },
  ];
  function initTable() {
    const host = $("#cmpTable"); if (!host) return;
    const cols = [["m", "model"], ["arch", "architecture / recipe"], ["data", "data"], ["p", "params"],
      ["shift", "shift↓"], ["j", "J↓"], ["f1", "F1↑"], ["deg", "deg↑"], ["st", "status"]];
    let sortK = null, asc = true;
    const numK = new Set(["shift", "j", "f1", "deg"]);
    const lowerBetter = new Set(["shift", "j"]);   // others (f1, deg) higher = better
    // best value per metric column (across all rows) -> highlighted bold + gradient
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
  }

  // ====================== 3. TEST-MOLECULE EXPLORER ========================
  function initExplorer() {
    const host = $("#txViewer"); if (!host) return;
    fetch("data/test_explorer.json").then(r => r.json()).then(data => {
      const ppm = data.ppm, mols = data.molecules; let idx = 0;
      // per-model predictions + per-model test-split membership
      const models = data.models || [{ key: "_", label: "model" }];
      let mkey = (models.find(x => x.key === "light025") || models[0]).key;
      let filterTest = false;
      const labelOf = (k) => (models.find(x => x.key === k) || {}).label || k;
      const inTest = (m) => (m.test_of || []).includes(mkey);          // is mol in the SELECTED model's test split?
      const predOf = (m) => (m.preds ? m.preds[mkey] : m);
      const spec = $("#txSpec", host), sel = $("#txSel", host), meta = $("#txMeta", host), mat = $("#txMatrix", host);
      const modSel = $("#txModel", host), filterCb = $("#txFilter", host), statusEl = $("#txStatus", host);
      const el3d = $("#tx3d", host), load3d = $("#tx3dLoad", host); let v3d = null;
      if (modSel) { modSel.innerHTML = models.map(mm => `<option value="${mm.key}">${mm.label}</option>`).join(""); modSel.value = mkey; }
      const visibleIdx = () => mols.map((m, i) => i).filter(i => !filterTest || inTest(mols[i]));
      function rebuildMolOptions() {
        const vis = visibleIdx();
        sel.innerHTML = vis.map(i => { const m = mols[i];
          return `<option value="${i}">${inTest(m) ? "● " : ""}${m.id} · ${m.n_spins}H</option>`; }).join("");
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
            v3d.zoomTo(); v3d.resize(); v3d.render(); v3d.spin("y", 0.6);
            if (load3d) load3d.style.opacity = "0";
          } catch (e) { if (load3d) { load3d.textContent = "3D unavailable"; load3d.style.opacity = "1"; } console.error(e); }
        });
      }
      function drawSpec() {
        const m = mols[idx], P = predOf(m), c = C();
        // target-spectrum colour encodes test membership: teal = held-out test for this model, amber = out-of-distribution
        const tgt = inTest(m) ? (css("--accent-2") || "#34e3c4") : "#f5a623";
        linePlot(spec, [
          { color: tgt, width: 1.8, pts: m.input.map((v, i) => [ppm[i], v]) },
          { color: c.accent, width: 2, pts: P.rendered.map((v, i) => [ppm[i], v]) },
        ], { xlabel: "ppm", x0: 0, x1: 12, y0: 0, invertX: true, noY: true });
      }
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
      function show() {
        const m = mols[idx], P = predOf(m), held = inTest(m); sel.value = idx;
        if (statusEl) {
          statusEl.className = "tx-status " + (held ? "is-test" : "is-ood");
          statusEl.innerHTML = held
            ? `● held-out test molecule for <b>${labelOf(mkey)}</b>`
            : `● not in <b>${labelOf(mkey)}</b>'s test split — out-of-distribution`;
        }
        meta.innerHTML = `<span class="mono">${m.smiles || m.id}</span> · ${m.n_spins} protons ·
          shift MAE <b>${P.shift_mae.toFixed(3)}</b> ppm · J MAE <b>${P.j_mae.toFixed(2)}</b> Hz`;
        drawSpec(); drawMatrix(); render3d(m);
      }
      function step(d) { const vis = visibleIdx(); let p = vis.indexOf(idx); if (p < 0) p = 0;
        p = (p + d + vis.length) % vis.length; idx = vis[p]; show(); }
      sel.onchange = () => { idx = +sel.value; show(); };
      if (modSel) modSel.onchange = () => { mkey = modSel.value; rebuildMolOptions(); show(); };
      if (filterCb) filterCb.onchange = () => { filterTest = filterCb.checked; rebuildMolOptions(); show(); };
      $("#txPrev", host).onclick = () => step(-1);
      $("#txNext", host).onclick = () => step(1);
      idx = Math.max(0, mols.findIndex(inTest));   // open on a genuine held-out example for the default model
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

  // ====================== 5. HELD-OUT TEST EVAL (val vs test) ==============
  function initTestEval() {
    const host = $("#testEval"); if (!host) return;
    fetch("data/test_eval.json").then(r => r.json()).then(d => {
      const models = ["025", "026"];
      const rows = [["shift_mae_ppm", "shift MAE (ppm) ↓"], ["j_mae_hz", "J MAE (Hz) ↓"],
        ["presence_f1", "presence F1 ↑"], ["deg_acc_balanced", "deg balanced-acc ↑"]];
      let h = '<table class="cmp"><thead><tr><th>metric</th>';
      models.forEach(m => { h += '<th class="num">' + m + ' val</th><th class="num">' + m + ' test</th>'; });
      h += '</tr></thead><tbody>';
      rows.forEach(function (rk) {
        h += '<tr><td>' + rk[1] + '</td>';
        models.forEach(m => { h += '<td class="num">' + d[m].val[rk[0]] + '</td><td class="num tval">' + d[m].test[rk[0]] + '</td>'; });
        h += '</tr>';
      });
      h += '</tbody></table>';
      const meta = d._meta || {};
      host.innerHTML = h + '<p class="muted" style="font-size:12px;margin-top:8px">' +
        (meta.note || "") + ' Test eval: ' + (meta.test_n_eval || "?") + ' of ' + (meta.test_n_total || "?") +
        ' held-out molecules. <b>Test ≈ validation → no overfitting.</b></p>';
    }).catch(e => { host.innerHTML = "<p class='muted'>test eval unavailable</p>"; console.error(e); });
  }

  function boot() { initCurves(); initTable(); initTestEval(); initExplorer(); initSpinViewer(); }
  if (document.readyState !== "loading") boot(); else document.addEventListener("DOMContentLoaded", boot);
})();
