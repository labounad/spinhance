/* Spinhance · dataset explorer
   Canvas histograms (themed to the site palette) over docs/data/dataset_explorer.json.
   No external libs — matches the rest of the site (vanilla canvas). */
(function () {
  "use strict";

  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const PAL = () => ({
    ink: css("--ink"), inkSoft: css("--ink-soft"), inkFaint: css("--ink-faint"),
    line: css("--line"), lineStrong: css("--line-strong"),
    accent: css("--accent"), accent2: css("--accent-2"), accent3: css("--accent-3"),
    panel: css("--panel"),
  });

  // mean of a histogram (bin centers weighted by counts)
  function histMean(h) {
    const isInt = h.edges.length === h.counts.length;
    let s = 0, n = 0;
    for (let i = 0; i < h.counts.length; i++) {
      const c = isInt ? h.edges[i] : (h.edges[i] + h.edges[i + 1]) / 2;
      s += c * h.counts[i]; n += h.counts[i];
    }
    return n ? s / n : 0;
  }
  function histTotal(h) { return h.counts.reduce((a, b) => a + b, 0); }

  // Draw one histogram into a <canvas>; returns a redraw fn. Hover -> highlight + tooltip.
  function makeHist(canvas, tip, h, opt) {
    opt = opt || {};
    const isInt = h.edges.length === h.counts.length;
    const n = h.counts.length;
    const maxC = Math.max(1, ...h.counts);
    let hover = -1;

    function geom() {
      const dpr = window.devicePixelRatio || 1;
      const W = canvas.clientWidth, H = canvas.clientHeight;
      canvas.width = W * dpr; canvas.height = H * dpr;
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return { ctx, W, H };
    }

    function xOf(i, W, padL, padR) {           // left edge of bar i in px
      const span = W - padL - padR;
      return padL + (span * i) / n;
    }

    // value -> plot transform (linear, or log10(1+v) for skewed counts)
    const T = opt.logY ? (v) => Math.log10(v + 1) : (v) => v;
    const yMax = T(maxC) || 1;
    const fmtV = (v) => v >= 1000 ? (v / 1000).toFixed(v >= 10000 ? 0 : 1) + "k" : v.toFixed(0);

    function draw() {
      const P = PAL();
      const { ctx, W, H } = geom();
      const padL = 44, padR = 10, padT = 12, padB = 26;
      const plotH = H - padT - padB;
      ctx.clearRect(0, 0, W, H);
      // y gridlines + labels
      ctx.font = "11px ui-monospace, Menlo, monospace";
      ctx.textBaseline = "middle"; ctx.fillStyle = P.inkFaint; ctx.strokeStyle = P.line;
      let tickVals;
      if (opt.logY) {                          // decade ticks: 0,1,10,100,...
        tickVals = [0];
        for (let d = 1; d <= maxC; d *= 10) tickVals.push(d);
      } else {
        tickVals = []; const nt = 4;
        for (let t = 0; t <= nt; t++) tickVals.push((maxC * t) / nt);
      }
      for (const v of tickVals) {
        const y = H - padB - (plotH * T(v)) / yMax;
        ctx.globalAlpha = 0.5; ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
        ctx.globalAlpha = 1; ctx.textAlign = "right";
        ctx.fillText(fmtV(v), padL - 6, y);
      }
      // bars
      const barGap = n > 40 ? 0.5 : 1.5;
      for (let i = 0; i < n; i++) {
        const x0 = xOf(i, W, padL, padR), x1 = xOf(i + 1, W, padL, padR);
        const bw = Math.max(1, x1 - x0 - barGap);
        const bh = h.counts[i] > 0 ? (plotH * T(h.counts[i])) / yMax : 0;
        const y = H - padB - bh;
        const grad = ctx.createLinearGradient(0, H - padB, 0, padT);
        grad.addColorStop(0, opt.c0 || P.accent);
        grad.addColorStop(1, opt.c1 || P.accent3);
        ctx.fillStyle = i === hover ? P.accent2 : grad;
        ctx.fillRect(x0 + barGap / 2, y, bw, bh);
      }
      // x axis labels (sparse)
      ctx.fillStyle = P.inkFaint; ctx.textAlign = "center"; ctx.textBaseline = "top";
      const labelEvery = Math.ceil(n / (W < 360 ? 6 : 10));
      for (let i = 0; i < n; i++) {
        if (i % labelEvery !== 0) continue;
        const lx = (xOf(i, W, padL, padR) + xOf(i + 1, W, padL, padR)) / 2;
        const lab = isInt ? String(h.edges[i]) : (+h.edges[i]).toFixed(h.edges[n] <= 25 ? 0 : 0);
        ctx.fillText(lab, lx, H - padB + 5);
      }
      // axis line
      ctx.strokeStyle = P.lineStrong; ctx.beginPath();
      ctx.moveTo(padL, H - padB); ctx.lineTo(W - padR, H - padB); ctx.stroke();
    }

    function locate(ev) {
      const r = canvas.getBoundingClientRect();
      const x = ev.clientX - r.left, W = canvas.clientWidth;
      const padL = 44, padR = 10, span = W - padL - padR;
      if (x < padL || x > W - padR) return -1;
      const i = Math.floor((n * (x - padL)) / span);
      return (i >= 0 && i < n) ? i : -1;
    }
    canvas.addEventListener("mousemove", (ev) => {
      const i = locate(ev); if (i === hover) { positionTip(ev); return; }
      hover = i; draw();
      if (i < 0) { tip.style.opacity = 0; return; }
      const total = histTotal(h), pct = ((100 * h.counts[i]) / total).toFixed(1);
      const range = isInt ? `${h.edges[i]}${opt.unit ? " " + opt.unit : ""}`
        : `${(+h.edges[i]).toFixed(opt.dec ?? 1)}–${(+h.edges[i + 1]).toFixed(opt.dec ?? 1)}${opt.unit ? " " + opt.unit : ""}`;
      tip.innerHTML = `<b>${range}</b><br>${h.counts[i].toLocaleString()} <span class="tdim">(${pct}%)</span>`;
      tip.style.opacity = 1; positionTip(ev);
    });
    canvas.addEventListener("mouseleave", () => { hover = -1; tip.style.opacity = 0; draw(); });
    function positionTip(ev) {
      const wrap = canvas.parentElement.getBoundingClientRect();
      tip.style.left = (ev.clientX - wrap.left + 12) + "px";
      tip.style.top = (ev.clientY - wrap.top - 10) + "px";
    }
    return draw;
  }

  const CARDS = [
    { key: "shifts", title: "Chemical shifts δ", sub: "all 8 groups · ppm", unit: "ppm", dec: 1,
      note: "Bimodal: the aliphatic cluster (~0.8–3 ppm) and the aromatic/olefinic band (~6.5–8 ppm)." },
    { key: "couplings", title: "Coupling constants |J|", sub: "all coupled pairs · Hz", unit: "Hz", dec: 1,
      note: "Scalar couplings from the Pretsch rule set — geminal, vicinal, aromatic ³/⁴J, long-range." },
    { key: "connections", title: "Couplings per molecule", sub: "non-zero J edges (of 28 possible)", unit: "",
      note: "How many of the 28 possible group–group couplings are active — the spin-graph's edge count." },
    { key: "components", title: "Coupled subsystems", sub: "connected components", unit: "",
      note: "Independent spin systems within a molecule (isolated groups count as their own component)." },
    { key: "max_clique", title: "Largest coupled clique", sub: "mutually-coupled groups", unit: "",
      note: "Biggest set of groups all coupled to each other — strongly-overlapping multiplets." },
    { key: "n_cliques", title: "Maximal cliques", sub: "per molecule", unit: "",
      note: "Count of maximal mutually-coupled groups — a proxy for spectral complexity." },
    { key: "degeneracy", title: "Group degeneracy", sub: "equivalent ¹H per group · log scale", unit: "H", logY: true,
      note: "Magnetically-equivalent protons per group — 1 (CH), 2 (CH₂), 3 (CH₃), 6/9 (equivalent methyls). Log y-axis: singlets dominate by orders of magnitude." },
    { key: "n_spins", title: "Protons per molecule", sub: "Σ degeneracy", unit: "H",
      note: "Total ¹H the 8 groups represent — the integrated proton count of the spectrum." },
    { key: "shift_spread", title: "Shift spread", sub: "δ(max) − δ(min) · ppm", unit: "ppm", dec: 1,
      note: "Spectral width spanned by a molecule's groups — wide = aliphatic+aromatic mix." },
    { key: "singlets", title: "Uncoupled groups", sub: "groups with no J · per molecule", unit: "",
      note: "Isolated singlets — groups with no scalar coupling to any other group." },
  ];

  fetch("data/dataset_explorer.json").then((r) => r.json()).then((D) => {
    document.querySelectorAll("[data-n]").forEach((el) => {
      el.textContent = (D.n_molecules || 0).toLocaleString();
    });
    const grid = document.getElementById("histGrid");
    const redraws = [];
    CARDS.forEach((card) => {
      const h = D[card.key]; if (!h) return;
      const el = document.createElement("div"); el.className = "hcard";
      el.innerHTML = `
        <div class="hhead">
          <div><div class="htitle">${card.title}</div><div class="hsub">${card.sub}</div></div>
          <div class="hstat">mean<br><b>${histMean(h).toFixed(card.dec != null ? 1 : 1)}</b></div>
        </div>
        <div class="hcanvas-wrap"><canvas class="hcanvas"></canvas><div class="htip"></div></div>
        <div class="hnote">${card.note}</div>`;
      grid.appendChild(el);
      const cv = el.querySelector("canvas"), tip = el.querySelector(".htip");
      const draw = makeHist(cv, tip, h, { unit: card.unit, dec: card.dec, logY: card.logY });
      redraws.push(draw); requestAnimationFrame(draw);
    });
    let t; window.addEventListener("resize", () => {
      clearTimeout(t); t = setTimeout(() => redraws.forEach((d) => d()), 120);
    });
  }).catch((e) => {
    document.getElementById("histGrid").innerHTML =
      '<p style="color:var(--ink-faint)">Could not load dataset_explorer.json.</p>';
    console.error(e);
  });
})();
