/* Spinhance — interactive 3D molecule view for "The representation".
   Renders the hero molecule's precomputed 3D structure (force-field XYZ coords
   from generate/data/pubchem_8spin.xyz.gz, shipped in field_sweep.json) with 3Dmol.js.
   Slowly auto-rotates by default; sleek toggle to stop. */
(() => {
  "use strict";
  const SPIN_SPEED = 0.6;
  const host = document.getElementById("mol3d");
  const loadEl = document.getElementById("mol3dLoad");
  const toggle = document.getElementById("spinToggle");

  let viewer = null, lastId = null;

  function waitFor3Dmol() {
    return new Promise((resolve, reject) => {
      let n = 0;
      (function check() {
        if (window.$3Dmol) return resolve(window.$3Dmol);
        if (++n > 100) return reject(new Error("3Dmol failed to load"));
        setTimeout(check, 50);
      })();
    });
  }

  function applySpin() {
    if (!viewer) return;
    if (toggle && toggle.checked) viewer.spin("y", SPIN_SPEED);
    else viewer.spin(false);
  }

  async function render(m) {
    if (!m || !m.xyz || m.id === lastId) {
      if (m && !m.xyz && loadEl) { loadEl.textContent = "no 3D structure for this molecule"; loadEl.style.opacity = "1"; }
      return;
    }
    lastId = m.id;
    try {
      const $3Dmol = await waitFor3Dmol();
      if (!viewer) viewer = $3Dmol.createViewer(host, { backgroundAlpha: 0 });
      viewer.removeAllModels();
      viewer.addModel(m.xyz, "xyz");                 // 3Dmol infers bonds by distance
      viewer.setStyle({}, { stick: { radius: 0.14 }, sphere: { scale: 0.25 } });
      viewer.zoomTo();
      viewer.render();
      applySpin();
      if (loadEl) loadEl.style.opacity = "0";
    } catch (err) {
      console.error("3D view error", err);
      if (loadEl) { loadEl.textContent = "3D viewer failed to load"; loadEl.style.opacity = "1"; }
    }
  }

  if (toggle) toggle.addEventListener("change", applySpin);
  window.addEventListener("spinhance:molecule", () => render(window.__heroMol));
  if (window.__heroMol) render(window.__heroMol);
})();
