/* SpinHance — interactive 3D molecule view for "The representation".
   Generates a 3D conformer from the hero molecule's SMILES with OpenChemLib
   (client-side) and renders it with 3Dmol.js. Slowly auto-rotates; sleek toggle. */
import { Molecule, ConformerGenerator } from "https://cdn.jsdelivr.net/npm/openchemlib@9.22.1/dist/openchemlib.js";

const SPIN_SPEED = 0.6;
const host = document.getElementById("mol3d");
const loadEl = document.getElementById("mol3dLoad");
const toggle = document.getElementById("spinToggle");

let viewer = null;
let lastSmiles = null;

function waitFor3Dmol() {
  return new Promise((resolve, reject) => {
    let tries = 0;
    (function check() {
      if (window.$3Dmol) return resolve(window.$3Dmol);
      if (++tries > 100) return reject(new Error("3Dmol failed to load"));
      setTimeout(check, 50);
    })();
  });
}

function smilesTo3DMol(smiles) {
  const mol = Molecule.fromSmiles(smiles);
  const gen = new ConformerGenerator(0);
  const conf = gen.getOneConformerAsMolecule(mol);   // adds H, sets 3D coords
  if (!conf) throw new Error("conformer generation failed");
  return conf.toMolfile();                            // V2000, with 3D coords
}

function applySpin() {
  if (!viewer) return;
  if (toggle && toggle.checked) viewer.spin("y", SPIN_SPEED);
  else viewer.spin(false);
}

async function render(smiles) {
  if (!smiles || smiles === lastSmiles) return;
  lastSmiles = smiles;
  try {
    const $3Dmol = await waitFor3Dmol();
    if (loadEl) { loadEl.textContent = "building 3D structure…"; loadEl.style.opacity = "1"; }
    const molblock = smilesTo3DMol(smiles);
    if (!viewer) viewer = $3Dmol.createViewer(host, { backgroundAlpha: 0 });
    viewer.removeAllModels();
    viewer.addModel(molblock, "mol");
    viewer.setStyle({}, { stick: { radius: 0.14 }, sphere: { scale: 0.25 } });
    viewer.zoomTo();
    viewer.render();
    applySpin();
    if (loadEl) loadEl.style.opacity = "0";
  } catch (err) {
    console.error("3D view error", err);
    if (loadEl) { loadEl.textContent = "3D structure unavailable for this molecule"; loadEl.style.opacity = "1"; }
  }
}

if (toggle) toggle.addEventListener("change", applySpin);
window.addEventListener("spinhance:molecule", () => { if (window.__heroMol) render(window.__heroMol.smiles); });
// in case the hero picked its molecule before this module finished loading
if (window.__heroMol) render(window.__heroMol.smiles);
