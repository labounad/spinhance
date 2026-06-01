# Spinhance — Hackathon Presentation Outline

**Team:** Lucas Abounader (Shenvi Lab), Samuel Mansfield (Seiple Lab), Yiming Zhang (Shenvi Lab)
**Format target:** 5–7 min, 4 slides, all three present
**Judges:** Andrew Su, Stefano Forli, Megan Ken (all Dept. of Integrative Structural & Computational Biology, Scripps) + AI judge (Claude)
**Live site:** https://labounad.github.io/spinhance/

---

## 0. The one-sentence thesis (say this verbatim, twice)

> *A 90 MHz benchtop spectrum looks blurry, but the blur isn't noise — it's the full spin system, just scrambled. Spinhance trains a network to invert that blur back into field-independent parameters (δ, J, degeneracies), so a cheap spectrum can be re-simulated exactly at any field.*

Everything on every slide should ladder back to this.

---

## 1. Format recommendation (decision needed)

**Recommended: a Reveal.js HTML deck styled to match your existing website, run full-screen in a browser, with a PDF export as the A/V backup.**

Why this wins for *you* specifically:

- **It looks the best and is the fastest to iterate with me.** I can edit HTML/CSS/JS directly and restyle in seconds; pptx editing via tooling is clunky and rarely looks polished (your own concern).
- **You already have a beautiful, on-brand asset.** The site's CSS, dark theme, spinning-proton icon, and the *live pyspin field-sweep animation* drop straight into slides. The deck and the site become one visual language.
- **It solves the "get them onto the website" goal natively.** The deck can embed the live 90→600 MHz sweep on the problem slide, and the closing slide can be the site itself (or a QR/short link on screen).
- **Robust:** runs offline from a single file, has a presenter view + speaker notes, and exports to PDF if the venue's machine misbehaves.

Fallbacks, ranked: (2) **Canva** (a Canva connector is available — good if you want a more "designed," less technical look, but weaker for exact scientific figures); (3) **pptx** (safe, familiar, but hardest to make look great and to iterate).

**Backup plan regardless of choice:** export final slides to PDF and put them on a USB stick + email. Never present a live-animated deck without a static fallback.

---

## 2. Audience-match strategy (this is how we win)

All three faculty judges sit in the **same department — Integrative Structural & Computational Biology**. Read their work and three shared values jump out. Hit all three explicitly.

| Judge | What they do | The hook for us |
|---|---|---|
| **Megan Ken** | NMR spectroscopist (trained Al-Hashimi lab). RNA structural dynamics, **excited/rare conformational states**, **high-throughput NMR**, small-molecule **antiviral** drug discovery. MD/PhD, Scripps Fellow. | She *is* our specialist. The second-order / spin-Hamiltonian content will land with her. She values extracting hidden information from spectra and democratizing NMR throughput. Do **not** hand-wave the physics in front of her. |
| **Stefano Forli** | Builds the **AutoDock** suite — the most-used *open-source* docking platform. Directs an NIH national resource giving academia + industry **accessible** computational drug-discovery tools. Trained medicinal chemist. | Frame Spinhance as an **open, accessible tool that lowers the barrier** to structure work — the AutoDock ethos. Our physics-grounded simulator (validated, license-free pyspin) speaks his language. |
| **Andrew Su** | Bioinformatics, **data integration**, knowledge graphs, **crowdsourcing / democratizing** biomedical data, drug repurposing, AI in medicine. | He rewards **accessibility and scale**: cheap instrument + synthetic data engine = more people, more molecules, more data. The ML inversion is squarely his lane. |

**The unifying message for all three:** *high-field structural information is gated behind $500k–$1M magnets; we use physics + ML to unlock it from a ~$50k benchtop — democratizing structure verification, which is foundational to small-molecule drug discovery.* That sentence touches accessibility (Su, Forli), drug discovery (Forli, Ken), and NMR (Ken) at once.

**Two-register rule (Communication rubric explicitly rewards this):** every technical beat needs a one-line plain-language translation. Specialists (Ken) get the rigor; non-specialists (Su, Forli on NMR specifics) never get lost.

**Note on the AI judge (Claude):** the "built from a one-line concept to a working, validated pipeline in ~2.5 days" story is genuinely strong here — lean into ambition and breadth, not lines of code.

---

## 3. Rubric coverage map (50 pts: 5 × 10)

Make sure each category is *unmistakably* addressed. Where it lives:

- **Innovation & Creativity (10)** — Slide 1: reframing low-field "blur" as *recoverable information* via the spin Hamiltonian. Slide 2 adds the *differentiable* spin-Hamiltonian renderer we built (regularized-eigh gradients verified to ~1e-6 vs finite difference) — it makes a spectral-consistency loss *possible*, a non-obvious trick. Frame it as a validated capability + roadmap, **not** as what the shipped checkpoint optimizes (`modelv2` trains on the matrix loss only). Take the intellectual risk out loud.
- **Scientific Merit (10)** — Slide 1 + Slide 4: accessibility of structure determination → impact on drug discovery / low-resource labs. Tie to *biomedical* relevance (the rubric asks for human-health consequence explicitly).
- **Technical Quality & Execution (10)** — Slide 2 + Slide 3: validated simulator (r ≈ 0.999), leakage-safe scaffold splits, honest failure logging (103 logged), reproducible open repo + README. Mention "anyone can clone and run it."
- **Communication (10)** — whole deck: clean visuals reused from the site, two-register narration, live animation, crisp thesis.
- **Interdisciplinary (10)** — Slide 2: name the disciplines as they appear — **cheminformatics (RDKit) + quantum spin physics (Hamiltonian sim) + deep learning + cloud orchestration**. Say the word "interdisciplinary" once; the boxes prove it.

---

## 4. Slide-by-slide

### Slide 1 — The problem & the bet  *(Lucas, ~1:30)*

**Purpose:** hook + accessibility stakes + the counterintuitive scientific insight (Innovation, Scientific Merit).

**Visual:**
- Side-by-side: a **90 MHz benchtop** (small, bench-top, ~$50k) vs a **600 MHz magnet** (room-sized, cryogen-fed, ~$0.5–1M + siting/maintenance). Use real photos; label cost + footprint + accessibility.
- The **live field-sweep animation** from the site (same molecule, 90→600 MHz) — *this is your best single visual.* Let it morph from overlapping second-order multiplets to clean first-order peaks.

**Talking points:**
1. High-field NMR is the gold standard for structure, but it's expensive, centralized, and gatekept.
2. At low field, peaks overlap and naive peak-picking fails — so benchtops are usually dismissed as "lower information."
3. **The reframe (say carefully — Ken is listening):** the appearance is governed by the ratio Δν/J. At low field Δν/J ≈ 1, so the spectrum is *second-order* — the line positions and intensities are a nonlinear function of the **entire** spin system (all shifts and couplings, including relative signs). The information isn't lost; it's encoded in a pattern too tangled to read by hand.
4. **The bet:** if the information is there, a network can learn the inverse map. Recover δ, J, degeneracies → reconstruct the spectrum at *any* field.

> ⚠️ **Accuracy guard (for the NMR judge):** the defensible claim is *"low-field second-order spectra preserve/encode the full parameter set; first-order spectra discard information such as relative coupling signs."* Do **not** overclaim "low field has more information than high field" as a blanket statement — say the information is *preserved and recoverable*, and that second-order patterns additionally encode relative signs that first-order spectra don't.

---

### Slide 2 — How it works: four modules  *(Sam + Yiming + Lucas, ~2:30)*

**Purpose:** the build, the scale, the rigor (Technical Quality, Interdisciplinary). Four boxes, one per corner. One molecule's data flows clockwise through them.

**Layout:** 2×2 grid, a faint arrow showing SMILES → matrix → spectra → model. Each box: icon, one-line "what," the key method, and **one punchy fact**. Keep each box to ~30–35 s.

| Box | Module | What it does + key method | The punchy fact | Speaker |
|---|---|---|---|---|
| ◷ 1 | **generate** | SMILES → 3D structures with **every proton labeled by spin group**, distinguishing *hard* equivalence (same shift, same group) from *soft* (same shift, different group). RDKit 3D embedding + a deuterium-substitution magnetic-equivalence test. Non-trivial: refined by whacking edge cases until the spin-group count was reliable. | **69,324** exactly-8-group ChEMBL molecules; one categorizing scan buckets *all* spin-group counts (1–26) at once. *Show the annotated 2D-structure viewer — it's the asset that made the edge-case whack-a-mole possible.* | **Sam** |
| ◶ 2 | **mol_to_spin_system** | Molecule → spin-system parameters: shifts δ via **NMRShiftDB2**, couplings J via **Karplus + large literature coupling tables**, with a dedicated code path per case (geminal, vicinal, benzylic, allylic, aromatic…). | **Nearly free, high quality:** generated a **2M+ molecule** dataset this weekend; throughput ~**3M molecules, multiple times, in a weekend**. *Competing methods cost far more for comparable-looking data — this is a standalone contribution.* | **Yiming** |
| ◵ 3 | **simulation (simulate)** | Diagonalize the spin Hamiltonian → realistic spectra at 90 + 600 MHz, capturing the **higher-order coupling** that makes low-field spectra work. Started as a library that *scripted* **MestReNova** (slow, paid license); then **reimplemented from scratch** as **pyspin**. | pyspin **reproduces MestReNova to r ≈ 0.999 (effectively exact)** — but is open-source, license-free, and far faster. **Lucas reverse-engineered + built it in one day.** *"The slide-1 animation is this engine, live."* | **Lucas** |
| ◴ 4 | **model** (`modelv2`) | One 90 MHz spectrum → the field-independent spin-system matrix. **ResNet-1D encoder → 4 typed heads** (δ, \|J\|, J-presence, degeneracy). Canonical sort resolves the **S₈** label-permutation problem; **GroupNorm + EMA** keep training stable. A clean **4-file rewrite** of the original `model/` (which fragmented and stopped working). | Trained **single-stage on the matrix loss** on **AWS (NVIDIA L40S, g6e.16xlarge)** over the **2M+ PubChem** set on S3; leakage-safe **scaffold + matrix-dedup** splits; anti-mean-collapse diagnostics (Var-ratio, Pearson r, constant-mean baseline). *Show the live Streamlit dashboard — it runs on 500 molecules the model never saw.* **(Final metrics locked in at the end — Sam.)** | **Sam** |

**Why the rigor matters (drop one line for the judges):** "These splits are scaffold-grouped and deduplicated, so the test molecules are genuinely unseen — the numbers aren't memorization."

**Interdisciplinary call-out (one sentence):** "Four fields in one pipeline — cheminformatics, quantum spin physics, deep learning, and cloud orchestration."

**Innovation note — say it honestly:** we *also* built a **differentiable** pyspin renderer (regularized-eigh gradients verified to ~1e-6 vs finite difference), so a spectral-consistency loss can be trained against the physics. This is a **validated capability and our roadmap** — the shipped `modelv2` checkpoint trains on the **matrix loss only**. Don't imply the spectral loss is already in the trained model (Ken will ask).

**The "from zero" beat (Sam, one line):** "We started this with a single sentence from Lucas and an empty repo — everything you're seeing is ~2.5 days and 189 commits old."

---

### Slide 3 — Proof: watch it invert the blur  *(Sam, ~1:30)*

**Purpose:** make the thesis *visible*. Lead with the reconstruction and the live demo — not a metrics table — and let the physics-grounded pipeline carry the proof (Technical Quality, Scientific Merit, Communication).

**Visual (the demo is the proof):**
- **The headline reconstruction:** feed a **90 MHz** spectrum → predict δ/J/degeneracies → **re-simulate at 600 MHz** → overlay on the true 600 MHz. *This is the one-sentence thesis made visible — they should sit on top of each other.* This single visual is the slide.
- **Live, on every reload:** the site simulates a fresh molecule in-browser, so the 90→600 story is interactive, not a static screenshot. If the in-browser inference demo lands (Section 6), do it live on stage: paste/upload a 90 MHz spectrum → predicted δ/J + reconstructed 600 MHz.
- **Credibility without numbers:** the diagnostics dashboard runs on **500 held-out molecules the model never saw** (scaffold + matrix-dedup split). Say that out loud — it's the "this isn't memorization" guarantee, and it survives even if you show zero metrics.
- Optional support if you want it: a ground-truth-vs-predicted overlay on a held-out molecule and a small predicted-vs-true δ/J scatter — both already presentation-grade and on-brand from the Streamlit GUI.

> 🔧 **Metrics are optional here (your call — demo-first).** The reconstruction + live demo carry this slide; numbers are a bonus, not the spine. **If** modelv2 results are ready, drop them in as one quiet line — shift MAE (ppm), J MAE (Hz), presence F1, degeneracy acc — **Hungarian-matched**, since the 8 group labels are permutation-arbitrary. No fabricated figures; an honest "early results, demo-first" reads fine for a 2.5-day build. Also state which set the checkpoint trained on (2M+ PubChem on S3 vs the 1,072-mol dev set).

**Talking points:** lead with the qualitative punch — "feed it a blurry 90 MHz spectrum, it reconstructs the clean 600 MHz, and they overlay." Then the honesty beat: "evaluated on scaffolds the model never saw." Numbers only if ready, stated plainly — don't let a missing table undercut a working demo.

---

### Slide 4 — Impact, the tool, and the team  *(Lucas closes, ~1:00)*

**Purpose:** land the significance + accessibility, drive them to the site, end on ambition (Scientific Merit, Innovation, Communication).

**Visual:** the live website on screen (or a clean screenshot + **QR code / short link**). Team row at the bottom (already on the site).

**Talking points:**
1. **Why it matters:** structure verification is the daily bottleneck of synthetic and medicinal chemistry; gating it behind million-dollar magnets gates *who* gets to do rigorous chemistry. Spinhance points at high-field-quality answers from an accessible instrument.
2. **Biomedical tie (for this room):** faster, cheaper structure confirmation accelerates small-molecule discovery — the exact front end of the drug-discovery work this department does.
3. **It's a real, open tool:** public repo, README, validated simulator, reproducible — clone and run. *(AutoDock/Su-style accessibility.)*
4. **How it scales — `autoai` (one vivid line, for the AI judge):** "We built a **two-level, human-out-of-the-loop** training orchestrator. An **Opus** agent is the *scientist* — it reads ML papers and picks training strategies. One **Sonnet** is the *worker* that writes the code, trains the model, and debugs until it runs. A second **Sonnet** is the *auditor* that checks the code does what it claims and interprets the results. AI training AI to train the model." Honest framing: "a fun idea, still nascent — not our scientific headline, but it's how this improves without us." Don't overclaim results.
5. **The ask / invitation:** "Try the live demo — every reload simulates a new molecule in your browser." Put the link up and leave it up during Q&A.
6. **Close on the bet:** restate the one-sentence thesis.

---

## 5. Speaker plan & timing (≈6:30 total — inside 5–7)

1. **Lucas** opens — Slide 1 (problem, physics, the bet). *His concept, his physics; he sets the intellectual frame.* ~1:30
2. **Yiming → Lucas → Sam** walk Slide 2's four boxes (Yiming: spin-system; Lucas: simulation; Sam: generate + model). Hand-offs are fast. ~2:30
3. **Sam** — Slide 3 results (he trained the models). ~1:30
4. **Lucas** closes — Slide 4 impact + site. ~1:00

Rehearse the two hand-offs (they're where time is lost). Each person owns ≤2 transitions.

---

## 6. Open decisions (need your input — see chat)

1. **Format:** a 9 MB **`LA_SM_YZ_spinhance_hackathon_2026-06-01.pptx` already sits in the repo**, so the real fork is *rebuild in Reveal.js* (recommended — matches the site, embeds the live sweep, fastest to iterate with me) vs *iterate the existing pptx* (safe, familiar) vs Canva. Still your call.
2. **Use the remaining ~15 h on a build?** Given the demo-first framing (Slide 3), the **in-browser inference demo is now the spine of the proof, not a nice-to-have** — see the elevated option below.
3. **Slide 3 = demo + recorded fallback**, not a metrics table. Numbers are optional (your call); the live 90→600 reconstruction is the proof. Record a fallback video regardless.

### 15-hour enhancement options (ranked)

| Option | Impact | Effort | Verdict |
|---|---|---|---|
| **Live in-browser demo: paste/upload a 90 MHz spectrum → see predicted δ/J + reconstructed 600 MHz** (extend the site; runs client-side or tiny backend) | ★★★★★ — turns "proof" into "watch it happen"; nails Technical Quality + Communication + the website goal | Medium | **Top pick** if a checkpoint can run in-browser/onnx or via a small hosted endpoint |
| Polished **results figures + a short screen-recorded fallback** of the demo | ★★★★ — guarantees a clean money-slide even if live demo flakes | Low | **Do this regardless** |
| **Windows app on a real 90 MHz spectrometer** (live acquisition → reconstruction) | ★★★★★ if it works live; ★ if it fails on stage | High + risky | Only if a spectrometer + time are truly available *and* you keep the recorded fallback |
| Squeeze more training / bigger dataset for better numbers | ★★ | Medium | Lower priority now the demo carries the proof — only if a clean reconstruction needs a better checkpoint; don't let it eat demo time |

My recommendation (reinforced by the demo-first choice): **the in-browser demo *is* the proof — build it, and record a fallback video no matter what.** Skip the live-spectrometer app unless everything else is locked.
