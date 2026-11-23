# Kelpod v2: what to build next to get much better results on real German scans

Research note, 2026-11-23. Builds on `docs/research/OPEN_SOURCE_SURVEY.md` (same date) and does not
repeat it. Web sources were checked today. Anything not checked is marked **unverified**. Every
"expected gain" and "target" below is a **target, not a result**. Nothing here has been measured
on real *Bestandspläne*.

Context read: `KELPOD_CONTEXT.md`, `docs/HANDOFF.md`, `docs/ARCHITECTURE.md`, `docs/AGENT_BRIEFS.md`,
`docs/results/benchmark_synth10_degrade05.md`, `src/kelpod/*`, and `training/README.md`. The
finetune agent has started a 5-class U-Net (segmentation_models_pytorch) with a deterministic
vectoriser. It also has a DIN-style renderer for Swiss Dwellings, ResPlan and CubiCasa5k, and it uses
the CubiCasa5k hourglass model as a zero-shot baseline. The bets below **extend** that work rather
than duplicate it.

---

## 0. The short version

The MVP is a sound design: detect, then fit deterministically to dimensions, then check as a graph.
Recent work backs it. fpvec-lab found that detection readouts beat sequence readouts on real scans,
and that fusing systems adds about 7 wall-F1 points. FloorplanVLM shows that VLM-to-JSON works, but
only with 2M training pairs and 32 H200 GPUs. Three things block a "significantly better on real
scans" claim:

1. **No real-scan number exists.** Every other bet is unmeasurable until one does.
2. **The training and test distribution is narrow.** The generator draws rectangular rooms in one
   drawing style, and the real-scan domain gap is unknown.
3. **Errors are not reconciled.** OCR, segmentation and geometry each make one guess. None of them
   uses the strong redundancy in German plans: chains sum to the overall dimension (*Gesamtmaß*),
   dimension values match pixel spans, room stamps give areas, and storeys stack.

The ranking below follows from this. Expected impact per effort, highest first:

| # | Bet | Expected impact on real scans | Effort | GPU-h | Paid-version licence |
|---|---|---|---|---|---|
| 1 | Real-scan benchmark (RealBench-DE, 60 plans) + edit-cost / correction-time metric | Enables everything; no direct gain | 2 people × 3 weeks of annotation + 1 agent | ~0 | OK (own data, with consent) |
| 2 | Redundancy-aware OCR for Maßketten (synthetic-trained recogniser + n-best + chain-sum / pixel-span decoding) | High: dimension recall is the anchor for metric accuracy | 1 agent × 4 weeks | 30–80 | OK (PaddleOCR Apache-2.0) |
| 3 | Synthetic data v2: style and degradation randomisation, non-rectangular, at scale (100k+ pages) | High: the main lever on the domain gap | 1 agent × 4 weeks (CPU-heavy) | ~0 (CPU) | OK if limited to Swiss Dwellings / ResPlan / own data |
| 4 | Multi-head detection model (walls + centreline + junction + opening + text/dimension heads) on the running U-Net work | High at heavy degradation | 1 agent × 6 weeks | 300–800 | OK if trained without CubiCasa5k |
| 5 | Multi-hypothesis geometry scored by the graph check (+ CP-SAT junction assembly, non-Manhattan angles) | Medium–high for closure rate | 1 agent × 4 weeks | ~0 | OK (OR-Tools Apache-2.0) |
| 6 | Consistency-gated self-training on unlabelled real scans | Medium–high once 1–5 exist | 1 agent × 3 weeks | 200–500 | Depends on scan source |
| 7 | LoRA-tuned VLM (Qwen3-VL-8B / Qwen3.5-9B) as page classifier, metadata reader and hypothesis verifier, not as the geometry engine | Medium (scope + verification) | 1 agent × 4 weeks | 150–400 | OK (Apache-2.0) |
| 8 | Scope for energy: multi-storey stacking, heights from sections, stairs, envelope-first metrics | High for HottCAD value; low for the "recognition" score | 1 agent × 6 weeks | ~50 | OK |

---

## 1. State of the art, late 2026 (checked today)

### 1.1 Raster plan → vector / graph

| Approach | Best current example | Reported numbers | Code / weights licence | Fits cluster? | Verdict for Kelpod |
|---|---|---|---|---|---|
| **Detection readout** (junction + centreline heatmaps → graph) | fpvec-lab, arXiv 2608.25608, <https://github.com/Cyprinus12138/fpvec-lab> | CubiCasa5k scans: detect 0.818 vs emit 0.790 wall F1@0.05. The detect advantage grows with plan size. Output fusion +7.1 wall F1. An opening heatmap never trained on gives opening F1 0.25→0.64. Re-scoring against corrected labels moves F1 0.74→0.825, so labels and conventions matter as much as models. | MIT code; ResPlan-FP CC BY 4.0; corrected CubiCasa labels CC BY-NC-SA | Yes (U-Net/HRNet scale) | **Core of v2.** Matches the existing pipeline. |
| **Junction + integer programming** | FloorplanTransformation (ICCV 2017, MIT) | Older, but still the clean way to enforce topology | MIT, unmaintained | Yes | Re-implement the assembly step with OR-Tools CP-SAT (bet 5) |
| **Sequence / polygon decoder** | Raster2Seq, arXiv 2602.09016 (SIGGRAPH 2026) | CubiCasa5k room F1 88.7 (RoomFormer 83.5), corner F1 59.4, angle F1 37.4. ResNet-50 + 6-layer decoder, **256×256 input**, trained on one A6000 for ~19 h + 16 h, 0.52 s/sample. No test on degraded scans. | Code MIT; weights CC BY-NC-4.0 | Yes, easily | **Fusion partner only** (later). 256 px input cannot resolve 1:100 Maßketten. |
| **VLM → JSON** | FloorplanVLM, arXiv 2602.06507 (Qwen2.5-VL-3B, SFT + GRPO) | FPBench-2K: room IoU 0.892, room F1 0.825, opening F1 0.733, validity 96.1 %. Trained on **Floorplan-2M** (proprietary platform data) on **32× H200**. Coordinates quantised to a 1024 grid. ~3k output tokens per plan. Code and weights not released. | Base Qwen2.5-VL-3B is under the Qwen Research licence (per search; **unverified** on the model card) | Data, not compute, is the blocker | **Do not try to replicate** as the main engine (see §8) |
| Community Qwen2.5-VL-3B LoRA on CubiCasa | <https://huggingface.co/manitocross/floorplan-vlm-training>, `mudasir13cs/qwen25-vl-3b-floorplan-grpo` | GRPO rewards wall IoU, room IoU and JSON validity; no credible eval | CC BY-NC (data) + research licence base | Yes | Recipe reference only |
| **Frontier closed VLM zero-shot** | Gemini 3.1 Pro (in fpvec-lab) | Wall F1 0.811@0.05 but 0.472@0.015 on 200 CubiCasa plans: coarse geometry is fine, precise geometry is poor | Closed, cloud | n/a | Not for production (GDPR, precision); fine as an annotation helper on public plans |
| VLM drawing literacy | AECV-Bench, arXiv 2601.04819 | OCR / text QA up to 0.95; door and window **counting 0.40–0.55** | – | – | VLMs read text well and symbols badly, so use them for text and metadata, not symbols |
| Production multimodal pipeline | SALI-FP, arXiv 2609.25615 ("evidence-gated") | 11,534 plans audited; no public metric found | No code found | – | Idea: gate every edit on image evidence. Kelpod's dimension and closure checks are the same idea. |

**Conclusion.** Keep "learned detection → deterministic geometry", make the detector multi-head, and
add a second opinion for fusion (classical + learned, later Raster2Seq-style). Do not switch to an
end-to-end VLM.

### 1.2 VLMs and document models (for OCR, metadata, verification)

| Model | Size | Licence | GPU fit (inference / LoRA training) | Use |
|---|---|---|---|---|
| **Qwen3-VL-8B-Instruct** <https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct> | 8B dense (family: 2B/4B/8B/32B, 30B-A3B, 235B-A22B) | **Apache-2.0** (model card) | Inference on any card in bf16/fp16 (~17 GB). LoRA bf16 fits A100 40 GB / A6000 48 GB with gradient checkpointing. V100 has no bf16: use fp16 or QLoRA (**unverified** stability). | Verifier, page classifier, crop reader |
| **Qwen3.5-9B / 27B** (Feb–Mar 2026, natively multimodal) <https://huggingface.co/Qwen/Qwen3.5-9B> | 9B dense; 27B dense; MoE 35B-A3B, 122B-A10B, 397B-A17B | Apache-2.0 (reported; check each repo) | 9B like above. 27B needs QLoRA on A100/A6000 (**unverified**). | Newer alternative to Qwen3-VL-8B; benchmark both on 100 crops before choosing |
| **PaddleOCR-VL-1.6** <https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6> | 0.9B | Apache-2.0 | Trivially; even 1080 Ti | OCR of crops. German not named on the card (**unverified** quality on `3,51`, `BRH 87,5`). |
| olmOCR-2-7B-1025 | 7B (Qwen2.5-VL-7B base) | Apache-2.0 (card) | Yes | Page-level OCR to text; drawings are not its domain; lower priority |
| MinerU2.5-Pro, GLM-OCR | ~1–2B | Apache-2.0 / MIT (per third-party tables, **unverified**) | Yes | Alternatives if PaddleOCR-VL fails |
| Florence-2 (base 0.23B / large 0.77B) | small | MIT (**unverified** today; from memory) | Yes | Fine-tunable OCR-with-region model; used for GD&T extraction (arXiv 2411.03707) |
| GOT-OCR2.0 | 0.58B | Apache-2.0 (**unverified** today) | Yes | Superseded by PaddleOCR-VL in 2026 tables |
| Surya, OXR-1.0 | – | Revenue-gated weight licences | – | Avoid for the paid version |

The OmniDocBench leader changes monthly (PaddleOCR-VL-1.6 ≈ 96.3; TeleOCR, OvisOCR2, OXR-1.0 slightly
above it per third-party blogs, **unverified**). Document benchmarks do not measure rotated
dimension numbers on drawings, so **our own synthetic + real dimension-crop set is the only
relevant benchmark**.

The engineering-drawing literature points the same way: localise and orient the text, then read
crops. A numeric-field VLM reached F1 0.963 vs 0.672 for alphabetic fields (arXiv 2510.21862). eDOCr2
(MDPI Machines 2025) segments dimensions first, then uses OCR, plus a VLM to verify.

---

## 2. Ranked bets in detail

### Bet 1: RealBench-DE, a real-scan benchmark with an expert-effort metric (do first)

**What.** 60 real *Bestandspläne* pages (40 test, 20 dev, split by building), each annotated as
vector ground truth: wall centrelines + thickness, openings, room polygons + names, dimension text
boxes with values. Report today's metrics plus two expert-effort metrics:

- **Edit cost:** the number of atomic edits (add/delete/move wall, add/delete opening, fix a
  dimension) needed to turn the output into the ground truth, after greedy matching. The idea comes
  from fpvec-lab's edit-cost metric, adapted to Kelpod's graph.
- **Expert correction time:** the pilot (or a trained student) fixes 10 outputs in HottCAD or a
  simple Kelpod editor, timed, against redrawing 3 from scratch (the 3–6 h baseline). This is the
  number that sells: "the expert verifies, not redraws".
- **Energy-weighted errors** (see bet 8): exterior wall length, window area per facade, and heated
  floor area, each as % error. HottCAD's output depends on these far more than on partition walls.

**How many plans.** For the binary "topologically closed" rate, a 95 % Wilson interval at n=40 is
about ±15 pp at p=0.5 and ±12 pp at p=0.8. At n=60 it is ±12 / ±10 pp. Per-wall and per-room metrics
pool hundreds of items but are clustered by plan, so bootstrap **by plan**. Compare systems **paired**
on the same plans (McNemar for closure, paired bootstrap for F1). With pairing, 40 test plans
can detect a closure difference of about 15–20 pp. Under 25 plans, nothing but large effects will
show. Recommendation: **40 test plans is the minimum for a credible headline. Report 95 % CIs. Never
tune on the test split.**

**Where the plans come from** (licence and legal caveats):

| Source | Volume | Rights | Paid version? |
|---|---|---|---|
| Pilot expert's archive | tens–hundreds | Needs the written data-use terms (open question 1 in the context doc). GDPR: addresses and owner names on title blocks must be redacted. | Only with a contract clause |
| Own, family and friends' houses; Hausverwaltungen and Wohnungsbaugesellschaften contacted in discovery interviews | 10–50 | Owner consent; the plans' copyright stays with the architect, but internal research use is low risk (**not legal advice**) | With a consent form |
| Deutsche Digitale Bibliothek / Europeana, filtered for CC0 / Public Domain Mark (e.g. Bauhaus-Uni Weimar CC0, TIB PDM) <https://www.deutsche-digitale-bibliothek.de> | hundreds, mostly pre-1950 | Per-item rights statement; TU Berlin Architekturmuseum items are mixed (some PD, some "nutzung nur nach Autorisierung") | PD/CC0 items yes |
| Wikimedia Commons "Floor plans of buildings in <Land>" categories | tens of relevant ones (many are churches and palaces) | Per-file CC licence | Yes, with attribution |
| SFS-A68 (TU Wien, 68 Austrian apartment buildings 1952–2019, room-function labels) <https://github.com/A2Amir/SFS-A68>, Zenodo 6426871 | 68 buildings | **GPL-3.0** (Zenodo) | **No** |
| Real-estate portals (ImmoScout24 etc.) | thousands | ToS forbid scraping. §60d UrhG allows TDM for non-commercial scientific research regardless of opt-out, but no redistribution. §44b (commercial) respects machine-readable opt-outs. Personal data in exposés. | **No** |
| Municipal Bauakten archives (Hamburg, Wien MA 37 with ~200k buildings) | large | On-site viewing by appointment with a legitimate interest; no open download found | No |

**Annotation tooling.** CVAT Community (MIT, self-hosted) supports polylines and polygons. Its SAM 3
interactive tools appear limited to paid or online tiers (**unverified** for Community). Better:
**pre-annotate with Kelpod itself** and correct. Export Kelpod's `PlanGraph` to CVAT XML or Label Studio
JSON, then fix it. The fix log is itself an edit-cost measurement of the MVP. Label Studio (Apache-2.0)
is the alternative. A thin custom web editor over `PlanGraph` (snap-to-axis, drag nodes) may well be
faster than either for wall graphs. Estimated time: 30–60 min per plan with pre-annotation (**unverified
estimate**; measure it in week 1). That gives 30–60 person-hours for 60 plans.

**Expected gain.** None directly, but it turns every later bet from a guess into a measured number,
and it gives the pitch its first real figure. **GPU:** 0. **Calendar:** 3 weeks. **Risk:** plan access
and consent; annotation drift (write a 1-page annotation guide: centreline vs face, what counts as a
wall, Kamin, Vormauerung). **Paid licence:** fine for own and consented data.

### Bet 2: Redundancy-aware OCR for Maßketten

**What.** Today: RapidOCR with 0°/90° passes, dimension recall 72 % / precision 78 % on synth, and
75 % of runtime. v2 has four parts:

1. **Synthetic dimension-crop generator** (CPU): millions of crops of `3,51`, `11,24`, `88,5`, `76`,
   superscript mm (`1,01⁵`), `BRH 87,5`, `Brh.=90`, `14,52 m²`, `qm`, with DIN 6776-style stencil
   fonts, handwriting-like fonts, pen jitter, ticks and lines crossing the text, rotation 0/90/±δ, and
   Augraphy degradation (MIT, <https://github.com/sparkfish/augraphy>). Crops from real scans
   (bet 1 dev split) for validation.
2. **Fine-tune the recogniser:** PP-OCRv5 rec (Apache-2.0; export to ONNX so RapidOCR still runs it)
   or a small PARSeq/TrOCR. The alphabet is tiny (digits, `,`, `.`, `m²`, a few words), so a dedicated
   model should beat a generic one. Train the **detector** on rotated text boxes from the synth
   renderer too, or add a text/dimension-line head to bet 4's network and drop the second OCR
   detection pass.
3. **n-best + joint decoding.** Keep the top-k strings per box. Choose the combination that best
   satisfies: (a) value ≈ pixel span × scale (the scale is estimated robustly from all chains); (b) the
   segments of a chain sum to its overall dimension (*Gesamtmaß*) and to the parallel chain on the
   opposite facade; (c) room stamp area ≈ polygon area. This is a small discrete search (beam or
   CP-SAT) and catches classic misreads (`3,51`↔`3,57`, missing comma, cm vs m).
4. **VLM fallback for low-confidence crops:** PaddleOCR-VL-1.6 (0.9B) or Qwen3-VL-8B zero-shot, then
   LoRA-tuned on the same synthetic crops. Only for the residual ~10 % of crops, so runtime stays small.

**Evidence.** Drawing-OCR papers converge on "localise and orient, then read crops, then verify"
(eDOCr2; arXiv 2510.21862 reports numeric F1 0.963). AECV-Bench: VLM text reading is strong (≤0.95).
The chain-sum constraint is specific to Maßketten. It is free information that no competitor claims to
use (Hott-KI's public material does not mention dimension chains).

**Expected gain (target).** Synth dimension recall 72 % → ≥ 92 % at precision ≥ 95 %; OCR time
3.5 s → < 1.5 s per page on GPU. On real scans, unknown until bet 1. **GPU:** 30–80 h (recogniser
fine-tunes on 1080 Ti / V100; VLM LoRA on A6000). **Calendar:** 4 weeks. **Risks:** handwritten
corrections on old plans (strike-throughs, values written over); chains in feet or old units
(rare); plans whose chains disagree with the drawing (dimensions are authoritative, so this is
fine). **Paid licence:** all Apache/MIT.

### Bet 3: Synthetic data v2, the main lever on the domain gap

**What.** Extend the generator and the finetune agent's renderer (`training/kelpod_train/render.py`)
in three ways:

- **Geometry sources:** Swiss Dwellings v3.0.0 (45,176 apartments, ~3.1k buildings, CC BY 4.0,
  <https://zenodo.org/record/7788422>; derived from real Swiss existing-building plans, which includes
  non-orthogonal walls and multi-unit floors), MSD (5.3k plans), ResPlan (17k, CC BY 4.0, provenance
  caveats), plus the own generator extended to L/T/U footprints, oblique walls, bay windows,
  chimneys (*Kamine*), stairs and multi-storey stacks with consistent outlines.
- **Style randomisation:** the strongest known recipe for sim-to-real. Hatching (45° single,
  cross-hatch, poché fill, none), line weights, wall-outline vs filled, door symbols (arc, no arc,
  1950s style), window symbols (2/3 lines, sill line), Maßkette style (ticks, dots, arrows,
  extension-line variants, chains on 1–3 levels), stamps, title blocks, north arrows, furniture,
  sanitary symbols, handwritten red-pen annotations, Lichtpause blue/brown background,
  pencil-on-tracing-paper look, multiple fonts including stencil and hand lettering.
- **Degradation:** Augraphy (MIT, 26 augmentations: ink bleed, dirty rollers, folding, bad photocopy,
  shadow) + existing degrade + scan geometry (perspective, page curl, A3 stitching seams, partial
  crops, 150–600 dpi, JPEG Q 20–95). Later, an **unpaired learned degradation** (CycleGAN- or
  diffusion-style, see DECDM, WACV 2024) trained on bet 1's unlabelled real scans. Only do this if the
  real-vs-synth gap stays large after Augraphy. No ready 2025–26 diffusion "degrade my documents" model
  was found (**unverified** that none exists).

Scale: 100k–300k rendered pages is CPU work. Rendering at ~1 s/page on 32 cores gives 100k pages in
about an hour of node time. Measure the gap with a **domain classifier** (if a small CNN can tell
synth from real with AUC > 0.95, the style is not realistic enough) and with dev-split metrics.

**Expected gain (target).** On RealBench-DE dev, cut the synthetic→real drop in wall F1 and room IoU
by half compared with a model trained on today's generator. **GPU:** ~0 for data (CPU). **Calendar:**
4 weeks, overlapping bet 4. **Risks:** style coverage is guesswork until real plans are seen, so
collect 200+ unlabelled real scans early to steer it. Swiss plan conventions differ from German ones
(Swiss chains are usually in metres too, close enough). **Paid licence:** fine if built only from
Swiss Dwellings / MSD / ResPlan / own geometry. **Not** from CubiCasa5k, ArchCAD-400K (CC BY-NC-4.0,
gated, <https://huggingface.co/datasets/jackluoluo/ArchCAD>) or SFS-A68 (GPL).

### Bet 4: A multi-head detection network, the "detect" readout done properly

**What.** Grow the finetune agent's 5-class U-Net into one network with dense heads:

- wall mask with thickness-class channels (exterior / load-bearing / partition), plus a **wall
  centreline** heatmap;
- **junction heatmap with type** (L, T, X, end), as in FloorplanTransformation and fpvec-lab;
- opening endpoint heatmaps + door / window / french-door class; door hinge side + swing;
- room interior + room-boundary (affinity) map;
- **dimension-line, tick and dimension-text** masks (feeds bet 2 and removes the line-removal hack);
- stair mask + run direction (feeds bet 8).

Backbone: ConvNeXt-T/S or HRNet-W32 (timm, Apache) at 1024–1536 px tiles. A 2026 DINOv3/SAM-style
encoder is optional (**check its licence first**; DINOv3 has a custom licence, **unverified**).
Readout: the existing deterministic vectoriser + bet 5's CP-SAT assembly. Keep the classical
segmenter as a second opinion for **output fusion** (fpvec-lab: +7.1 wall F1 from deterministic
fusion). Pretraining order: ResPlan-FP / Swiss renders → own DIN synth → (bet 6) pseudo-labelled real.
CubiCasa5k may be added in this repo for a measured comparison but must stay out of any paid
checkpoint.

**Expected gain (target).** At degrade 1.0 on synth: room recall 0.72 → ≥ 0.92, door recall 0.81 →
≥ 0.93. On RealBench-DE: a measured ≥ 10 pp wall-F1 gain over classical at 10 cm. **GPU:**
300–800 GPU-h total (≈ 20–40 runs of 12–24 h on A100/A6000 within the 2-day limit; ablations on
V100 within its 4-day limit; 1080 Ti for evaluation sweeps). **Calendar:** 6 weeks. **Risks:**
precise thin-line localisation at 200 dpi (1 px ≈ 1.27 cm; consider 300 dpi for the junction head);
over-fitting to the renderer. **Paid licence:** OK with permissive data only.

### Bet 5: Multi-hypothesis geometry scored by the graph check

**What.** Today the geometry step makes one guess. v2 generates K candidates (K ≈ 8–32) by varying:
segmentation source (classical / learned / fused); thresholds; OCR n-best combinations; snapping
mode (Manhattan / dominant-angle / free); gap bridging. Each candidate goes through the existing fit
and check and gets a score: closure (hard), dimension residual, stamp-area agreement, opening-in-wall
validity, wall-thickness consistency with classes, prior plausibility (room sizes, door widths
0.76/0.885/1.01), and later the VLM verifier (bet 7). The best one is kept, and the runner-up goes to
the UI as "alternative".

Two building blocks:

- **CP-SAT junction assembly** (OR-Tools, Apache-2.0): choose which candidate wall segments and
  junctions exist, subject to degree constraints, no crossings, rooms closed, and dimensions
  matched within tolerance. This is the FloorplanTransformation integer program, modernised, and it
  guarantees closure by construction where a solution exists.
- **Non-Manhattan walls:** estimate dominant directions per plan from a line-angle histogram, snap
  to those (handles a 7° Altbau wing, 45° bays and oblique plots). The `least_squares` fit gains
  angle parameters only where the histogram demands them.

**Evidence.** FloorplanTransformation (IP gives topology); fpvec-lab (post-processing and
reconciliation matter as much as the representation); SALI-FP (evidence-gated edits). Kelpod's
two synth failures (seeds 2, 9: one missing room, dangling ends) are exactly the kind that a
candidate set + closure score fixes.

**Expected gain (target).** Closed share on synth degrade 0.5: 8/10 → ≥ 19/20. On real scans: a
measured gain in closure at equal or better room IoU. **GPU:** 0 (CPU; K candidates × ~0.1 s geometry).
**Calendar:** 4 weeks. **Risks:** a scoring function that rewards closed-but-wrong (guard with room IoU
on dev); combinatorial blow-up (cap K, prune). **Paid licence:** OK.

### Bet 6: Consistency-gated self-training on unlabelled real scans

**What.** Collect 500–5,000 unlabelled real plan pages (pilot archive, PD archives; in this
educational repo also §60d research TDM). Run the bet 4 + 5 pipeline. **Keep only plans or regions where the
independent checks agree:** closed graph, dimension residual < 2 cm, stamp areas within 5 %, chain
sums consistent. Retrain on those pseudo-labels plus synth. Iterate 2–3 rounds (teacher → student,
strong augmentation on the student). The graph check is an unusually good confidence filter because
it is independent of the network.

**Expected gain (target).** Close a further part of the real-vs-synth gap left after bet 3. Measured
on RealBench-DE dev, never test. **GPU:** 200–500 h (2–3 rounds). **Calendar:** 3 weeks, after
bets 1–5 produce a usable teacher. **Risks:** confirmation bias (errors that pass the checks get
reinforced; audit 30 accepted pseudo-labels by hand each round); data rights. **Paid licence:** only
on data the company has rights to (pilot contract); listing-portal scans **cannot** be used.

### Bet 7: A LoRA-tuned VLM as reader, classifier and verifier (not the geometry engine)

**What.** One Apache-2.0 VLM (Qwen3-VL-8B-Instruct or Qwen3.5-9B; pick by a 1-day bake-off on 100
crops and pages) fine-tuned with LoRA for:

1. **Page understanding:** sheet type (Grundriss / Schnitt / Ansicht / Lageplan / Detail), storey
   (KG/EG/OG/DG/Spitzboden), scale ("M 1:100", "1:50"), north arrow, title-block fields (year, which
   helps the U-value era for HottCAD), and redaction of personal data.
2. **Crop reading** (bet 2 fallback) and **room-stamp reading**.
3. **Verifier:** given the plan and a rendered overlay of candidate k, return "consistent / which
   region is wrong". This is a learned term in bet 5's score. Train it on synth pairs (correct overlay
   vs perturbed overlay) and RealBench-DE dev corrections. An RL-style (GRPO) stage with the closure
   and dimension checks as reward is possible but not needed in v2.

**Why not plan → JSON end-to-end?** FloorplanVLM needed 2M pairs and 32 H200s to reach room IoU
0.89. It quantises coordinates to a 1024 grid, which is far too coarse for a 15 m façade at 1 cm.
AECV-Bench shows VLMs are weak at symbols. Gemini 3.1 Pro drops to 0.47 wall F1 at the tight
threshold. A VLM here is a reader and a judge.

**Feasibility.** 8–9B LoRA (rank 16–64) in bf16 with gradient checkpointing fits one A100 40 GB or
A6000 48 GB at 1–2k-px images; ~10–40 GPU-h per run on 50–200k samples (**unverified estimate**).
Inference ~17 GB bf16, or 4-bit for the pilot's dual-GPU server. V100: fp16 only, no FlashAttention-2
(**unverified** for Qwen3.5). **Expected gain (target):** sheet type and storey accuracy ≥ 98 % on
RealBench-DE; scale string read on ≥ 95 % of pages; verifier lifts top-1 hypothesis selection by a
measured margin over the hand-made score. **GPU:** 150–400 h. **Calendar:** 4 weeks. **Risks:**
hallucinated numbers (never let the VLM write geometry or dimensions directly; it only selects among
candidates or flags issues); latency (~1–3 s per call on A100; keep calls per plan < 20). **Paid
licence:** Apache-2.0 for Qwen3-VL-8B (checked on the card) and Qwen3.5 (reported; check each repo).
**Avoid Qwen2.5-VL-3B** (research licence, per search, **unverified**) and Gemini or other cloud APIs
for customer data.

### Bet 8: Scope that matters for HottCAD and energy

What an energy expert needs from the model is the **thermal envelope**: exterior wall areas per
orientation, window areas per facade, roof and top-floor ceiling, floor against ground or basement,
heated volume and storey heights. Interior partitions matter far less (room-wise heat load is a
secondary use). v2 should therefore:

1. **Envelope-first metrics** (in bet 1): exterior outline IoU, perimeter error, window area per
   orientation, gross / net floor area, heated volume. These track what HottCAD computes.
2. **Multi-storey stacking:** classify sheets (bet 7), then register storeys by aligning exterior
   outlines (rigid 2D fit on the outline polygon plus stair position and load-bearing walls, using
   ICP or a RANSAC fit on corners). Enforce shared exterior walls, and write one `IfcBuildingStorey` per
   sheet with elevations. Basement and attic are flagged as possibly unheated, for the expert to
   decide.
3. **Heights from sections (Schnitt):** OCR level markers (`±0,00`, `+2,75`, `OK FFB`, `OK RD`,
   `UK Decke`, `Kniestock`) and the vertical chains in the section; storey height = difference of OK FFB
   levels; ceiling thickness; parapet heights. Fallback: 2.50–2.75 m default flagged as "assumed".
   A section also gives the **roof pitch and Kniestock**, which drives the top-floor envelope and is
   arguably worth more for energy than any interior wall.
4. **Stairs:** detection head (bet 4) + run line; export as `IfcStair` (minimal) and, more important,
   as the void in the ceiling and the alignment anchor across storeys. No floor-plan stair dataset
   was found (HouseLayout3D, arXiv 2512.02450, is scan-based), so stairs come from synth.
5. **HottCAD import specifics:** still unknown publicly. HottCAD IFC import is a paid or
   maintenance-contract option (Hottgenroth catalogue 2021, exhibitor sheet 2024). A Graphisoft forum
   claims it does not read façades (**unverified**). There is no public documentation of IfcSpace
   handling. **Action:** with the pilot, round-trip 5 hand-made IFCs that vary one feature each (IFC2X3
   vs IFC4, IfcSpace with/without boundaries, IfcSlab, storey elevations, IfcWindow types) and encode
   what imports in an **IDS** file (IfcTester). This is cheap and has the highest certainty of value.

**Expected gain (target).** 2–3 storey buildings exported as one IFC with consistent outlines;
storey heights read from sections on ≥ 80 % of buildings that have a section sheet. **GPU:** ~50 h.
**Calendar:** 6 weeks, mostly CPU. **Risks:** many *Bestandspläne* have no section; storeys drawn at
different scales or offsets; Hott-KI already "erkennt Bauteile stockwerksweise"
(<https://www.holz-handwerk.de/en/exhibitors/hottscan-gmbh-2488561/the-modeler-package-gold-2561513>),
so the differentiator remains dimension-exact, closed and audited. **Paid licence:** OK.

---

## 3. GPU budget on this cluster

Partitions: `ampere` (A100 40 GB, A6000 48 GB; 2-day limit), `volta` (3× V100 32 GB; 4 days), `pascal`
(1080 Ti; no limit). Assume about 2–4 GPUs available at once on average (shared cluster).

| Bet | GPU-h | Where |
|---|---|---|
| 2 OCR recogniser + VLM crop reader | 30–80 | pascal / volta; VLM on ampere |
| 4 Multi-head detector | 300–800 | ampere (main runs ≤ 48 h, with checkpoint-resume); volta for ablations |
| 6 Self-training (2–3 rounds) | 200–500 | ampere / volta |
| 7 VLM LoRA | 150–400 | ampere only (bf16) |
| 8 Stairs, section heads | ~50 | any |
| Inference on RealBench + unlabelled sets | ~50 | pascal |
| **Total** | **~800–1,900 GPU-h** | ≈ 3 GPUs × 11–26 days of wall time over 12 weeks: feasible, with queue slack |

Every job must checkpoint and resume (2-day limit on ampere).

---

## 4. Phased plan for the rest of the semester (12 weeks, 24 Nov 2026 – mid Feb 2027)

All numbers are **targets**.

| Phase | Weeks | Work | Targets (exit criteria) |
|---|---|---|---|
| **P0 Measure** | W1–W3 (24 Nov – 14 Dec) | Data-use terms with the pilot; collect ≥ 200 real pages (≥ 60 to annotate); annotation guide; pre-annotate with Kelpod and correct in CVAT or a thin editor; edit-cost + envelope metrics in `kelpod.evaluation`; first MVP number on RealBench-DE. HottCAD round-trip test with the pilot → IDS file. | 60 annotated pages (40 test / 20 dev, split by building); MVP baseline with 95 % CIs; annotation time per plan measured; IDS v1 |
| **P1 Build** (parallel) | W2–W7 (1 Dec – 11 Jan; holidays included) | Synth v2 (bet 3); OCR crop generator + recogniser + n-best/chain-sum decoder (bet 2); multi-head detector trained on synth v2 (bet 4); multi-hypothesis + CP-SAT (bet 5); VLM bake-off and page classifier (bet 7a) | Synth (degrade 1.0): room recall ≥ 0.92, dimension recall ≥ 0.92 at P ≥ 0.95, closed share ≥ 95 %. RealBench dev: wall F1 +10 pp over MVP. Sheet type ≥ 98 %. |
| **P2 Adapt** | W7–W10 (5 Jan – 1 Feb) | Self-training round 1–2 (bet 6); output fusion classical + learned; VLM verifier in the hypothesis score (bet 7c); multi-storey stacking and section heights (bet 8) | RealBench dev: closed share ≥ 70 %; edit cost −50 % vs MVP; 3 multi-storey buildings exported and imported in HottCAD by the pilot |
| **P3 Freeze and report** | W11–W12 (2 – 15 Feb) | Freeze; run RealBench test once; timed correction study with the pilot (≥ 10 plans); `docs/MVP_REPORT.md`-style v2 report; pitch numbers | Reported (whatever they are): closed share, room IoU, wall F1, dimension recall, envelope errors, edit cost and **expert minutes per plan** vs the 3–6 h redraw baseline; runtime < 60 s/plan on one GPU |

**Staffing across parallel agents** (each owns one folder and talks through `contracts.py`, as today):

| Agent | Owns | Phases | Notes |
|---|---|---|---|
| A **eval** | `kelpod.evaluation` (RealBench loader, edit cost, envelope metrics, CI bootstrap), annotation export/import | P0 → P3 | The founders do the annotation and pilot contact; the agent builds the tools |
| B **data** | `training/kelpod_train/render.py`, synth v2, Augraphy, domain classifier | P1 → P2 | CPU-heavy; hands datasets to C and D |
| C **finetune** (existing agent) | multi-head detector, fusion, self-training | P1 → P2 | Continue the current U-Net work; add heads |
| D **ocr** | crop generator, recogniser fine-tune, n-best + chain-sum decoder, VLM fallback | P1 → P2 | |
| E **geometry** | multi-hypothesis, CP-SAT assembly, dominant angles, scoring | P1 → P2 | |
| F **scope** | sheet classification glue, storey stacking, section heights, stairs, IFC/IDS for HottCAD | P0 (IDS) + P2 | Needs the pilot |
| G **vlm** | Qwen3-VL / Qwen3.5 bake-off, LoRA classifier/reader/verifier | P1 → P2 | Shares ampere time with C: schedule jobs |

Humans: one founder runs pilot data, consent and HottCAD tests (P0 critical path). Both founders
annotate about 30 h each in P0.

---

## 5. What would NOT be worth doing (this semester)

- **Replicating FloorplanVLM or another end-to-end VLM → JSON as the main engine.** It needs millions
  of paired plans and 32-H200-class compute. Its 1024-grid coordinates are too coarse for
  dimension-exact output, and VLMs are measurably weak at drawing symbols (AECV-Bench).
- **Pretraining any backbone or foundation model.** Fine-tune pretrained encoders only.
- **Chasing CubiCasa5k / Structured3D leaderboard numbers** or porting density-map methods (RoomFormer,
  PolyRoom, FRI-Net, HEAT). Different input domain; numbers do not transfer to German scans.
- **Training Raster2Seq-style sequence models from scratch now.** Useful later as a fusion partner.
  Its 256 px input and the fpvec-lab real-scan result make it a weak first choice.
- **Diffusion-based learned degradation before Augraphy + style randomisation has been measured.**
- **Large-scale scraping of listing portals for the paid version** (ToS, §44b opt-outs, personal
  data). Fine only as §60d research data in this educational repo, never redistributed.
- **Reconstructing 3D from elevations beyond heights and roof pitch**, and building the DIN V 18599
  engine (Phase 2 of the business plan; out of scope for a recognition v2).
- **More hand-tuning of the classical segmenter at heavy degradation.** Keep it as a fusion partner
  and fallback; put new effort into the learned heads.
- **Using Surya, OXR-1.0, Qwen2.5-VL-3B, CubiCasa5k, ArchCAD-400K, SFS-A68 or Raster2Seq weights in
  anything destined for the paid version** (revenue-gated, research-only, NC or GPL).

---

## 6. Licence flags for the paid version (summary)

| Item | Status | Paid version |
|---|---|---|
| Qwen3-VL-8B-Instruct | Apache-2.0 (model card, checked) | Yes |
| Qwen3.5-9B / 27B | Apache-2.0 (reported; check each repo) | Yes, after checking |
| Qwen2.5-VL-3B (FloorplanVLM base) | Qwen Research licence per search (**unverified**) | No |
| PaddleOCR-VL-1.6, PP-OCRv5 | Apache-2.0 (card checked for VL-1.6) | Yes |
| olmOCR-2-7B | Apache-2.0 (card) | Yes |
| Augraphy | MIT | Yes |
| OR-Tools, timm, segmentation_models_pytorch | Apache-2.0 / Apache-2.0 / MIT | Yes |
| Swiss Dwellings, MSD data, ResPlan / ResPlan-FP | CC BY 4.0 (ResPlan has provenance and takedown caveats) | Yes, with attribution |
| CubiCasa5k, ArchCAD-400K, Raster2Seq weights, fpvec corrected CubiCasa labels | CC BY-NC(-SA) | No |
| SFS-A68 | GPL-3.0 (Zenodo) | No (as training data, unclear; avoid) |
| Portal-scraped plans | §60d research only | No |
| Pilot plans | Needs contract | Only with an explicit clause |
| Gemini / cloud VLMs on customer plans | GDPR / zero-retention conflict | No |

---

## 7. Unverified items and open questions

- HottCAD's exact IFC import behaviour (IfcSpace, space boundaries, IFC4 vs 2X3, façades). Only
  forum hearsay was found. Resolve with the pilot (bet 8.5).
- Hott-KI's accuracy and whether it reads dimension chains. Public material (BAKA Laudatio 2025,
  GebäudeScan 2026) does not mention Maßketten.
- Whether Swiss Dwellings ships raster plan images (it ships geometry; images **unverified**).
- Qwen2.5-VL-3B licence on the card; Florence-2 / GOT-OCR2 licences (from memory); PaddleOCR-VL German
  quality; MinerU2.5-Pro and GLM-OCR licences (third-party tables).
- CVAT Community support for SAM 2/3 interactors (appears to be paid tiers).
- GPU-hour and annotation-time estimates are engineering estimates, not measurements.
- Whether a ready diffusion model for realistic scan degradation exists (none found).

## Sources

- fpvec-lab / "When should a network emit geometry…": <https://github.com/Cyprinus12138/fpvec-lab>, <https://arxiv.org/abs/2608.25608>
- Raster2Seq: <https://arxiv.org/abs/2602.09016>, <https://cornell-vailab.github.io/Raster2Seq/>
- FloorplanVLM: <https://arxiv.org/abs/2602.06507>
- SALI-FP (evidence-gated parsing): <https://arxiv.org/abs/2609.25615>
- AECV-Bench: <https://arxiv.org/abs/2601.04819>
- ResPlan: <https://arxiv.org/abs/2508.14006>; ArchCAD-400K: <https://arxiv.org/abs/2503.22346>, <https://huggingface.co/datasets/jackluoluo/ArchCAD>
- Community VLM floor-plan LoRAs: <https://huggingface.co/manitocross/floorplan-vlm-training>, <https://huggingface.co/mudasir13cs/qwen25-vl-3b-floorplan-grpo>
- Qwen3-VL: <https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct>, report <https://arxiv.org/abs/2511.21631>; Qwen3.5: <https://huggingface.co/Qwen/Qwen3.5-9B>, <https://simonwillison.net/2026/Feb/17/qwen35/>
- PaddleOCR-VL-1.6: <https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6>; olmOCR-2: <https://huggingface.co/allenai/olmOCR-2-7B-1025>
- OCR leaderboards (third-party): <https://www.docsumo.com/blog/best-ocr-models>, <https://roboflow.com/blog/best-open-source-ocr-models>, <https://www.koncile.ai/en/ressources/10-open-source-ocr-tools-you-should-know-about>
- Engineering-drawing OCR: eDOCr2 <https://mdpi.com/2075-1702/13/3/254>, <https://arxiv.org/abs/2510.21862>, Florence-2 GD&T <https://arxiv.org/abs/2411.03707>
- Augraphy: <https://github.com/sparkfish/augraphy>, <https://arxiv.org/abs/2208.14558>; DECDM: <https://arxiv.org/abs/2311.09625>; PreP-OCR: <https://arxiv.org/abs/2505.20429>
- Swiss Dwellings: <https://zenodo.org/record/7788422>; MSD: <https://doi.org/10.4121/e1d89cb5-6872-48fc-be63-aadd687ee6f9>
- SFS-A68: <https://github.com/A2Amir/SFS-A68>, <https://zenodo.org/record/6426871>
- Versailles-FP <https://arxiv.org/abs/2103.08064>; MLSTRUCT-FP <https://github.com/MLSTRUCT/MLStructFP>
- German TDM law: §60d <https://www.gesetze-im-internet.de/urhg/__60d.html>, §44b <https://www.gesetze-im-internet.de/urhg/__44b.html>, commentary <https://www.uni-bremen.de/urheberrecht/wissensplattform/7-sonderfall-text-und-data-mining>
- Archives: DDB <https://www.deutsche-digitale-bibliothek.de>, Wien MA 37 Planarchiv <https://www.wien.gv.at/kontakt/ma37-baupolizei-planarchiv-digitale-hauseinlage>, Hamburg Bauakten <https://www.hamburg.de/service/info/11254021/n0/>
- HottCAD / Hott-KI: <https://bakaberlin.de/wp-content/uploads/2025/01/06-2463-24-Laudatio-Hottgenroth-P2.pdf>, <https://www.baulinks.de/webplugin/2026/1119.php4>, <https://haustec.de/sites/default/files/2021-02/gesamtkatalog_2021.pdf>, <https://community.graphisoft.com/t5/Dokumentation/IFC-als-Austauschformat/m-p/439246>, <https://www.holz-handwerk.de/en/exhibitors/hottscan-gmbh-2488561/the-modeler-package-gold-2561513>
- CVAT SAM 3: <https://www.cvat.ai/resources/blog/sam-3-image-segmentation>
- HouseLayout3D (stairs, multi-floor): <https://arxiv.org/abs/2512.02450>
