# Kelpod: project context

Read this first. It is the single source of truth for what Kelpod is and where it stands.

## In one sentence

Kelpod turns old German architectural floor plans (*Bestandspläne*, usually degraded PDF scans) into
topologically closed 3D building models (IFC 2x3 / IFC4) that energy-calculation software such as
HottCAD can import directly.

## Team and context

- **Founders:** Arsha Fahima Tajdeen and Kevin Antony.
- **Course:** Frontier Systems, WiSe 2026/2027 (L3S, Leibniz University Hannover). Project-based:
  build a real "AI for Good" startup, using AI agents as employees.
- **Graded on:** periodic status updates (6–7 min, one speaker), a ~30-min final video (includes a
  10-min investor pitch), and the code on GitHub (well structured, documented, built iteratively).
- **Course asks every update to cover:** current status, progress since last update, customers and
  business (who you talked to, what you learned, what changed), obstacles and lessons, next steps.
  It also looks for: a concrete, checkable AI-for-Good benefit; customers, who pays, competitors,
  revenue; tests with real users; agents as employees; measurable goals with numbers and dates.

## Problem

- The German Building Energy Act (GEG) and EU rules mean millions of existing buildings need energy
  audits, heat-load calculations and renovation roadmaps (iSFP).
- Most existing buildings only have degraded PDF scans or old paper blueprints.
- Energy experts spend **3–6 hours per plan** redrawing it by hand in HottCAD (Hottgenroth / ETU),
  which dominates government-certified energy calculations.
- Existing floor-plan AI (DeepFloorPlan, RoomFormer) is trained on US plans: it misreads German
  DIN 1356-1 conventions (wall hatching, door swings, rotated dimension chains / *Maßketten*) and
  outputs pixel masks, which BIM software cannot use without closed vector geometry.

## Product (first version)

**Input:** PDF (single or multi-page), JPG, PNG, TIFF.

**Pipeline (each step is an agent; deterministic tools do geometry and maths):**

1. **Clean the scan:** deskew, remove stains and fold marks, normalise scale/DPI.
2. **Segment:** walls (exterior solid masonry vs. interior drywall, from DIN 1356-1 hatching), windows
   and doors (swing arcs, line weights), stairs (step counts, direction arrows).
   Ideas to test: adapt DeepFloorPlan's attention to interrupted wall lines and junctions; stronger
   geometric relations between neighbouring corners in RoomFormer.
3. **OCR the annotations:** dimension chains (override pixel scale), sill heights (e.g. "BRH 87.5"),
   room stamps (e.g. "Wohnen, 14,5 m²").
4. **Fit to dimensions:** associate each dimension with its wall segment and use them as constraints in
   a geometric optimisation; snap walls to intended angles; close every node.
5. **Check as a graph:** walls, openings and rooms as one connected spatial graph; catch disconnected
   walls and open rooms; compare alternative layouts. Later: cross-check floor plans against sections
   and elevations of the same building.
6. **Export:** extrude to 3D with standard room heights (2.50–2.75 m) and extracted sill/lintel heights;
   write IFC 2x3 and IFC4 (primary, for HottCAD); DWG/DXF and gbXML as secondary outputs.

**Targets:** under 3 minutes per plan; over 95% geometric closure (the expert verifies, not redraws).
**Deployment:** SaaS API on GDPR-compliant servers in Frankfurt, or an on-premise Docker container on
the customer's NVIDIA GPUs. Zero retention on SaaS unless the customer opts in to training.
**Privacy:** German GDPR (DSGVO). AI output is not a certified measurement; a qualified expert approves
every result.

## Open-source building blocks (prefer permissive licences)

| Step | Tools |
|---|---|
| Clean scan | OpenCV (Apache-2.0), scikit-image (BSD), pypdfium2 (Apache/BSD) |
| Segment | RoomFormer (MIT), FloorplanTransformation (MIT), Detectron2 / MMSegmentation (Apache-2.0); labelling: CVAT (MIT), Label Studio (Apache-2.0) |
| OCR | PaddleOCR (Apache-2.0, good for rotated text), docTR (Apache-2.0), Tesseract (Apache-2.0), TrOCR / Donut (MIT) |
| Fit to dimensions | Shapely (BSD), SciPy optimize (BSD), CVXPY (Apache-2.0), OR-Tools (Apache-2.0) |
| Check as graph | NetworkX (BSD), Shapely `polygonize` |
| Export | IfcOpenShell (LGPL-3.0), ezdxf (MIT) |
| App | FastAPI (Python) backend, Next.js + Tailwind frontend |

Avoid for the commercial product unless the licence is resolved: DeepFloorPlan (GPL-3.0), Ultralytics
YOLO and PyMuPDF (AGPL-3.0), the CubiCasa5k dataset (non-commercial as far as known).

## Users, market, business

- **Market:** DACH region. Energy-efficiency experts (EEE), building surveyors, architects, housing
  companies (*Wohnungsbaugesellschaften*).
- **Personas:** (1) independent EEE: very technical, hates manual CAD tracing, values data sovereignty,
  runs local GPUs, needs 95%+ accurate HottCAD imports; (2) draftsman at a housing company with 10,000+
  units: batch-processes archive PDFs into IFC for portfolio-wide ESG reporting.
- **Revenue:** 30–80 EUR per plan; monthly licence for firms; API / white-label for large clients.
- **Competitors:** Hottgenroth (HottCAD), ENVISYS (EVEBI).
- **AI for Good:** faster audits → more renovations → less energy use and CO₂.

## Strategy: integrate first, then replace

- **Phase 1 (now):** Kelpod produces the model; HottCAD does the certified calculation. Wins trust,
  real data and revenue.
- **Phase 2 (later):** Kelpod does the model *and* the certified energy calculation, replacing HottCAD.
  This needs a DIN V 18599 calculation engine (mandatory method under the GEG since 2024) validated
  against the industry test cases (18599 Gütegemeinschaft seal: 32 residential validation cases).
  Options: license a validated engine (e.g. check Fraunhofer IBP's IBP:18599) or build and validate one.
  iSFP output must use the official iSFP print application (v2.6 from 1 Sep 2026). Only experts
  authorised under §88 GEG may issue certificates and iSFPs, so experts stay Kelpod's users.
- Later platform features: energy estimates, renovation options with costs, best order of measures
  within a budget (EnergyPlus / OpenStudio, pymoo).

## Pilot and traction

- **One pilot customer is locked in:** an independent energy-efficiency expert (keep their name out of
  this public repo). They already work on converting German plans to BIM and run a dual-GPU server.
- **Exchange:** Kelpod installs a free alpha (Docker) on their server; in return, real *Bestandspläne*
  to train on, and their expertise to map exactly which IFC attributes HottCAD needs.
- **Still to clarify:** written terms for data use and IP ownership of jointly developed work.
- **Semester validation targets:** 15 discovery interviews, 2 pilot partners (feedback from 5+ end
  users), 3 willingness-to-pay conversations.

## Roadmap (semester)

| When | Milestone |
|---|---|
| Weeks 3–6 | Plan to IFC model working on 3 sample plans |
| Weeks 7–10 | Energy estimates + 3 renovation options per building (each with cost range, energy/CO₂ impact, order of work) |
| End of semester | Demo on 10 buildings, reviewed with pilot partners |

## Where things are

- **Code repo:** github.com/kevnantony/kelpod (public). Work on branch `kelpod-pivot-slides` or a branch
  cut from it; it is not pushed yet.
  - `presentation/`: the current slides (LaTeX Beamer, `make slides | handout | notes`, built with
    Tectonic), `speaker-notes.md` (with `[click]` cues and a Q&A glossary), PDFs in `presentation/pdf/`.
  - Everything else in the repo (`src/`, `web/`, `deploy/`, `bench/`, `configs/`, `tests/`, `docs/`,
    `presentation/archive/`) is left over from an earlier, abandoned idea. Do not build on it; remove or
    archive it when restructuring the repo for this product.
- **Published slides:** https://kevinantony.de/fs-slides/ (website repo
  `~/Desktop/Study/kevnantony.github.io`, folder `fs-slides/`; update it by copying new PDFs there).

## Working environment and conventions

- Dev machine: Apple M4 Pro, 24 GB unified memory, **no NVIDIA GPU** (Apple MLX is available).
  GPU training/inference runs on the pilot's server or rented GPUs.
- Python via `uv`; LaTeX via Tectonic.
- Ask before pushing to `main` or publishing anything. Commit messages end with the attribution line
  the session gives.
- Slides: dead simple, minimum text, technical terms kept where they matter; animations are
  appear-only (presented from PDF); speaker notes in plain language with `[click]` cues.
- Never state targets as results. Mark ideas that are not built yet as ideas.

## Open questions

1. Data-use and IP terms with the pilot expert.
2. Which IFC attributes HottCAD needs for an error-free import (first task with the pilot).
3. Training data: how many annotated German plans, and how to label them.
4. Phase 2: license or build the DIN V 18599 engine.
5. Product and repo naming are settled as "Kelpod"; the Python package name for the new code is open.
