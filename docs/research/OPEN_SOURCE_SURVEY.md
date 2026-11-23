# Open-source survey: raster floor plan → vector → IFC

Status: research note, 2026-11-23. Stars and last-push dates were read from the GitHub API on that
date. Licences were checked against each repo's `LICENSE` file or README unless marked
**unverified**. This note is not legal advice: before shipping, have a person re-check any licence the
product depends on.

Legend for "Use": **use** = adopt as a dependency as is · **fine-tune** = use the code, retrain on
our own (German) data · **ideas** = read the paper or code, write our own · **avoid** = licence or fit
rules it out for the product.

"No NVIDIA GPU" means the dev Mac (CPU / Apple MLX). The pilot server has NVIDIA GPUs, so a model
that needs CUDA for training is still usable, but inference on the dev Mac matters for development.

---

## 1. Headline findings

1. **No open-source end-to-end "scan → closed IFC" tool exists.** Searches found research
   prototypes (FHNW, Osaka University, Cambeiro Barreiro et al. 2023) and commercial services, but
   no maintained, permissively licensed repo. Kelpod's pipeline (detect → fit to dimensions → graph →
   IFC) is not duplicated in open source.
2. **The best floor-plan models are blocked by licence, data or input type.**
   - DeepFloorPlan and its forks, Raster-to-Graph, FloorplanToBlender3d and ROBIN are **GPL-3.0**.
   - HEAT and SceneScript are **non-commercial**.
   - Raster2Seq's code is MIT but its **weights are CC BY-NC-4.0**.
   - CubiCasa5k is **CC BY-NC-4.0**, so any weights trained on it are commercially unsafe.
   - RoomFormer, PolyRoom and FRI-Net are built for **point-cloud density maps** (Structured3D /
     SceneCAD), not drawn plans. PolyRoom and FRI-Net also have **no licence** (all rights reserved).
3. **The permissive code worth using:** RoomFormer (MIT), FloorplanTransformation (MIT), Raster2Seq
   code (MIT), fpvec-lab (MIT), and generic detection and segmentation libraries (Detectron2,
   RF-DETR, D-FINE, SAM: Apache-2.0). In every case the **weights must be trained on our own data**.
4. **Commercially usable training data exists, but none of it is German.** The two best sources are
   **Swiss Dwellings** (CC BY 4.0, about 45k Swiss apartments as vector geometry; Swiss plans are
   close to DACH practice) and **ResPlan / ResPlan-FP** (CC BY 4.0, 17k South-Asian plans). Both are
   vector data, so they suit Kelpod's synthetic renderer: draw them in DIN 1356-1 style, then
   degrade. No open German *Bestandspläne* dataset was found, so the pilot's plans remain the key
   asset.
5. **Recent evidence backs our design.** fpvec-lab (Aug 2026, MIT) reports that on real scans,
   *detecting* junctions and centrelines and then assembling a graph beats autoregressive
   sequence decoding on every wall metric. Fusing the two raises wall F1 by about 7 points. Kelpod's
   "detect, then do the geometry deterministically" pipeline fits this result.
6. **Competitive risk: Hottgenroth already has AI floor-plan recognition.** "Hott-KI", described as
   "KI-basierte Grundrisserfassung", won the BAKA Award 2025 for product innovation. The award text
   says it reads scanned or image plans, was trained on "mehreren Tausend realen Vorhaben", and
   recognises interior and exterior walls, windows and doors to build a 3D model. The GebäudeScan
   app (Aug 2026) also lists "Import bestehender Grundrisse" via the HottKI CAD module. We could not
   verify how well it works, whether it reads dimension chains, or whether it ships in HottCAD
   itself. **This matters for the pitch and for the pilot interviews:** ask the pilot how Hott-KI
   performs on their plans. Sources:
   <https://bakaberlin.de/wp-content/uploads/2025/01/06-2463-24-Laudatio-Hottgenroth-P2.pdf>,
   <https://www.baulinks.de/webplugin/2026/1119.php4>.

---

## 2. Floor-plan recognition models and repos

| # | Project | What it does / I→O | Code licence | Weights / data licence | Maturity | No NVIDIA GPU? | Use |
|---|---|---|---|---|---|---|---|
| 1 | **RoomFormer** (CVPR 2023) <https://github.com/ywyue/RoomFormer> | Two-level DETR queries predict room polygons directly. Input: point-cloud **density map** (Structured3D, SceneCAD). Output: room polygons. | MIT | Checkpoints on polybox; trained on Structured3D (terms of use, believed non-commercial, **unverified**) → do not ship them | 346★, push 2025-04, research code (CUDA 11.1, torch 1.9) | Deformable-attention CUDA op; Deformable-DETR has a pure-PyTorch fallback, so CPU inference likely works with a code change (**unverified**) | **ideas / fine-tune**: room-polygon queries and corner relations; would need retraining on drawn plans |
| 2 | **FloorplanTransformation / Raster-to-Vector** (ICCV 2017) <https://github.com/art-programmer/FloorplanTransformation> | CNN predicts junction heatmaps and semantics; **integer programming** assembles walls, doors and rooms. Raster → vector lines + "popup" 3D file. | MIT | Images from LIFULL, **not redistributed** ("don't have the permission"); only vector annotations shared; weights' training-data rights unclear | 689★, push 2022-06, unmaintained; Torch7 + a PyTorch port | Inference on CPU plausible (plain CNN); original used CUDA 8 | **ideas**: junction detection + IP assembly is the closest published match to our geometry step. Re-implement on our data. |
| 3 | **Raster-to-Graph** (Eurographics 2024) <https://github.com/SizheHu/Raster-to-Graph> | Autoregressive transformer predicts the wall-junction graph with room types. Raster → graph. Includes a 10k-plan dataset. | **GPL-3.0** | Dataset licence **unverified** | 69★, push 2026-05 | Tested with CUDA 11.1 / Windows | **avoid** code; ideas from the paper only |
| 4 | **HEAT** (CVPR 2022) <https://github.com/woodfrog/heat> | Edge-attention transformer for planar graphs (floor plans from density maps, roofs). | **Non-commercial**; GPL-3 for research | Code, data and weights all non-commercial | 139★, push 2025-03 | CUDA ops | **avoid** |
| 5 | **PolyRoom** (ECCV 2024) <https://github.com/3dv-casia/PolyRoom> | Room-aware transformer; Mask2Former room init → polygons. Density maps (Structured3D / SceneCAD). | **No licence** (all rights reserved) | Checkpoints on Google Drive; Structured3D-trained | 110★, push 2025-05 | Needs MMDetection + CUDA | **ideas only** |
| 6 | **FRI-Net** (ECCV 2024) <https://github.com/Daisy-1227/FRI-Net> | Room-wise implicit representation; **point cloud → floor plan**. | **No licence** | Structured3D / SceneCAD checkpoints | 38★, push 2025-05 | CUDA assumed | **avoid** (wrong input, no licence) |
| 7 | **Raster2Seq** (SIGGRAPH 2026) <https://github.com/Cornell-VAILab/Raster2Seq>, weights <https://huggingface.co/haopt/Raster2Seq> | Floor plan image → polygon token sequence (anchor-guided autoregressive decoder). Has CubiCasa5k checkpoints. | MIT | **Weights CC BY-NC-4.0** (HF card) | 82★, push 2026-10, active | PyTorch; CPU inference probably slow but possible (**unverified**) | **fine-tune** (code only, retrain from scratch on permissive data) or ideas |
| 8 | **fpvec-lab** ("When should a network emit geometry…", arXiv 2608.25608) <https://github.com/Cyprinus12138/fpvec-lab> | Compares sequence and detection readouts; edit-cost metric; ResPlan-FP benchmark (17k plans). | MIT | ResPlan-FP CC BY 4.0; its corrected CubiCasa annotations stay non-commercial | 2★, push 2026-10, brand new | **unverified** | **ideas + benchmark**: adopt the edit-cost metric idea for evaluation; ResPlan-FP usable for pretraining |
| 9 | **CubiCasa5k model** <https://github.com/CubiCasa/CubiCasa5k> | Multi-task CNN (heatmaps + room/icon segmentation) and post-processing to polygons. | **CC BY-NC-4.0** (whole repo) | Weights and data non-commercial | 635★, push 2026-11 | PyTorch, CPU inference possible | **avoid** in product; useful only as a non-commercial research baseline |
| 10 | **DeepFloorPlan** <https://github.com/zlzeng/DeepFloorplan> and TF2 fork <https://github.com/zcemycl/TF2DeepFloorplan> | Room-boundary-guided attention; pixel masks for walls, doors and rooms. | **GPL-3.0** (both) | Trained on R2V / R3D | 420★ / 270★; last push 2024-02 / 2023-07 | TF, CPU possible | **avoid** code; the attention idea can be re-implemented clean-room from the paper |
| 11 | **FloorplanToBlender3d** <https://github.com/grebtsew/FloorplanToBlender3d> | Classical OpenCV wall detection → Blender 3D. | **GPL-3.0** | – | 647★, push 2024-10 | Yes | **avoid** |
| 12 | **plan2scene** <https://github.com/3dlg-hcvc/plan2scene> | Floor plan + photos → textured 3D house (texture synthesis, GNN). | MIT | Uses Rent3D++ (terms **unverified**) | 618★, push 2021-10, unmaintained | CUDA for rendering | **not relevant** (textures, not BIM) |
| 13 | **CADTransformer** <https://github.com/VITA-Group/CADTransformer> | Panoptic symbol spotting on **vector** CAD (FloorPlanCAD). | MIT | FloorPlanCAD reported CC BY-NC (**unverified**) | 138★, push 2023-07 | CUDA | **ideas** (only if inputs are vector PDFs) |
| 14 | **SceneScript** (Meta) <https://github.com/facebookresearch/scenescript> | Scene → structured language (walls, doors) from egocentric video and point clouds. | **CC BY-NC-4.0** | Non-commercial | 399★, push 2026-11 | – | **avoid** |

**Generic detectors and segmenters** (training targets for our own German data):

| Project | Licence (code / weights) | Maturity | Note |
|---|---|---|---|
| Detectron2 <https://github.com/facebookresearch/detectron2> | Apache-2.0 / Apache model zoo | 35k★, push 2026-10 | Mask R-CNN etc.; CPU inference works; mature |
| RF-DETR <https://github.com/roboflow/rf-detr> | Apache-2.0 | 10k★, push 2026-11, very active | Permissive alternative to Ultralytics YOLO; some larger variants may have a different licence (**unverified**, check per checkpoint) |
| D-FINE <https://github.com/Peterande/D-FINE> | Apache-2.0 | 3.4k★, push 2026-10 | Real-time DETR; permissive |
| MMSegmentation <https://github.com/open-mmlab/mmsegmentation> | Apache-2.0 | 10k★, **last push 2024-08** (stagnant) | Usable but OpenMMLab is quiet; prefer Detectron2 or plain PyTorch / segmentation-models |
| Segment Anything <https://github.com/facebookresearch/segment-anything> | Apache-2.0 (code + weights) | 55k★, push 2024-09 | Zero-shot masks for annotation help (CVAT / Label Studio), not wall semantics |
| Ultralytics YOLO | **AGPL-3.0** | – | **avoid** (already ruled out in the context doc) |

---

## 3. Datasets

| Dataset | Content | Licence | Commercial? | Use for Kelpod |
|---|---|---|---|---|
| **CubiCasa5k** <https://github.com/CubiCasa/CubiCasa5k> | 5k Finnish marketing plans; wall, room and icon polygons (SVG) | CC BY-NC-4.0 (LICENSE file) | **No** | **avoid** for training product models |
| **R2V** (FloorplanTransformation) | 870 annotated LIFULL (Japan) plans; images not distributed | Code MIT; images not shareable | **No / unclear** | avoid |
| **RPLAN** <http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/> (URL **unverified**) | ~80k Chinese apartments (raster) | Request form; non-commercial academic use as commonly reported (**unverified**) | Probably no | avoid |
| **Structured3D** <https://github.com/bertjiazheng/Structured3D> | Synthetic 3D houses, density maps, floor-plan annotations | Code MIT; data under "Structured3D Terms of Use" via agreement form (non-commercial as reported, **unverified**: terms PDF on Google Drive not read) | Probably no | avoid for product weights |
| **Swiss Dwellings** v3.0.0 <https://zenodo.org/records/7788422> | 45k apartments, ~3.1k buildings; 2D geometry of 1.7M separators (walls), 715k openings, 520k areas | **CC BY 4.0** (Zenodo) | **Yes**, with attribution | **adopt**: render in DIN 1356-1 style (hatching, Maßketten, room stamps) for synthetic pretraining; Swiss plans are close to DACH practice |
| **MSD (Modified Swiss Dwellings)** <https://doi.org/10.4121/e1d89cb5-6872-48fc-be63-aadd687ee6f9>, code <https://github.com/caspervanengelenburg/msd> | 4,167 train plans (graphs + structure arrays), derived from Swiss Dwellings | Data **CC BY 4.0** (4TU); code repo has **no licence** | Data yes | Data use; ML-ready subset of the above |
| **ResPlan** <https://github.com/m-agour/ResPlan> | 17k South-Asian residential plans; vector polygons + room graphs, metric | Data CC BY 4.0, code MIT (LICENSE); the authors grant rights only over their derived annotations, recovered from public listing renderings; has a takedown policy | Yes, with some provenance risk | Pretraining and synthetic rendering; conventions differ from German plans. "ModRes" (arXiv 2610.17178) reports errors and duplicates. |
| **ResPlan-FP** (in fpvec-lab) | Raster benchmark of 16,998 plans, frozen splits | CC BY 4.0 | Yes | Benchmark and pretraining |
| **Raster-to-Graph dataset** | 10k real residential plans (China) | **unverified** | unknown | Do not use until the licence is clear |
| **ROBIN** <https://github.com/gesstalt/ROBIN> | 510 plans, room and furniture symbols | Repo GPL-3.0 | Unclear for data | avoid |
| **FloorPlanCAD** | 15k vector CAD drawings, symbol spotting | Reported CC BY-NC-4.0 (**unverified**) | Probably no | avoid |
| **Versailles-FP, CVC-FP, R-FP500, multi-unit (Chile, 954 plans)** | Small wall-mask sets | Mixed / **unverified** | – | Evaluation inspiration only |
| **German / DIN plans** | **None found** | – | – | The pilot's *Bestandspläne* plus our synthetic generator are the only German source |

---

## 4. OCR for rotated dimension text

All of the engines below run on CPU.

| Project | Licence (code / weights) | Maturity | Rotated text | Notes / Use |
|---|---|---|---|---|
| **RapidOCR** <https://github.com/RapidAI/RapidOCR> | Apache-2.0 / PaddleOCR weights (Apache-2.0), converted to ONNX | 8.6k★, push 2026-11, active | Detector gives quadrilaterals; angle classifier handles 0/180°. For 90° vertical Maßketten, rotate crops ourselves. | **use** (already installed): fast ONNX on CPU, no Paddle dependency |
| **PaddleOCR** (PP-OCRv5) <https://github.com/PaddlePaddle/PaddleOCR> | Apache-2.0 / Apache-2.0 | 93k★, push 2026-09 | Same as above; orientation models | **use via RapidOCR**; use Paddle itself only to fine-tune the recogniser on German dimension strings (`3,51`, `BRH 87,5`) |
| **docTR** <https://github.com/mindee/doctr> | Apache-2.0 / Apache-2.0 (Mindee-hosted) | 6.4k★, push 2026-11 | `assume_straight_pages=False` handles rotated boxes | **use as a second engine / fallback**; PyTorch, easy to fine-tune |
| **EasyOCR** <https://github.com/JaidedAI/EasyOCR> | Apache-2.0 | 30k★, push 2026-11 | `rotation_info=[90,180,270]` tries rotations | Option for ensemble; slower |
| **Tesseract** <https://github.com/tesseract-ocr/tesseract> | Apache-2.0 (incl. `deu` traineddata) | 78k★, active | Weak on sparse, rotated text in drawings | Fallback for room stamps only |
| **MMOCR** <https://github.com/open-mmlab/mmocr> | Apache-2.0 | 4.7k★, push 2024-11 (stagnant) | Rotated detectors | avoid (stagnant) |
| **Surya** <https://github.com/datalab-to/surya> | Code Apache-2.0; **weights modified OpenRAIL-M** (free under $5M funding/revenue, paid above) | 21.6k★, push 2026-11 | VLM-based | Usable while small, but a licence trap when we scale; **avoid as core** |
| **TrOCR** (microsoft/unilm) | MIT | mature | Recognition on crops only | **ideas / fine-tune**: small recogniser for digit crops if RapidOCR mis-reads German decimals |

---

## 5. Vectorisation, line detection, constraint fitting

| Project | Licence | Maturity | Use |
|---|---|---|---|
| OpenCV (LSD line detector, Hough, morphology) | Apache-2.0 | mature | **use** (installed) |
| scikit-image (skeletonize, probabilistic Hough) | BSD-3 | mature | **use** (installed) |
| **skan** <https://github.com/jni/skan> | BSD-3 | 155★, push 2026-10 | **use**: skeleton → graph of branches and junctions (wall centrelines → nodes and edges) |
| **DeepLSD** <https://github.com/cvg/DeepLSD> | MIT | 658★, push 2026-09 | **fine-tune / try**: learned line detection robust to noise; weights' training data **unverified**; PyTorch, CPU possible |
| **HAWP** <https://github.com/cherubicXN/hawp> | MIT | 381★, push 2024-02 | ideas: wireframe (junction + line) parsing; weights trained on ShanghaiTech wireframe (licence **unverified**) |
| **vtracer** <https://github.com/visioncortex/vtracer> | MIT | 7.6k★, active | Raster → SVG curves; not semantic. Use only for debug overlays or symbol outlines. |
| Potrace | **GPL-2.0** | – | **avoid** |
| SolveSpace / python-solvespace <https://github.com/solvespace/solvespace> | **GPL-3.0** | 4.3k★ | **avoid** (good 2D geometric-constraint solver, wrong licence) |
| FreeCAD Sketcher solver (planegcs) | LGPL-2.1 (**unverified** for the standalone extraction) | mature | ideas, or an LGPL library if we ever need a full CAD constraint solver |
| SciPy `optimize.least_squares` | BSD | mature | **use**: dimension-constraint fitting (wall lengths = Maßkette values, orthogonality, shared nodes) |
| CVXPY / OR-Tools | Apache-2.0 | mature | **use if needed**: OR-Tools CP-SAT / MIP for discrete choices (which dimension goes with which wall, junction assembly à la FloorplanTransformation) |
| Shapely (`polygonize`, `snap`, `unary_union`), NetworkX | BSD | mature | **use** (installed): rooms from wall graph, closure checks |

---

## 6. IFC writers and validators

| Project | Licence | Maturity | Use |
|---|---|---|---|
| **IfcOpenShell** (`ifcopenshell.api`, `geom`, `validate`, IfcTester for IDS) <https://github.com/IfcOpenShell/IfcOpenShell> | **LGPL-3.0** | 3.0k★, push 2026-11, very active | **use** (installed): write IFC2X3 and IFC4 via `ifcopenshell.api` (walls, openings, spaces, storeys); `ifcopenshell.validate` for schema checks; IfcTester to check an **IDS** file encoding "what HottCAD needs". LGPL is fine as an unmodified dynamically linked library. **Bonsai** (Blender add-on in the same repo) is GPL: do not import it. |
| **buildingSMART Validation Service** <https://github.com/buildingSMART/validate>, rules <https://github.com/buildingSMART/ifc-gherkin-rules> | MIT / MIT | 74★ / 24★, push 2026-11, official | **use**: the rules run locally with pytest (no upload of customer data, which matters for GDPR); the web service <https://validate.buildingsmart.org> takes .ifc up to 256 MB; API is a preview, token by email to buildingSMART |
| **web-ifc** (That Open) <https://github.com/ThatOpen/engine_web-ifc> | MPL-2.0 | 1.1k★, active | **use later** in the Next.js frontend to preview IFC in the browser |
| xBIM (.NET) | CDDL (**unverified**) | mature | not needed (Python stack) |
| **ifc2plan** <https://github.com/datarefinerylab/ifc2plan> | MIT | 0★, push 2026-08 | **ideas / eval**: IFC → plan PNG per storey. Round-trip test: render our IFC back to a plan and compare to the input. |
| ezdxf <https://github.com/mozman/ezdxf> | MIT | 1.5k★, active | **use** for the DXF secondary output |
| Cloud2BIM (arXiv 2503.11498) | **unverified** | research | ideas: point cloud → IFC wall / opening writer structure |

---

## 7. Plan-to-BIM products and services (competitors / benchmarks)

All are closed source; claims come from vendor or press pages and were **not tested**.

| Product | Origin | What it claims | Source |
|---|---|---|---|
| **Hott-KI / HottKI** (Hottgenroth) | DE (Köln) | AI floor-plan capture from scanned or image plans, trained on "several thousand real projects"; walls (interior / exterior), windows and doors → 3D model; BAKA Award 2025; also in the GebäudeScan app (2026) | links in §1 |
| **Plans2BIM** (WiseBIM, listed by Graphisoft / Nemetschek) | FR | PDF / PNG / JPEG + scale → walls, windows, doors, spaces, slabs; IFC, DXF, CSV/XLSX; free and paid tiers; online | <https://www.graphisoft.com/en-az/partner-solutions/plans2bim/> |
| **Bimify** | SE (Stockholm, partly owned by Addnode) | PDF / drawings → RVT, IFC, ArchiCAD; rules such as standard wall thicknesses | <https://aecmag.com/bim/from-2d-to-3d-and-back/>, <https://www.symetri.co.uk/products/bimify/faqs/> |
| **Make a BIM** | FI | PDF → IFC in the cloud (Google Cloud, container option); its founder says the AI sometimes omits walls | <https://research.tuni.fi/cme-en/livecol/make-a-bim-turns-2d-drawings-into-3d-models-with-the-help-of-ai/> |
| **AmpliFy** (INEX BET, iFy suite) | FR | Free PDF → IFC / Revit (quality "may not be perfect") | <https://ovacen.com/en/convert-pdf-to-bim-revit-or-ifc/> |
| **CubiCasa** | FI / US | Real-estate floor plans from phone scans and plan conversion; pricing sales-led (aggregators list about 0–35 USD per plan, **unverified**) | <https://www.capterra.com/p/193054/CubiCasa/> |
| Autodesk ioFM | US | Accepts PDF floor plans and scanned hand-drawn plans for facility models (**unverified** detail) | search result only |
| Aurivus | DE (Ulm) | AI for digitising existing buildings; PDF input or IFC output **unverified** (appears point-cloud focused) | baunetzwissen.de |

Takeaway: generic PDF → IFC services exist (FR, SE, FI). None of them claim **German Maßketten as
constraints**, **closure guarantees** or **HottCAD-specific IFC attributes**. Hottgenroth's own
Hott-KI is the one to beat on HottCAD integration. Kelpod's differentiators should be the measurable
ones: dimension-fitted geometry, closed topology, an IDS-checked HottCAD import, and an on-prem /
GDPR deployment.

---

## 8. Ranked shortlist

### Adopt next (in order)

1. **IfcOpenShell `api` + `validate` + IfcTester/IDS** (LGPL lib). Write IFC2X3/IFC4; encode
   "HottCAD needs X" as an IDS file once the pilot tells us, so it becomes an automated test.
2. **buildingSMART ifc-gherkin-rules** (MIT), run locally in CI and evaluation. This gives the
   official validation without uploading customer data.
3. **RapidOCR** (Apache, ONNX, CPU), plus our own 90° crop rotation and German number parsing.
   Later, fine-tune the PP-OCR recogniser on synthetic Maßketten strings. Keep docTR as a second
   engine for low-confidence boxes.
4. **Swiss Dwellings / MSD + ResPlan(-FP)** (CC BY 4.0) as vector sources for the synthetic
   generator. Render them in DIN 1356-1 style with degradation to scale up training data beyond
   hand-drawn synthetic plans. Keep an attribution file.
5. **Classical vectorisation stack**: OpenCV LSD / morphology, skan (BSD), Shapely, NetworkX, and
   SciPy least-squares / OR-Tools for dimension fitting and junction assembly. This follows the
   FloorplanTransformation IP idea, re-implemented ourselves.
6. **A learned wall / opening model trained only on permissive data.** Options: (a) a
   junction + centreline heatmap model (the detection readout that fpvec-lab found best on real
   scans; FloorplanTransformation style); (b) instance detection for doors, windows and stairs with
   Detectron2, RF-DETR or D-FINE (Apache); (c) later, the Raster2Seq or RoomFormer architecture (MIT
   code) retrained from scratch on synthetic and pilot data. Train on the pilot's NVIDIA server; CPU
   inference on the Mac.
7. **Evaluation ideas**: the fpvec-lab **edit-cost metric** ("how much would the expert have to
   fix") fits Kelpod's promise that "the expert verifies, not redraws". The ifc2plan-style round
   trip (IFC → plan render → compare) is a second check.

### Avoid (with reasons)

- **DeepFloorPlan, TF2DeepFloorplan, Raster-to-Graph, FloorplanToBlender3d, ROBIN**: GPL-3.0.
- **HEAT, SceneScript**: non-commercial licences.
- **CubiCasa5k data, model and any weights trained on it** (including the Raster2Seq checkpoints,
  which are CC BY-NC) and **corrected CubiCasa annotations** (fpvec-lab): non-commercial.
- **RPLAN, Structured3D, FloorPlanCAD, R2V images**: non-commercial or unclear terms. That includes
  the published RoomFormer and PolyRoom checkpoints trained on them.
- **PolyRoom, FRI-Net, MSD code**: no licence (all rights reserved); PolyRoom and FRI-Net also use
  the wrong input type (point-cloud density maps).
- **Ultralytics YOLO, PyMuPDF**: AGPL-3.0. **Potrace, SolveSpace**: GPL. **Bonsai**: GPL.
- **Surya** as a core OCR: its weight licence becomes paid above $5M funding or revenue.
- **MMSegmentation / MMOCR** as core dependencies: no longer actively maintained (last push 2024).

---

## 9. Implications for the current MVP modules

| Module | Reuse | Build ourselves |
|---|---|---|
| **clean** | pypdfium2 (rasterise), OpenCV / scikit-image (deskew via Hough or projection profile, background normalisation, morphology for stains and folds) | Glue code, DPI and scale normalisation, fold-line removal tuned to our synthetic degradations. No learned model needed for the MVP. |
| **segment** | OpenCV morphology and hatching texture filters, skan for skeleton graphs; later Detectron2, RF-DETR or D-FINE (Apache) for doors, windows and stairs | The DIN 1356-1 rules (hatched vs. unhatched walls, door arcs, window triple lines) as a classical baseline. Later, a junction / centreline heatmap model trained on synthetic data (Swiss Dwellings / ResPlan rendered DIN-style) plus pilot plans. Ideas from FloorplanTransformation, RoomFormer and Raster2Seq; **no pretrained weights from the surveyed repos are commercially usable.** |
| **ocr** | RapidOCR (installed) for detection and recognition; docTR as an optional fallback | Rotate crops for vertical chains, a German number parser (`3,51`, `88,5`, `BRH 87,5`, `14,52 m²`), linking text to dimension lines and ticks. Later, fine-tune the recogniser on synthetic strings. |
| **geometry** | Shapely, NetworkX, SciPy `least_squares`; OR-Tools for discrete assignment if needed | Dimension-to-wall association, constraint fitting, angle snapping, node closing, consistency report. Nothing permissive exists for this; SolveSpace is GPL. |
| **export** | IfcOpenShell `api` (IFC2X3 + IFC4), `ifcopenshell.validate`, IfcTester / IDS; ezdxf for DXF | Mapping from `PlanGraph` / `BuildingModel` to IfcWall / IfcWindow / IfcDoor / IfcSpace / IfcBuildingStorey with heights; HottCAD-specific attributes once known (write them as an IDS file). |
| **evaluation** | buildingSMART ifc-gherkin-rules (local pytest); ideas from the fpvec-lab edit-cost metric and the ifc2plan round trip; Swiss Dwellings / ResPlan as extra ground-truth geometry for the synthetic generator | Our synthetic DIN-style renderer, the closure metrics in `ARCHITECTURE.md`, room IoU / wall F1, timing. |

---

## 10. Not verified / open items

- The Structured3D Terms of Use (Google Drive PDF), RPLAN terms and FloorPlanCAD licence were not
  read directly; "non-commercial" is as commonly reported.
- The Raster-to-Graph dataset licence, DeepLSD / HAWP weight provenance and per-checkpoint RF-DETR
  licences.
- Whether RoomFormer and Raster2Seq run CPU-only without code changes.
- Hott-KI quality, whether it reads dimension chains, and its availability inside HottCAD.
- Commercial prices of Plans2BIM, Bimify, Make a BIM and CubiCasa.
- Research prototypes with possibly unreleased code: FHNW (2D plans → IFC), Osaka University
  (Automation in Construction 181, 106584; synthetic dataset + annotation tool), Cambeiro Barreiro et
  al. 2023. Licences **unverified**.
