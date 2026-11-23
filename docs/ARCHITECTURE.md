# Kelpod MVP architecture

Status: first milestone (plan to IFC on 3 sample plans). Everything here works on **synthetic** plans
only until the pilot provides real *Bestandspläne*.

## Pipeline

| Step | Module | Input → output (see `src/kelpod/contracts.py`) |
|---|---|---|
| 1 Clean | `kelpod.clean.run(path)` | file → `list[RasterPage]` |
| 2 Segment | `kelpod.segment.run(page)` | `RasterPage` → `Segmentation` (px) |
| 3 OCR | `kelpod.ocr.run(page)` | `RasterPage` → `OcrAnnotations` (px + parsed values) |
| 4–5 Fit + check | `kelpod.geometry.run(page, seg, ocr)` | → `PlanGraph` (m) with `ConsistencyReport` |
| 6 Export | `kelpod.export.build_model(graph)`, `write_ifc(model, path)` | → `BuildingModel` → `ExportResult` |
| Eval | `kelpod.evaluation` | synthetic plans + ground truth, closure metric, timing benchmark |

`kelpod.pipeline.convert` chains the steps and times each one; `kelpod convert` is the CLI.

Rules for every module: it owns only its folder in `src/kelpod/<module>/` and `tests/<module>/`,
and it talks to other modules only through `contracts.py`.

Licences: This public repo is Kelpod's open-source, educational version. Any open-source code, model or dataset may be used here, including GPL/AGPL code and non-commercial datasets such as CubiCasa5k, as long as we follow its terms and credit it in `CREDITS.md`. The paid version is built separately; licence checks for it happen there, not in this repo.

## Drawing conventions of the synthetic plans

The generator (`kelpod.evaluation.synth`) and the segmentation/OCR baselines agree on these
conventions, which follow common German practice (DIN 1356-1 style). Real plans vary; this is a
starting point, not a claim about all *Bestandspläne*.

- **Scale and resolution:** drawn at 1:100, rendered at 300 dpi, then degraded; the pipeline works at
  200 dpi by default. At 1:100 and 200 dpi, 1 px ≈ 1.27 cm.
- **Exterior walls:** 36.5 cm (also 30 / 24 cm), solid outline with diagonal 45° hatching
  (masonry).
- **Interior load-bearing walls:** 24 or 17.5 cm, outline with hatching.
- **Light partitions:** 11.5 cm or 10 cm, outline only, or outline with sparse hatching.
- **Doors:** a gap in the wall, a straight door leaf line from the hinge, and a quarter-circle swing
  arc (thin line). Typical clear widths 0.76 / 0.885 / 1.01 m.
- **Windows:** a gap in an exterior wall with two or three thin parallel lines along the wall (frame
  and glass). Typical widths 0.885–2.01 m.
- **Dimension chains (Maßketten):** outside the exterior walls, parallel to them, thin line with
  short 45° tick strokes at each measured point and extension lines towards the wall. The value sits
  centred above the line; vertical chains are rotated 90° (read bottom-to-top).
- **Number formats:** metres with a German decimal comma, e.g. `3,51`, `11,24`, `0,885`. Values
  under 1 m may also be centimetres, e.g. `88,5` or `76`. Sill heights as `BRH 0,90` or `BRH 87,5`.
- **Room stamps:** centred in the room, name on the first line (e.g. `Wohnen`, `Küche`, `Bad`, `Flur`,
  `Schlafen`, `Kind`, `WC`, `Abst.`), area on the second (`14,52 m²`).
- **Degradation:** small rotation (±3°), Gaussian noise, blur, JPEG artefacts, stains, fold lines,
  uneven background.

## Closure metric (honest version)

Reported per plan by `kelpod.evaluation`:

- **node closure:** share of graph nodes with degree ≥ 2 (no loose wall ends);
- **room closure:** share of rooms that are valid closed polygons;
- **topologically closed:** one connected component, no dangling nodes, all rooms closed;
- **room IoU / wall F1 against ground truth** (synthetic plans only): how *correct* the closed
  model is, not just whether it is closed;
- **dimension residual:** RMS difference between fitted wall lengths and OCR dimensions;
- **runtime:** seconds per step and in total (target: under 3 minutes per plan).
