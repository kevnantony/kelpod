# Kelpod

Turns old German floor-plan scans (*Bestandspläne*: PDF, PNG, JPG, TIFF) into topologically closed
IFC 2x3 / IFC4 models that energy software such as HottCAD can import.

See [KELPOD_CONTEXT.md](KELPOD_CONTEXT.md) for the product and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the pipeline.

<!-- status:start (kept current by the lead agent; see CLAUDE.md) -->
## Status

_Last updated: 2026-11-23 22:05 CET. All numbers are from synthetic plans; nothing has been tested on real scans yet._

**Done**
- Pipeline steps built and tested: clean (deskew, folds, stains), classical segmentation (walls,
  doors, windows, rooms), OCR (RapidOCR; dimension chains, BRH, room stamps), geometry (fit to
  dimensions, closed graph), IFC4 / IFC2X3 export.
- End to end: `kelpod convert` writes IFC that re-opens. 10-plan synthetic benchmark (degrade 0.5):
  8/10 fully closed, room IoU 0.92, wall F1 0.997, ~4.7 s per plan
  ([report](docs/results/benchmark_synth10_degrade05.md)).
- FastAPI endpoint (`kelpod serve`): `/health`, `/convert`, `/jobs/{id}`. No auth yet.
- Open-source survey ([docs/research/OPEN_SOURCE_SURVEY.md](docs/research/OPEN_SOURCE_SURVEY.md)).

**Doing now**
- Fixing the closure failures on synthetic seeds 2 and 9 (segmentation).
- Setting up fine-tuning of a learned segmentation model on the cluster GPUs (CubiCasa5k and Swiss
  Dwellings downloaded).
- Researching what would make the next version much better on real scans.

**Next**
- Compare the learned model with the classical baseline on a fixed test split.
- Write `docs/MVP_REPORT.md` (what works, what does not, what needs real plans).
- Then: next-version work picked from the research, and first real *Bestandspläne* from the pilot.
<!-- status:end -->

```bash
uv sync --all-extras
uv run kelpod convert plan.pdf -o model.ifc
uv run pytest
```
