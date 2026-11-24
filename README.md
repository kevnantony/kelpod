# Kelpod

Turns old German floor-plan scans (*Bestandspläne*: PDF, PNG, JPG, TIFF) into topologically closed
IFC 2x3 / IFC4 models that energy software such as HottCAD can import.

See [KELPOD_CONTEXT.md](KELPOD_CONTEXT.md) for the product and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the pipeline.

<!-- status:start (kept current by the lead agent; see CLAUDE.md) -->
## Status

_Last updated: 2026-11-24 02:30 CET. All numbers are from synthetic plans; nothing has been tested on real scans yet._

**Done**
- Pipeline steps built and tested: clean, classical segmentation, OCR (RapidOCR), geometry
  (fit to dimensions, closed graph), IFC4 / IFC2X3 export; `kelpod convert` writes IFC that re-opens.
- Closure fixes in segmentation. Held-out synthetic seeds 100-129, before -> after: degrade 0.5
  fully closed 73 % -> 80 %, room IoU 0.94 -> 0.96; degrade 1.0 closed 7 % -> 53 %, room IoU
  0.77 -> 0.87 (synthetic, dev numbers).
- FastAPI endpoint (`kelpod serve`): `/health`, `/convert`, `/jobs/{id}`. No auth yet.

**Doing now**
- Paused for a usage limit until about 02:20 CET; the loop resumes the work.
- Fine-tuning a learned segmentation model (U-Net, CubiCasa5k zero-shot baseline); training data
  built, no training results yet.
- Evaluation protocol v1 (frozen splits, locked evaluation) on branch `eval/protocol-v1`.

**Next**
- Compare the learned model with the classical baseline; write `docs/MVP_REPORT.md`.
- Merge MVP + eval protocol into `main` as `baseline-v0`; then experiments on `exp/*` branches,
  scored on real dev scans as well.
<!-- status:end -->

```bash
uv sync --all-extras
uv run kelpod convert plan.pdf -o model.ifc
uv run pytest
```
