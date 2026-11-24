# Kelpod

Turns old German floor-plan scans (*Bestandspläne*: PDF, PNG, JPG, TIFF) into topologically closed
IFC 2x3 / IFC4 models that energy software such as HottCAD can import.

See [KELPOD_CONTEXT.md](KELPOD_CONTEXT.md) for the product and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the pipeline.

<!-- status:start (kept current by the lead agent; see CLAUDE.md) -->
## Status

_Last updated: 2026-11-24 07:30 CET. All numbers are from synthetic plans; nothing has been tested on real scans yet._

**Done**
- Pipeline (clean, classical segmentation, OCR, geometry, IFC4 / IFC2X3 export); `kelpod convert`
  writes IFC that re-opens. FastAPI endpoint (`kelpod serve`).
- Closure fixes. Held-out synthetic seeds, before -> after: degrade 0.5 fully closed 73 % -> 80 %;
  degrade 1.0 7 % -> 53 % (synthetic, dev numbers).
- Evaluation protocol v1: frozen val-synth (200) and test-synth (300, held-out drawing style),
  `scripts/locked_eval.py`. Current baseline on val-synth: correct-closed 52.5 %, room IoU 0.90
  (synthetic smoke run).
- First fine-tuning run of a U-Net segmenter finished (1.0 GPU hour on the cluster).
- MVP report draft: [docs/MVP_REPORT.md](docs/MVP_REPORT.md).

**Doing now**
- Evaluating the fine-tuned segmenter against the classical one and the zero-shot CubiCasa5k model
  (zero-shot is worse: closed 20 % vs. 88 % at degrade 0.5, synthetic).
- Collecting openly licensed real floor-plan scans for a real dev / sealed test set.

**Next**
- Finish `docs/MVP_REPORT.md`; merge MVP + eval protocol into `main` as `baseline-v0`.
- Then experiments on `exp/*` branches, each promoted only through `scripts/promotion_check.sh`.
<!-- status:end -->

```bash
uv sync --all-extras
uv run kelpod convert plan.pdf -o model.ifc
uv run pytest
```
