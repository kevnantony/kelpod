# Kelpod

Turns old German floor-plan scans (*Bestandspläne*: PDF, PNG, JPG, TIFF) into topologically closed
IFC 2x3 / IFC4 models that energy software such as HottCAD can import.

See [KELPOD_CONTEXT.md](KELPOD_CONTEXT.md) for the product and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the pipeline.

<!-- status:start (kept current by the lead agent; see CLAUDE.md) -->
## Status

_Last updated: 2026-11-24 12:55 CET. All numbers are from synthetic or re-rendered plans; nothing has been tested on real scans yet._

**Done**
- MVP finished and promoted to `main` as **`baseline-v0`**. It contains the pipeline (clean,
  classical segmentation, OCR, geometry, IFC4 / IFC2X3 export), the API (`kelpod serve`) and
  evaluation protocol v1. Report: [docs/MVP_REPORT.md](docs/MVP_REPORT.md).
- Baseline on the frozen val-synth split (200 synthetic plans): correct-closed 72 %, room IoU 0.94.
- Learned segmentation (fine-tuned U-Net, 1 GPU hour) is available as an option. On val-synth it
  reaches correct-closed 98 % (vs. 72 %) and edit cost 0.45 (vs. 4.1) edits per plan. These are
  synthetic dev numbers; it is not the default yet.

**Doing now** (one branch and worktree each)
- `exp/learned-v1`: retrain the learned segmenter with drawing-style randomisation, 2 training
  seeds, and make it the default.
- `exp/ocr-chainsum`: read dimension chains jointly (chain sums, pixel spans, n-best).
- `eval/learned-runner`: let the locked evaluation score learned models (declared checkpoints,
  checked by hash).
- Collecting openly licensed real floor-plan scans for a real dev set and a sealed test set.

**Next**
- Promote whatever passes `scripts/promotion_check.sh`, then pick the next bet from
  [docs/research/V2_RESEARCH.md](docs/research/V2_RESEARCH.md).
<!-- status:end -->

```bash
uv sync --all-extras
uv run kelpod convert plan.pdf -o model.ifc
uv run pytest
```
