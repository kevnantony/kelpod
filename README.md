# Kelpod

Turns old German floor-plan scans (*Bestandspläne*: PDF, PNG, JPG, TIFF) into topologically closed
IFC 2x3 / IFC4 models that energy software such as HottCAD can import.

See [KELPOD_CONTEXT.md](KELPOD_CONTEXT.md) for the product and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the pipeline.

<!-- status:start (kept current by the lead agent; see CLAUDE.md) -->
## Status

_Last updated: 2026-11-24 19:10 CET. Numbers are synthetic unless marked "real"._

**Done**
- MVP promoted to `main` as **`baseline-v0`**: pipeline, API, evaluation protocol v1
  ([docs/MVP_REPORT.md](docs/MVP_REPORT.md)). Baseline on val-synth (200 synthetic plans):
  correct-closed 72 %.
- The locked evaluation can score learned models (declared checkpoints, checked by hash).
- Streamlit sandbox: the app now imports Kelpod from the repo it is deployed from.
  **Action for you:** in the Streamlit UI, switch the app to deploy from the `main` branch, once.
  After that, every push to `main` updates https://kelpod.streamlit.app.
- The OCR timing test no longer fails just because the shared cluster node is busy.

**Doing now**
- `exp/ocr-chainsum`: read dimension chains jointly (chain sums, pixel spans, n-best readings),
  including centimetre chains with raised millimetre digits (`178⁵`).
- `exp/learned-v1`: learned segmenter retrained on randomised drawing styles, 2 seeds, as the default.
- Collector for real plan images from the web, via official APIs (Openverse, Wikimedia Commons);
  first run on the cluster.

**Next**
- Real DIN plans from the web (`exp/din-web-plans`): double-line walls, upscaling of low-resolution
  images, cm/superscript OCR. Your two web plans currently give **0 rooms** each (real).
- Then the 3 historic plans (queued): none works yet (real).
<!-- status:end -->

```bash
uv sync --all-extras
uv run kelpod convert plan.pdf -o model.ifc
uv run pytest
```
