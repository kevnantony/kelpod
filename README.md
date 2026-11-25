# Kelpod

Turns old German floor-plan scans (*Bestandspläne*: PDF, PNG, JPG, TIFF) into topologically closed
IFC 2x3 / IFC4 models that energy software such as HottCAD can import.

See [KELPOD_CONTEXT.md](KELPOD_CONTEXT.md) for the product and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the pipeline.

<!-- status:start (kept current by the lead agent; see CLAUDE.md) -->
## Status

_Last updated: 2026-11-25 CET (midday). Numbers are synthetic unless marked "real"._

**Done**
- **`baseline-v1` on `main`: the learned segmenter is now the default.** It passed the locked
  promotion run on the held-out synthetic test splits:
  - plans that come out closed and correct: **30 % -> 60 %** (fresh split, worse of 2 training
    seeds);
  - expert edits per plan: -62 to -72 %;
  - runtime: +18 to +28 %.
  - On val-synth: 72 % -> 92-95 %.
- The model weights are not in git. The sandbox uses the classical segmenter until they are
  published (see the question below).
- Web-plan fixes (`exp/din-web-v2`): closed after 2 locked runs; the gain was too small
  (+1.5 pp).
- Historic plans: the scale is now right and sheets with several plans are split (branch, real).
  **None of your 3 historic plans works yet.**

**Doing now**
- Historic plans on top of `baseline-v1`: close door gaps in solid walls, plus scale bars and
  sheet splitting.

**Questions for you**
- May the model weights (98 MB ONNX) be published, e.g. as a GitHub release asset or on Hugging
  Face, so the sandbox and other users get the learned segmenter? Its training data includes
  CubiCasa5k (CC BY-NC), so the weights should carry a non-commercial note.
- The +25 % runtime rule: keep it, or rely only on "p95 under 180 s"?
- Switch the Streamlit app to deploy from `main` (once).
<!-- status:end -->

```bash
uv sync --all-extras
uv run kelpod convert plan.pdf -o model.ifc
uv run pytest
```
