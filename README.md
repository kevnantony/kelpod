# Kelpod

Turns old German floor-plan scans (*Bestandspläne*: PDF, PNG, JPG, TIFF) into topologically closed
IFC 2x3 / IFC4 models that energy software such as HottCAD can import.

See [KELPOD_CONTEXT.md](KELPOD_CONTEXT.md) for the product and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the pipeline.

<!-- status:start (kept current by the lead agent; see CLAUDE.md) -->
## Status

_Last updated: 2026-11-24 13:20 CET (priorities reset by the user; agent paused for a usage
limit until 17:20)._ Numbers are synthetic unless marked "real"._

**Done**
- MVP promoted to `main` as **`baseline-v0`**: pipeline, API, evaluation protocol v1
  ([docs/MVP_REPORT.md](docs/MVP_REPORT.md)). Baseline on the frozen val-synth split (200 synthetic
  plans): correct-closed 72 %.
- The locked evaluation can now score learned models (declared checkpoints, checked by hash, 2
  training seeds). Merged into `main`.
- Learned segmentation (fine-tuned U-Net) is available as an option. On val-synth: correct-closed
  98 % vs. 72 % (synthetic).

**Doing now**
- Resume `exp/ocr-chainsum` (joint reading of dimension chains) and the other work that was in
  progress before the historic-plans detour.

**Next: real historic plans (queued by the user, not urgent)**
- Your 3 historic plans, real, measured 2026-11-24: **none works yet**, with either segmenter.
  - Villa Garten/Vorhof: classical gives 9 rooms but 19 loose wall ends.
  - Villa Gundel: 1 room.
  - Bauernhäuser Tafel 71 (7 plans on one sheet): 2 rooms.
  - The learned model does no better on any of them, and the scale is always guessed.
- `exp/historic-plans`: hand labels for the 3 plans, scale from scale bars and dimensions written
  inside rooms, splitting sheets with several plans, solid black walls, exterior spaces.
- `exp/learned-v1`: retrain the learned segmenter with a historic drawing style (solid walls, low
  resolution, scale bars, clutter) plus randomised styles, 2 seeds.
- Collecting more openly licensed historic German plans for a real dev set.

**After that**
- Report per plan which of the 3 historic plans work after each change; then promote what passes
  the protocol.
<!-- status:end -->

```bash
uv sync --all-extras
uv run kelpod convert plan.pdf -o model.ifc
uv run pytest
```
