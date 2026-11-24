# Kelpod

Turns old German floor-plan scans (*Bestandspläne*: PDF, PNG, JPG, TIFF) into topologically closed
IFC 2x3 / IFC4 models that energy software such as HottCAD can import.

See [KELPOD_CONTEXT.md](KELPOD_CONTEXT.md) for the product and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the pipeline.

<!-- status:start (kept current by the lead agent; see CLAUDE.md) -->
## Status

_Last updated: 2026-11-24 22:45 CET. Numbers are synthetic unless marked "real"._

**Done**
- `main` = **`baseline-v0`** (pipeline, API, evaluation protocol). On val-synth (200 synthetic
  plans): correct-closed 72 %.
- Sandbox: the app imports Kelpod from the repo. **Action for you (once):** switch the Streamlit
  app to deploy from `main`.
- Learned segmenter v1 (2 training seeds, 3.5 GPU-h): correct-closed **98.5 %** on val-synth for
  both seeds (synthetic). It is not promoted, because it is about 65 % slower than the baseline on
  one CPU core (81 s vs 49 s per plan).
- Web-plan work (`exp/din-web-plans`), locked run 1 of 2: room IoU +0.05 on the held-out test style,
  but door precision, edit cost and exterior wall length got worse. **Not promoted.**
  - Your 2 real web plans: before, 0 rooms each. Now each comes out as one connected wall network
    with 5 rooms, but the rooms are still wrong (real).
- Plan collector (Openverse, Wikimedia Commons): code and tests done. Kept 0 images: Wikimedia
  answers every download from the cluster with HTTP 429, and Openverse returned 504.

**Doing now**
- `exp/learned-fast`: make the learned segmenter fast enough on CPU to pass the runtime rule.
- `exp/din-web-v2`: find and fix the door / envelope regressions on validation data only.
- `exp/historic-plans`: labels for your 3 historic plans, solid walls, scale bars, sheets with
  several plans. **None of the 3 works yet (real).**

**Question for you**
- The protocol blocks any change that makes a plan more than 25 % slower, even one far more accurate
  and still well under the 3-minute target. Keep that rule as it is, or allow, for example, "p95
  under 180 s" alone? I have not changed it.
<!-- status:end -->

```bash
uv sync --all-extras
uv run kelpod convert plan.pdf -o model.ifc
uv run pytest
```
