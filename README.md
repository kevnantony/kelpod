# Kelpod

Turns old German floor-plan scans (*Bestandspläne*: PDF, PNG, JPG, TIFF) into topologically closed
IFC 2x3 / IFC4 models that energy software such as HottCAD can import.

See [KELPOD_CONTEXT.md](KELPOD_CONTEXT.md) for the product and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the pipeline.

<!-- status:start (kept current by the lead agent; see CLAUDE.md) -->
## Status

_Last updated: 2026-11-25 CET (morning). Numbers are synthetic unless marked "real"._

**Done**
- `main` = **`baseline-v0`**. On val-synth (200 synthetic plans): correct-closed 72 %.
- Web-plan fixes (`exp/din-web-v2`): two locked runs, both not promoted.
  - Run 1 failed on door recall.
  - Run 2, after a fix, has no regressions, but its gain is only +1.5 pp correct-closed
    [-1.0, +4.0]; the bar is +3 pp.
  - The branch is closed; its parts can return in a later experiment.
- Historic plans (`exp/historic-plans`), real, 3 plans with agent-made labels:
  - The scale is now right (error 59 % -> 1-2 %), and the sheet with several plans is split into
    separate plans.
  - With the learned segmenter the walls are mostly found (wall F1 0.70-0.97 on 3 of 4 plans), but
    rooms stay open at the door gaps (room IoU 0.12-0.50). **None of the 3 plans works yet.**
- Plan collector merged; it cannot download from the cluster (Wikimedia HTTP 429).

**Doing now**
- `exp/learned-fast`: the learned segmenter, now about 3x faster (whole page in one pass, at 0.65x
  scale). Smoke run on 40 plans: correct-closed 95 % vs 90 %, runtime +17-20 % (limit +25 %).
  The full 200-plan check is running as a Slurm job; then the locked run.

**Next**
- If learned-fast is promoted, close door gaps in solid walls for the historic plans on top of it.

**Question for you** (still open)
- The +25 % runtime rule: keep it, or rely only on "p95 under 180 s"?
- Action for you (once): switch the Streamlit app to deploy from `main`.
<!-- status:end -->

```bash
uv sync --all-extras
uv run kelpod convert plan.pdf -o model.ifc
uv run pytest
```
