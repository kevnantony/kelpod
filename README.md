# Kelpod

Turns old German floor-plan scans (*Bestandspläne*: PDF, PNG, JPG, TIFF) into topologically closed
IFC 2x3 / IFC4 models that energy software such as HottCAD can import.

See [KELPOD_CONTEXT.md](KELPOD_CONTEXT.md) for the product and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the pipeline.

<!-- status:start (kept current by the lead agent; see CLAUDE.md) -->
## Status

_Last updated: 2026-11-25 16:30 CET. Numbers are synthetic unless marked "real"._

**Done**
- **`baseline-v1` on `main`: the learned segmenter is the default.** On the held-out synthetic test
  splits, plans that come out closed and correct went from **30 % to 60 %** (worse of 2 seeds), and
  expert edits per plan dropped by 62-72 %.
- Sandbox: it uses the learned segmenter when its weights are available, otherwise the classical
  segmenter, with a note in the app.
- Historic plans (`exp/historic-v2`): your 3 plans, room IoU against agent-made labels (real):

  | Plan | `main` | branch |
  |---|---|---|
  | Tafel 71, plan 2 | 0.00 | 0.72 |
  | Tafel 71, plan 3 | 0.00 | 0.55 |
  | Villa Gundel | 0.00 | 0.25 |
  | Villa Garten/Vorhof | 0.17 | 0.41 |

  Scale error drops from 68-91 % to 2-5 % on Tafel and Gundel. **None of the 3 is fully right
  yet.** Tafel 71 plan 2 comes closest.

**Doing now**
- `exp/learned-v2`: the learned segmenter is being retrained on a wider set of drawing styles
  (2 seeds, on the cluster GPUs since about 15:30).
- Done today:
  - New out-of-distribution test sets (Swiss Dwellings, ResPlan, CubiCasa5k) are frozen and now
    count for promotion. The current model closes almost none of them: correct-closed 0 % on the
    validation sets, room IoU 0.14-0.62.
  - The evaluation now compares each training seed with the same baseline seed.
  - OCR speed-up: negative result.

**Questions for you**
- **How should real-plan improvements be promoted?** `exp/historic-v2` clearly helps your real
  plans and is neutral on synthetic plans. The protocol only rewards gains on test splits, and there
  is no real-plan test split yet; it needs at least 40 pages annotated by people. Options:
  - (a) you or the pilot annotate real pages, so test-real can exist;
  - (b) allow merging changes that pass "no regression" on synthetic plans and improve the real dev
    plans.

  Until you decide, the branch stays unmerged.
- **Conflict between the protocol and historic plans.** The held-out synthetic test style uses solid
  (poché) walls and cross-hatching, so training may never use them. Your historic plans are drawn
  exactly like that, so the learned segmenter is forbidden from learning the style it needs most.
  Proposal: retire that test style (EVAL_PROTOCOL §7 allows rotating held-out sets), define a new
  held-out style (e.g. other fonts and damage profiles), freeze a new test-synth, and allow solid
  walls in training. That is an eval change: I would do it on an `eval/*` branch only with your OK.
- May the model weights (98 MB) be published (GitHub release / Hugging Face, non-commercial note
  because of CubiCasa5k)? Then the sandbox gets the learned segmenter.
- The +25 % runtime rule: keep it, or rely only on "p95 under 180 s"?
- Switch the Streamlit app to deploy from `main` (once).
<!-- status:end -->

```bash
uv sync --all-extras
uv run kelpod convert plan.pdf -o model.ifc
uv run pytest
```
