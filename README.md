# Kelpod

Turns old German floor-plan scans (*Bestandspläne*: PDF, PNG, JPG, TIFF) into topologically closed
IFC 2x3 / IFC4 models that energy software such as HottCAD can import.

See [KELPOD_CONTEXT.md](KELPOD_CONTEXT.md) for the product and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the pipeline.

<!-- status:start (kept current by the lead agent; see CLAUDE.md) -->
## Status

_Last updated: 2026-11-25 17:35 CET. **Kelpod is PAUSED by Kevin (25 Nov) to free the agent and
GPUs for another project.** Resume: `scripts/cockroach.sh start`. Numbers are synthetic unless
marked "real"._

**Done**
- `main` = **`baseline-v1`**: the learned segmenter is the default. On the held-out synthetic test
  splits, plans that come out closed and correct went from 30 % to 60 % (worse of 2 seeds); expert
  edits per plan dropped by 62-72 %.
- Evaluation:
  - out-of-distribution test sets (Swiss Dwellings, ResPlan, CubiCasa5k) frozen and used for
    promotion;
  - each training seed is compared with the same baseline seed;
  - learned checkpoints are checked by hash.
- Historic plans (`exp/historic-v2`, unmerged), real, agent-made labels, room IoU: Tafel 71
  0.00 -> 0.72 / 0.55, Gundel 0.00 -> 0.25, villa 0.17 -> 0.41. None of the 3 is fully right yet.

**Doing now**
- Nothing (paused).
- In flight at the pause: `exp/learned-v2`, the learned segmenter retrained on wider drawing styles
  (2 seeds, 6.5 GPU-h). No evaluation numbers yet.
  - Slurm jobs 129106 (validation comparison) and 129107 (dev proxies) were left queued; their
    results land in `~/kelpod-wt/learned-v2/training/logs/`.

**Next (on resume)**
- Read the learned-v2 results. If they are better, run the locked promotion gate (~4.5 h).
- Answer the questions below. They decide what can be merged next.

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
