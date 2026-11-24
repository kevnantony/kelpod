# Evaluation splits and locked runs

This directory is **protected** (`docs/EVAL_PROTOCOL.md` §6). It is changed only on `eval/*` branches.

- `manifests/<split>.json`: one manifest per split, frozen once it is built (SHA-256 of every file).
- `TEST_LOG.md`: one line per run of `scripts/locked_eval.py`. It is append-only: lines are never
  edited or deleted.
- Data are **not** in git. Frozen synthetic splits are in `~/kelpod-data/eval/<split>/`
  (`KELPOD_EVAL_DATA` overrides the root). Dataset and real splits stay where their manifest's
  `data_dir` points.

## Splits (status 2026-11-24)

| split | family | status | size | where | notes |
|---|---|---|---|---|---|
| `val-synth` | val | frozen | 200 plans | `~/kelpod-data/eval/val-synth/` | seeds 200000-200199, train style, degrade uniform 0 to 1.2 per plan |
| `test-synth` | test-synth | frozen | 300 plans | `~/kelpod-data/eval/test-synth/` | seeds 300000-300299, **held-out style**, degrade uniform 0 to 1.2 |
| `fresh` | fresh | generated per run | 200 plans | temp dir, deleted after the run | seeds from `sha256(candidate commit + main commit)`, held-out style, degrade 0 to 1.2 |
| `test-ood-swiss` | test-ood | **pending** | (site/building ID list written) | | held out by site; the ID lists are in the manifest so training can exclude them |
| `test-ood-cubicasa` | test-ood | **pending** | (official test list, 400) | | official CubiCasa5k `test.txt` |
| `test-ood-resplan` | test-ood | **pending** | (canonical test list, 1632) | | ResPlan `split.json` `test` |
| `val-swiss`, `val-resplan`, `val-cubicasa` | val | **pending** | (ID lists written) | | |
| `real-dev` | val (proxy) | **pending** | | `~/kelpod-data/real/dev/` | no data on disk yet; when there is, `pending` writes a hash manifest and runs use the proxy panel |
| `test-real` | test-real | **pending** | | `~/kelpod-data/sealed/real-test/` | no data on disk yet; hashes only, items are never opened one by one |

Why the dataset splits are pending: scoring them needs a renderer for vector plans that draws
DIN-style input pages **and** writes a `PlanGraph` ground truth (walls, rooms, openings). The
finetune agent's renderer (`training/kelpod_train/render.py` in the main checkout) is uncommitted
and writes pixel class maps for training, not graphs. A frozen split must come from committed
code. To build them: commit a renderer + graph converter (on an `eval/*` branch, or import a
committed one), add a `build` entry in `kelpod.evaluation.splits`, freeze, and record it here.

`locked_eval.py` skips a pending or missing split with a note in the table and the log line. Until
test-ood exists, the primary split of rule 2 is **fresh** alone (protocol: test-ood + fresh), and
rule 4 can only count test-synth and fresh.

## Reserved seeds (synthetic generator)

| range | use |
|---|---|
| 0-9999 | dev benchmark (`scripts/benchmark.py`, the user's baseline job uses 1000-1199) |
| 100000-100049 | finetune agent's own synthetic test seeds |
| 200000-200199 | `val-synth` |
| 300000-300299 | `test-synth` |
| 1,000,000 to < 2^40 | training |
| >= 2^40 | `fresh` splits |

The test style does not depend on seeds: `generate(seed, style="heldout")` is the only way to get
held-out fonts, hatching, line weights and scan damage. Training must use the default
`style="train"`.

## Manifest format (`manifest_version` 1)

```json
{
  "manifest_version": 1,
  "split": "test-synth",
  "status": "frozen | pending | sealed-no-annotations",
  "kind": "synth | dataset | proxy | sealed",
  "family": "val | test-synth | test-ood | test-real | fresh",
  "created_utc": "2026-11-24 10:00",
  "data_dir": "~/kelpod-data/eval/test-synth",
  "input_file": "plan.pdf",
  "gt_file": "ground_truth.json",
  "generator": {"module": "kelpod.evaluation.synth", "commit": "<sha>", "version": "1",
                "style": "heldout", "dpi": 300.0},
  "params": {"seeds": "300000-300299", "degrade": "..."},
  "heldout_ids": {"swiss_site_id": [], "swiss_building_id": [], "cubicasa_id": [],
                  "resplan_id": []},
  "n_items": 300,
  "items": [{"id": "s300000", "seed": 300000, "degrade": 0.41, "style": {"font": "..."},
             "n_rooms": 5, "phash": "<16 hex>",
             "files": {"plan.pdf": "<sha256>", "ground_truth.json": "<sha256>", "...": "..."}}]
}
```

Hash-only manifests (real-dev, test-real) list `{"id": "<relative path>", "sha256": ...}` per file.

## Training manifests (what the overlap check reads)

Every training source writes `training/manifests/<source>.json` (or a path passed with
`--training-manifests`):

```json
{"source": "synth-train", "files": [
  {"path": "synth_1000000_img.png", "sha256": "...", "phash": "<optional, splits.phash>",
   "seed": 1000000, "style": "train", "site_id": 123, "cubicasa_id": "high_quality/1234",
   "resplan_id": 17}
]}
```

`scripts/locked_eval.py --verify-manifests` fails if any entry has the hash of a test file, a
held-out Swiss site/building ID, a held-out CubiCasa5k / ResPlan ID, the held-out style, or a pHash
within 4 bits of a test image. The pHash helper is `kelpod.evaluation.splits.phash` /
`near_duplicates`.

## Running

```bash
uv run python scripts/locked_eval.py --verify-manifests          # hashes + training overlap
uv run python scripts/locked_eval.py --candidate WT --baseline WB --branch exp/x \
    --log eval/TEST_LOG.md                                       # what promotion_check.sh runs
uv run python scripts/locked_eval.py --candidate . --baseline . --branch eval/x \
    --splits val-synth --smoke                                   # smoke / val, not a locked run
python -m kelpod.evaluation.splits build <split>                 # freeze a synthetic split (once)
python -m kelpod.evaluation.splits pending                       # (re)write pending manifests
```

Results go to `~/kelpod-runs/locked/<UTC time>_<branch>/` (`comparison.md`, `results.json`).
Runs whose kind is `smoke` or `val` are logged with `(<kind>)` after the branch name, so they do
not use up the 2-run budget (rule 6).

## Implementation choices and known limits (v1)

- **Runtime reference.** Each pipeline run gets one dedicated physical core (both hyperthreads),
  8 runs in parallel on a 16-CPU allocation of a pascal node. Candidate and baseline runs of the
  same plan are interleaved, so the paired Δ is fair, but the absolute numbers depend on the node's
  load (pascal-node02 is shared): about 24 s median per plan under full load, against about 6 s
  for one plan alone with all cores. The p95 < 180 s check uses these loaded numbers.
- **Runtime noise.** In the smoke run with identical code on both sides, the runtime median Δ was
  +4.2% [-2.1%, +10.8%] (val-synth, 200 plans). That is inside the +25% tolerance, but runtime is the
  noisiest metric.
- **Headline.** "Closed" = `closure_metrics(pred).is_closed`. Room IoU >= 0.85 for every GT room
  after Hungarian matching, equal room count, total floor-area error <= 3%.
- **Errors** are relative and capped at 1.0 per plan. A crashed or rejected plan scores 0 on rates,
  IoU and F1, 1.0 on errors, and the edit cost of building the GT from nothing.
- **Thickness class.** Exterior = `WallKind.EXTERIOR`, or `UNKNOWN` with thickness >= 0.30 m.
  Everything else is interior. Accuracy is length-weighted over GT centreline that a parallel
  predicted wall covers within 10 cm.
- **Edit cost** = node, wall-edge and opening insertions and deletions, plus opening kind changes,
  after 1:1 node matching within 20 cm. Moving a matched node costs nothing, so this is a lower
  bound on correction effort.
- **Envelope.** Exterior wall centreline length; window area (width x height, a missing height
  counts as 1 m) per N/E/S/W facade, summed as absolute error over GT total; gross volume as gross
  footprint (walls + rooms, holes filled) x storey height, which is the footprint error for one
  storey.
- **IFC.** Re-opened by the evaluator with ifcopenshell (at least one `IfcWall` and one
  `IfcSpace`), independent of the pipeline's own `reopened_ok`.
- **Not measured yet:** the OCR panel (dimension-value P/R, dimension-to-wall linking).
  `PipelineResult` exposes no OCR annotations, and the runner may only call the public
  `kelpod.pipeline.convert`.
- **Primary split.** Until test-ood is frozen, the primary split is fresh alone, and rule 4 needs
  a gain on both test-synth and fresh.
