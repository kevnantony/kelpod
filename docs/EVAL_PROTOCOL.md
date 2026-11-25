# Evaluation protocol (v1, 2026-11-23)

This is how Kelpod decides that a change is better. `main` is always the current **baseline**.
A change is merged into `main` (**promoted**) only if it passes this protocol. Everything here is
designed so that it is easier to improve the real pipeline than to improve the score without
improving the pipeline.

This file, `scripts/promotion_check.sh`, `scripts/benchmark.py`, `src/kelpod/evaluation/` and
`eval/` are **protected**. See §6.

## 1. Branch workflow

1. Each experiment gets its own branch `exp/<short-name>`, cut from `main`, in its own worktree:
   `git worktree add ~/kelpod-wt/<name> -b exp/<name> origin/main`. Parallel experiments never
   share a working tree.
2. Develop and select using **dev** and **val** only (§2).
3. When a candidate looks better on val, run `scripts/promotion_check.sh exp/<name>`. It runs the
   integrity checks (§6) and then the locked evaluation (§3) for the candidate and for `main`, on the
   same machine and the same splits.
4. If it passes the promotion rule (§4), merge it into `main` with `--no-ff`, tag
   `baseline-vN`, and push. The merge commit message contains the comparison table.
5. If it does not pass, the branch stays unmerged. Record the result in `docs/EXPERIMENTS.md`, with
   the reason it failed.
6. Changes to evaluation itself go on `eval/<name>` branches that contain **no** pipeline or model
   change (§6).

## 2. Splits

Every split is a set of files with a SHA-256 manifest in `eval/manifests/<split>.json`. A split is
generated once, then frozen. Later changes to the generator, renderer or loaders do not change a
frozen split.

| Split | Content | Who may look at it |
|---|---|---|
| **dev** | Anything: synthetic plans from any seed, training-set samples | Freely, per item |
| **val** | Frozen: 200 synthetic plans (degrade 0 to 1.2, mixed layouts), 100 rendered Swiss Dwellings, 100 ResPlan-FP, 100 CubiCasa5k val. Once real plans exist: about 20 real dev pages. | Per item, for model selection and debugging |
| **test-synth** | Frozen: 300 synthetic plans in a **held-out style**. Fonts, hatching patterns, line weights and scan-damage profiles in this style are flagged test-only in the generator and must never be sampled for training. | Aggregates only, through `promotion_check.sh` |
| **test-ood** | Frozen: CubiCasa5k test split; Swiss Dwellings held out **by building ID**, so no apartment from a test building is in training; the frozen ResPlan-FP test split | Aggregates only |
| **test-real** | RealBench-DE: real German Bestandspläne, annotated by people, at least 40 pages. Stored outside the repo in `~/kelpod-data/sealed/` | Aggregates only. Never trained on, never inspected per item by an agent |
| **fresh** | 200 synthetic plans generated **at promotion time**, from seed `sha256(candidate commit + main commit)`, in the held-out style | Aggregates only. Nobody could have tuned on them in advance |

The "aggregates only" rule is a working rule, not a technical lock. The agent can technically read
these files. Every locked-split run is appended to `eval/TEST_LOG.md` (§5), so peeking shows up when
the log is audited.

**Leak hygiene:**
- Training manifests list every training file with its hash. `promotion_check.sh` fails if any
  test hash, or any Swiss Dwellings test building ID, appears in a training manifest.
- Near duplicates are checked with a perceptual hash against all test splits before a new training
  source is used.

## 3. Metrics (a panel, never a single number)

For each split, the table reports the following. "Correct" is defined by the headline metric below.

- **Headline: correct-closed rate.** The share of plans that meet all four conditions:
  - the model is topologically closed (no dangling walls, every room enclosed);
  - Hungarian-matched room IoU is at least 0.85 for every GT room;
  - the room count equals the GT room count;
  - total floor-area error is at most 3%.

  Closure alone does **not** count. An empty model or one big box is closed but scores 0.
- **Rooms:** mean IoU, precision, recall, area error.
- **Walls:** F1 at 10 cm (precision *and* recall), and the thickness-class accuracy (exterior vs.
  interior).
- **Openings:** door and window precision *and* recall; sill-height error where the GT has one.
- **OCR:** dimension-value precision and recall; the share of dimensions linked to the right wall.
- **Edit cost:** the number of node and edge edits needed to turn the output graph into the GT graph.
  This is the proxy for expert correction time.
- **Envelope error:** exterior wall length, window area per facade, and gross volume. These drive the
  energy calculation.
- **Robustness:** the crash rate, and how many plans return something other than a valid IFC that
  re-opens.
- **Runtime:** median and p95 seconds per page on the reference CPU node. p95 must stay under 180 s.

Real plans (test-real) are scored against human annotations. Expert correction time, measured with
the pilot, is added once available. It then outranks every automatic metric.

## 4. Promotion rule ("significant")

A candidate is promoted only if **all** of the following hold:

1. **Integrity checks pass** (§6). All tests pass, the test count is not lower than `main`'s, and no
   new `skip` or `xfail` has been added.
2. **Real gain on the primary split.** The primary split is test-real once it has at least 40 pages,
   until then test-ood plus fresh. On it, the 95% paired-bootstrap CI (10,000 resamples over plans)
   of Δ correct-closed rate has a lower bound > 0, **and** the point estimate is at least +3 pp. The
   alternatives to the +3 pp are: Δ edit cost of at least −10% with its CI excluding 0, or Δ room IoU
   of at least +0.02 with its CI excluding 0.
3. **No hidden regression.** On every split and every panel metric, the CI of Δ must not lie entirely
   below the tolerance:
   - −1 pp for rates;
   - −0.01 for IoU and F1;
   - +5% for edit cost and errors;
   - +25% for runtime, with p95 under 180 s.
4. **It generalises.** The gain is present (Δ > 0) on at least two of test-synth, test-ood and
   fresh, not only on one. A gain that appears only on synthetic splits is reported, but it is not
   enough for promotion once test-real exists.
5. **Seed stability.** Trained models are evaluated from at least 2 training seeds, and the rule
   must hold for the **worse** seed. The seeds are the checkpoints declared in the candidate's
   run config (`kelpod-run.toml`, §6); each is compared with the baseline's checkpoint of the same seed (the baseline is run
   once per seed too; unmatched seeds: every pair, the worst pair counts).
6. **Budget.** At most **2 locked runs per experiment branch**. Further tries need a new branch with a
   new idea, recorded in `docs/EXPERIMENTS.md`. This prevents trying until a lucky draw passes.

Small, safe changes (refactors, speed-ups, bug fixes with no metric change) may be merged when they
show no regression under rule 3 and pass rules 1 and 6. They are not counted as a new baseline
version.

## 5. Records (append-only)

- `eval/TEST_LOG.md`: one line per locked run. It records the date, branch, commit, splits, headline
  Δ with its CI, pass/fail, and the job ID. Lines are never edited or deleted. `promotion_check.sh`
  writes them.
- `docs/EXPERIMENTS.md`: hypothesis, change, data, GPU hours and their justification, val results,
  locked results, and the decision.
- Tags `baseline-vN` on `main`, each with the full comparison table in the tag message.

## 6. Integrity checks (anti reward hacking)

`scripts/promotion_check.sh` fails the candidate if:

- it changes any **protected path** (`src/kelpod/evaluation/`, `scripts/benchmark.py`,
  `scripts/promotion_check.sh`, `scripts/locked_eval.py`, `docs/EVAL_PROTOCOL.md`, `eval/`)
  compared with `main`. Evaluation
  changes go on `eval/*` branches with no model change. They must report both the old and new metric
  on the current baseline, and they are merged on their own, never in the same step as a model that
  benefits from them.
- pipeline code (`src/kelpod/` outside `evaluation/`) imports `kelpod.evaluation`, or mentions
  `ground_truth.json`, `manifests` or `sealed`.
- tests are removed, or `skip`/`xfail` markers are added.
- a frozen split's files do not match their manifest hashes.
- a training manifest overlaps a test split.

During locked runs, the benchmark copies **only the input file** (`plan.pdf`) into an empty temporary
directory and runs the pipeline there, so ground truth cannot be read. How each worktree's
pipeline is run (backends, interpreter project, one trained checkpoint per seed with its SHA-256)
comes only from its committed run config `kelpod-run.toml` (format: `eval/README.md`); absent
means the defaults. The config is not protected, but it is printed in the report, and the run
stops if a checkpoint's hash does not match, if it lies inside a worktree or an evaluation/sealed
data directory, or if its training manifests overlap a test split. Checkpoint hashes are logged. It rejects any result whose
scale came from `scale_source="ground_truth"`.

**Things the agent must never do:**
- tune thresholds on a locked split;
- add test-time special cases keyed on file names, seeds or image hashes;
- weaken a metric definition in the same branch as a model change;
- cherry-pick splits or seeds;
- report val numbers as test numbers.

If an agent finds a flaw in the protocol, it writes up the flaw and fixes it on an `eval/*` branch.
It never exploits it.

## 7. Keeping the benchmark honest over time

- After every 4 promotions, or once a month, add a new fresh OOD set, for example a new held-out
  rendering style or new real pages. Retire any split that has been looked at per item.
- When test-real grows, the newest pages are kept back as a "never seen" check for the next
  promotion.
- A person (the founders) reviews `eval/TEST_LOG.md`, the promotion commits and a few random outputs
  on real plans at every status update.
