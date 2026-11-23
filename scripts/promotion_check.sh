#!/usr/bin/env bash
# Integrity gate for promoting a branch into main (docs/EVAL_PROTOCOL.md §4, §6).
#   scripts/promotion_check.sh exp/<name>            integrity checks, then locked evaluation
#   scripts/promotion_check.sh exp/<name> --no-eval  integrity checks only
# The locked evaluation is scripts/locked_eval.py (built on an eval/* branch; see the protocol).
# Exit 0 = may be promoted; anything else = do not merge.
set -u
CAND=${1:?usage: promotion_check.sh <branch> [--no-eval]}
MODE=${2:-}
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"
fail=0; bad() { echo "FAIL: $*"; fail=1; }
git fetch -q origin
BASE=$(git rev-parse origin/main); HEADC=$(git rev-parse "$CAND") || exit 2
MB=$(git merge-base "$BASE" "$HEADC")

PROTECTED="src/kelpod/evaluation scripts/benchmark.py scripts/promotion_check.sh scripts/locked_eval.py docs/EVAL_PROTOCOL.md eval"
case "$CAND" in
  eval/*) echo "eval branch: protected paths may change; it must not touch pipeline code"
          git diff --name-only "$MB" "$HEADC" -- src/kelpod | grep -v '^src/kelpod/evaluation/' | grep -q . \
            && bad "eval branch changes pipeline code" ;;
  *)      changed=$(git diff --name-only "$MB" "$HEADC" -- $PROTECTED)
          [ -n "$changed" ] && bad "protected eval paths changed (use an eval/* branch):
$changed" ;;
esac

# Pipeline code must not touch evaluation or ground truth.
leak=$(git grep -nE 'kelpod\.evaluation|ground_truth\.json|eval/manifests|kelpod-data/sealed' "$HEADC" -- src/kelpod \
        ':!src/kelpod/evaluation' 2>/dev/null)
[ -n "$leak" ] && bad "pipeline code references evaluation / ground truth:
$leak"

# No removed tests, no new skip/xfail.
added_skips=$(git diff "$MB" "$HEADC" -- tests | grep -E '^\+.*(pytest\.mark\.(skip|xfail)|pytest\.skip\(|@unittest\.skip)' )
[ -n "$added_skips" ] && bad "new skip/xfail markers:
$added_skips"
deleted_tests=$(git diff --diff-filter=D --name-only "$MB" "$HEADC" -- tests)
[ -n "$deleted_tests" ] && bad "test files deleted: $deleted_tests"

# Test count and test result, in a clean worktree of the candidate.
WT=$(mktemp -d); git worktree add -q --detach "$WT" "$HEADC"
WB=$(mktemp -d); git worktree add -q --detach "$WB" "$BASE"
count() { (cd "$1" && uv run --quiet pytest --collect-only -q 2>/dev/null | grep -oE '^[0-9]+ tests? collected' | grep -oE '^[0-9]+'); }
nc=$(count "$WT"); nb=$(count "$WB")
echo "tests collected: candidate=${nc:-?} main=${nb:-?}"
{ [ -z "$nc" ] || [ -z "$nb" ] || [ "$nc" -lt "$nb" ]; } && bad "test count dropped or could not be read"
(cd "$WT" && uv run --quiet pytest -q -x >/tmp/promo-pytest-$$.log 2>&1) || { tail -20 /tmp/promo-pytest-$$.log; bad "tests fail"; }

# Frozen splits and train/test overlap.
if [ -d eval/manifests ]; then
  (cd "$WT" && uv run --quiet python scripts/locked_eval.py --verify-manifests) || bad "manifest verification failed"
fi

if [ $fail -eq 0 ] && [ "$MODE" != "--no-eval" ]; then
  if [ -f "$WT/scripts/locked_eval.py" ]; then
    n=$(grep -c "| $CAND |" eval/TEST_LOG.md 2>/dev/null || echo 0)
    [ "$n" -ge 2 ] && bad "locked-run budget used up for $CAND (2 per branch)"
    [ $fail -eq 0 ] && { (cd "$REPO" && uv run --quiet python "$WT/scripts/locked_eval.py" \
        --candidate "$WT" --baseline "$WB" --branch "$CAND" --log eval/TEST_LOG.md) || bad "promotion rule not met"; }
  else
    bad "scripts/locked_eval.py does not exist yet: build it on an eval/* branch first"
  fi
fi

git worktree remove --force "$WT"; git worktree remove --force "$WB"
[ $fail -eq 0 ] && echo "PASS: $CAND may be promoted" || echo "NOT PROMOTABLE: $CAND"
exit $fail
