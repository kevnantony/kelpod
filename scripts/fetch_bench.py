#!/usr/bin/env python3
"""Fetch SMALL subsets of permissively licensed benchmarks for the forgetting gate (G-2) and
the replay buffer (train stage). Build-host only (needs ``ZLLF_ALLOW_ONLINE=1``); the appliance
never runs this — it reads the committed/generated files offline.

Outputs::

    bench/anchor/<suite>.jsonl     EvalItem JSON lines (suite = "anchor/<suite>")
    bench/anchor/sources.json      per suite: dataset, config, split, licence, URL, rows, sha256
    bench/replay/replay.jsonl      {"messages": [...], "source": "<dataset>#<row_idx>"}
    bench/replay/sources.json

Rows are read through the Hugging Face datasets-server ``/rows`` API (JSON, ≤100 rows per
request), so only the sampled rows travel over the (slow) network — no parquet shards.
Sampling is deterministic (``--seed``): a few random blocks of consecutive rows per dataset.

The ``json_schema`` suite is self-written (generated deterministically, no download).
Hand-written fallback suites live in ``bench/anchor/fallback/`` and are used for any category
for which no fetched suite exists, so evaluation works fully offline.

Examples::

    ZLLF_ALLOW_ONLINE=1 python3 scripts/fetch_bench.py                  # everything
    ZLLF_ALLOW_ONLINE=1 python3 scripts/fetch_bench.py --only gsm8k mmlu
    python3 scripts/fetch_bench.py --only json_schema                   # offline, generated

Stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import string
import sys
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ANCHOR_DIR = REPO_ROOT / "bench" / "anchor"
REPLAY_DIR = REPO_ROOT / "bench" / "replay"
API = "https://datasets-server.huggingface.co"
USER_AGENT = "zllf-fetch-bench/0.1 (offline LoRA appliance; small benchmark subsets)"
LETTERS = string.ascii_uppercase

Row = dict[str, Any]
Item = dict[str, Any]


# ---------------------------------------------------------------- network
def _get_json(url: str, retries: int = 3, timeout: float = 120.0) -> Any:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"GET failed: {url}: {last}")


def num_rows(dataset: str, config: str, split: str) -> int:
    q = urllib.parse.urlencode({"dataset": dataset, "config": config})
    data = _get_json(f"{API}/size?{q}")
    for s in data["size"]["splits"]:
        if s["config"] == config and s["split"] == split:
            return int(s["num_rows"])
    raise RuntimeError(f"split {split!r} not found for {dataset}/{config}")


def fetch_rows(dataset: str, config: str, split: str, n: int, seed: int,
               block: int = 50) -> list[tuple[int, Row]]:
    """~n rows as (row_idx, row): random non-overlapping blocks of consecutive rows."""
    total = num_rows(dataset, config, split)
    block = min(block, 100, total)
    n_blocks = max(1, -(-min(n, total) // block))
    starts = list(range(0, total - block + 1, block))
    rng = random.Random(f"{seed}:{dataset}:{config}:{split}")
    chosen = sorted(rng.sample(starts, min(n_blocks, len(starts))))
    out: list[tuple[int, Row]] = []
    for off in chosen:
        q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                    "offset": off, "length": block})
        data = _get_json(f"{API}/rows?{q}")
        out.extend((int(r["row_idx"]), r["row"]) for r in data["rows"])
        print(f"  {dataset}: {len(out)} rows", file=sys.stderr)
    return out


# ---------------------------------------------------------------- converters
def _user(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}]


def _mc_prompt(question: str, choices: list[str]) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return f"{question.strip()}\n{opts}\nAnswer with the letter of the correct option only."


def conv_mmlu(idx: int, row: Row) -> Item | None:
    choices = [str(c) for c in row["choices"]]
    ans = int(row["answer"])
    subject = str(row.get("subject", "")).replace("_", " ")
    q = f"The following is a multiple choice question about {subject}.\n\n{row['question']}"
    return {"item_id": f"mmlu-{idx}", "suite": "anchor/mmlu",
            "prompt": _user(_mc_prompt(q, choices)), "choices": choices,
            "reference": choices[ans]}


def conv_arc(idx: int, row: Row) -> Item | None:
    texts = [str(t) for t in row["choices"]["text"]]
    labels = [str(lb) for lb in row["choices"]["label"]]
    if row["answerKey"] not in labels:
        return None
    ref = texts[labels.index(row["answerKey"])]
    return {"item_id": f"arc_easy-{idx}", "suite": "anchor/arc_easy",
            "prompt": _user(_mc_prompt(str(row["question"]), texts)), "choices": texts,
            "reference": ref}


def conv_gsm8k(idx: int, row: Row) -> Item | None:
    answer = str(row["answer"]).split("####")[-1].strip().replace(",", "")
    try:
        float(answer)
    except ValueError:
        return None
    q = (f"{str(row['question']).strip()} Show brief working, then end with "
         "'The answer is <number>.'")
    return {"item_id": f"gsm8k-{idx}", "suite": "anchor/gsm8k", "prompt": _user(q),
            "reference": answer}


def _ifeval_check(iid: str, kw: dict[str, Any]) -> dict[str, Any] | None:
    rel = kw.get("relation")
    if iid == "punctuation:no_comma":
        return {"type": "no_commas"}
    if iid == "change_case:english_lowercase":
        return {"type": "lowercase"}
    if iid == "change_case:english_capital":
        return {"type": "uppercase"}
    if iid == "length_constraints:number_words" and kw.get("num_words"):
        n = int(kw["num_words"])
        return ({"type": "max_words", "n": n - 1} if rel == "less than"
                else {"type": "min_words", "n": n})
    if iid == "length_constraints:number_sentences" and kw.get("num_sentences"):
        n = int(kw["num_sentences"])
        return ({"type": "max_sentences", "n": n - 1} if rel == "less than"
                else {"type": "min_sentences", "n": n})
    if iid == "detectable_format:number_bullet_lists" and kw.get("num_bullets"):
        return {"type": "bullets", "n": int(kw["num_bullets"])}
    if iid == "keywords:existence" and kw.get("keywords"):
        return {"type": "contains", "words": list(kw["keywords"])}
    if iid == "keywords:forbidden_words" and kw.get("forbidden_words"):
        return {"type": "not_contains", "words": list(kw["forbidden_words"])}
    if iid == "startend:end_checker" and kw.get("end_phrase"):
        return {"type": "ends_with", "text": str(kw["end_phrase"])}
    if iid == "startend:quotation":
        return {"type": "quoted"}
    if iid == "detectable_format:title":
        return {"type": "title"}
    if iid == "detectable_format:json_format":
        return {"type": "json_only"}
    if iid == "length_constraints:number_paragraphs" and kw.get("num_paragraphs"):
        return {"type": "paragraphs", "n": int(kw["num_paragraphs"]), "sep": "***"}
    return None


def conv_ifeval(idx: int, row: Row) -> Item | None:
    checks = []
    kwargs = row.get("kwargs") or [{}] * len(row["instruction_id_list"])
    for iid, kw in zip(row["instruction_id_list"], kwargs):  # noqa: B905 (runs on py3.9)
        chk = _ifeval_check(str(iid), {k: v for k, v in (kw or {}).items() if v is not None})
        if chk is None:
            return None  # skip items with constraints we cannot verify
        checks.append(chk)
    if not checks:
        return None
    return {"item_id": f"ifeval-{row.get('key', idx)}", "suite": "anchor/ifeval",
            "prompt": _user(str(row["prompt"])), "reference_json": {"checks": checks}}


def conv_dolly(idx: int, row: Row) -> dict[str, Any] | None:
    instr, ctx, resp = (str(row.get(k) or "").strip()
                        for k in ("instruction", "context", "response"))
    if not instr or not resp or len(resp) > 1500 or len(ctx) > 2000:
        return None
    user = f"{instr}\n\n{ctx}" if ctx else instr
    return {"messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": resp}],
            "source": f"databricks/databricks-dolly-15k#{idx}",
            "category": row.get("category")}


# ---------------------------------------------------------------- self-written json suite
_NAMES = ["Ada", "Ben", "Chen", "Dara", "Eli", "Fatima", "Gus", "Hana", "Ivo", "Jun"]
_CITIES = ["Berlin", "Lyon", "Porto", "Graz", "Turku", "Ghent", "Bergen", "Brno"]
_ITEMS = ["valve", "filter", "pump", "sensor", "relay", "gasket", "fuse", "hose"]


def gen_json_schema(n: int, seed: int) -> list[Item]:
    """Deterministic, self-written JSON formatting items (no licence constraints)."""
    rng = random.Random(f"{seed}:json_schema")
    out: list[Item] = []
    for i in range(n):
        kind = i % 4
        if kind == 0:
            v = {"name": rng.choice(_NAMES), "age": rng.randint(18, 90)}
            s = {"type": "object", "properties": {"name": {"type": "string"},
                                                  "age": {"type": "integer"}},
                 "required": ["name", "age"], "additionalProperties": False}
            q = (f"Return a JSON object with keys 'name' (string) and 'age' (integer) for "
                 f"{v['name']}, aged {v['age']}. No other keys.")
        elif kind == 1:
            k = rng.randint(2, 5)
            v = {"part": rng.choice(_ITEMS), "quantity": k, "in_stock": rng.random() < 0.5}
            s = {"type": "object", "properties": {"part": {"type": "string"},
                                                  "quantity": {"type": "integer"},
                                                  "in_stock": {"type": "boolean"}},
                 "required": ["part", "quantity", "in_stock"]}
            q = (f"Return a JSON object with 'part' = \"{v['part']}\", 'quantity' = {k} and "
                 f"'in_stock' = {str(v['in_stock']).lower()}.")
        elif kind == 2:
            nums = rng.sample(range(1, 50), rng.randint(3, 6))
            v = {"values": nums, "count": len(nums)}
            s = {"type": "object", "properties": {
                "values": {"type": "array", "items": {"type": "integer"},
                           "minItems": len(nums), "maxItems": len(nums)},
                "count": {"type": "integer"}}, "required": ["values", "count"]}
            q = (f"Return a JSON object with 'values' = the list {nums} (in this order) and "
                 "'count' = the number of values.")
        else:
            city = rng.choice(_CITIES)
            v = {"city": city, "location": {"lat": round(rng.uniform(40, 60), 2),
                                            "lon": round(rng.uniform(-5, 25), 2)}}
            s = {"type": "object", "properties": {
                "city": {"type": "string"},
                "location": {"type": "object", "properties": {"lat": {"type": "number"},
                                                              "lon": {"type": "number"}},
                             "required": ["lat", "lon"]}}, "required": ["city", "location"]}
            q = (f"Return a JSON object with 'city' = \"{city}\" and a nested object "
                 f"'location' with 'lat' = {v['location']['lat']} and "
                 f"'lon' = {v['location']['lon']}.")
        out.append({"item_id": f"json_schema-{i:04d}", "suite": "anchor/json_schema",
                    "prompt": _user(q + " Reply with JSON only."),
                    "reference_json": {"checks": [{"type": "json_only"},
                                                  {"type": "json", "schema": s},
                                                  {"type": "json_equals", "value": v}]}})
    return out


# ---------------------------------------------------------------- registry
# suite -> (dataset, config, split, licence, converter, n_rows_to_fetch, n_keep)
SUITES: dict[str, tuple[str, str, str, str, Callable[[int, Row], Item | None], int, int]] = {
    "mmlu": ("cais/mmlu", "all", "test", "MIT", conv_mmlu, 200, 200),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "test", "CC BY-SA 4.0", conv_arc, 150, 150),
    "gsm8k": ("openai/gsm8k", "main", "test", "MIT", conv_gsm8k, 150, 150),
    "ifeval": ("google/IFEval", "default", "train", "Apache-2.0", conv_ifeval, 541, 150),
}
REPLAY = ("databricks/databricks-dolly-15k", "default", "train", "CC BY-SA 3.0", 1200, 1000)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(data, encoding="utf-8")
    return hashlib.sha256(data.encode()).hexdigest()


def _update_sources(path: Path, key: str, entry: dict[str, Any]) -> None:
    sources = json.loads(path.read_text()) if path.exists() else {}
    sources[key] = entry
    path.write_text(json.dumps(sources, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fetch_suite(name: str, seed: int) -> int:
    dataset, config, split, licence, conv, n_fetch, n_keep = SUITES[name]
    rows = fetch_rows(dataset, config, split, n_fetch, seed,
                      block=100 if n_fetch > 300 else 50)
    items = [it for idx, row in rows if (it := conv(idx, row)) is not None]
    rng = random.Random(f"{seed}:{name}:keep")
    if len(items) > n_keep:
        items = sorted(rng.sample(items, n_keep), key=lambda it: str(it["item_id"]))
    digest = _write_jsonl(ANCHOR_DIR / f"{name}.jsonl", items)
    _update_sources(ANCHOR_DIR / "sources.json", name, {
        "dataset": dataset, "config": config, "split": split, "licence": licence,
        "url": f"https://huggingface.co/datasets/{dataset}", "rows": len(items),
        "sha256": digest, "fetched": time.strftime("%Y-%m-%d"), "seed": seed})
    return len(items)


def fetch_replay(seed: int) -> int:
    dataset, config, split, licence, n_fetch, n_keep = REPLAY
    rows = fetch_rows(dataset, config, split, n_fetch, seed, block=100)
    samples = [s for idx, row in rows if (s := conv_dolly(idx, row)) is not None][:n_keep]
    digest = _write_jsonl(REPLAY_DIR / "replay.jsonl", samples)
    _update_sources(REPLAY_DIR / "sources.json", "replay", {
        "dataset": dataset, "config": config, "split": split, "licence": licence,
        "url": f"https://huggingface.co/datasets/{dataset}", "rows": len(samples),
        "sha256": digest, "fetched": time.strftime("%Y-%m-%d"), "seed": seed,
        "attribution": "databricks-dolly-15k, Copyright Databricks, Inc., CC BY-SA 3.0"})
    return len(samples)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", default=None,
                    help=f"subset of {[*SUITES, 'json_schema', 'replay']}")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--json-items", type=int, default=100)
    args = ap.parse_args(argv)
    wanted = args.only or [*SUITES, "json_schema", "replay"]
    online = [w for w in wanted if w != "json_schema"]
    if online and os.environ.get("ZLLF_ALLOW_ONLINE") != "1":
        print("refusing to download: set ZLLF_ALLOW_ONLINE=1 (build host only)", file=sys.stderr)
        return 2
    failed = []
    for name in wanted:
        try:
            if name == "json_schema":
                items = gen_json_schema(args.json_items, args.seed)
                digest = _write_jsonl(ANCHOR_DIR / "json_schema.jsonl", items)
                _update_sources(ANCHOR_DIR / "sources.json", "json_schema", {
                    "dataset": "self-written (scripts/fetch_bench.py:gen_json_schema)",
                    "licence": "same as this repository", "rows": len(items),
                    "sha256": digest, "seed": args.seed})
                n = len(items)
            elif name == "replay":
                n = fetch_replay(args.seed)
            elif name in SUITES:
                n = fetch_suite(name, args.seed)
            else:
                print(f"unknown suite {name!r}", file=sys.stderr)
                failed.append(name)
                continue
            print(f"{name}: {n} items")
        except Exception as exc:
            print(f"{name}: FAILED ({exc}); offline fallback suites remain in use",
                  file=sys.stderr)
            failed.append(name)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
