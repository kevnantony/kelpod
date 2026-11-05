# mypy: disable-error-code="attr-defined,no-untyped-call,misc,union-attr,str-unpack"
"""MLX LoRA/QLoRA training worker, run as a subprocess per trial (NFR-REL-01).

    python -m zllf.stages.train.mlx_worker <job.json>

The controller never loads the model itself; a Metal OOM or crash only kills this process.
Progress is printed to stdout as ``@@ZLLF {json}`` lines (no document text, only numbers).

Built on mlx-lm building blocks (``load``, ``linear_to_lora_layers``, ``ChatDataset`` with
``mask_prompt``, ``iterate_batches``, ``default_loss``) but with our own loop so that we get:
AdamW + warmup/cosine schedule, exact resume (adapter + optimizer state + step + data order)
from atomic checkpoints, step- and time-based checkpointing, and extra eval sets
(anchor / hallucination proxy) at the end.

Exit codes: 0 ok, 3 out of memory, 4 model missing, 1 other error.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_OOM = 3
EXIT_MODEL_MISSING = 4
PREFIX = "@@ZLLF "

OOM_SIGNATURES = (
    "out of memory", "outofmemory", "insufficient memory", "resource limit",
    "kiogpucommandbuffercallbackerroroutofmemory", "failed to allocate",
    "attempting to allocate", "maximum allowed buffer size", "std::bad_alloc",
)


def is_oom_message(text: str) -> bool:
    low = text.lower()
    return any(sig in low for sig in OOM_SIGNATURES)


def emit(event: str, **payload: Any) -> None:
    print(PREFIX + json.dumps({"event": event, **payload}), flush=True)


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def _read_jsonl(path: str | None) -> list[dict[str, Any]]:
    if not path or not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- model helpers
def lora_keys(model: Any, target: str) -> list[str]:
    """Layer-relative module paths of the linears to adapt for a target preset."""
    import mlx.nn as nn

    names: set[str] = set()

    def visit(path: str, m: Any) -> None:
        if isinstance(m, (nn.Linear, nn.QuantizedLinear)) or hasattr(m, "to_lora"):
            names.add(path)

    model.layers[0].apply_to_modules(visit)
    attn = {n for n in names if any(t in n for t in ("attn", "attention"))}
    mlp = {n for n in names if "mlp" in n or "feed_forward" in n}
    chosen = {"attention": attn, "mlp": mlp, "all": names}[target] or names
    return sorted(chosen)


def lora_parameters(job: dict[str, Any], model: Any) -> dict[str, Any]:
    return {
        "rank": int(job["rank"]),
        "scale": float(job["scale"]),
        "dropout": float(job["dropout"]),
        "keys": lora_keys(model, job["target_modules"]),
    }


def build_schedule(job: dict[str, Any], n_updates: int) -> Any:
    import mlx.optimizers as optim

    lr = float(job["learning_rate"])
    warmup = int(min(job.get("warmup_steps", 0), max(n_updates - 1, 0)))
    decay = optim.cosine_decay(lr, max(n_updates - warmup, 1), lr * 0.1)
    if warmup <= 0:
        return decay
    return optim.join_schedules([optim.linear_schedule(lr * 0.01, lr, warmup), decay],
                                [warmup])


def make_batch(items: list[tuple[list[int], int]], max_seq: int) -> tuple[Any, Any]:
    """Pad a list of (tokens, prompt_offset) to one batch (same layout as mlx-lm)."""
    import mlx.core as mx
    import numpy as np

    lengths = [min(len(t), max_seq) for t, _ in items]
    width = min(1 + 32 * ((max(lengths) + 31) // 32), max_seq)
    arr = np.zeros((len(items), width), np.int32)
    for j, (toks, _) in enumerate(items):
        n = min(lengths[j], width)
        arr[j, :n] = toks[:n]
        lengths[j] = n
    offsets = [off for _, off in items]
    return mx.array(arr), mx.array(list(zip(offsets, lengths, strict=True)))


def eval_loss(model: Any, dataset: Any, batch_size: int, max_seq: int,
              max_examples: int | None = None) -> float | None:
    """Mean per-token loss on completion tokens over the whole set (no dropped remainder)."""
    import mlx.core as mx
    from mlx_lm.tuner.trainer import default_loss

    n = len(dataset) if max_examples is None else min(len(dataset), max_examples)
    if n == 0:
        return None
    model.eval()
    total, toks = 0.0, 0.0
    for i in range(0, n, batch_size):
        batch = make_batch([dataset[j] for j in range(i, min(i + batch_size, n))], max_seq)
        loss, ntok = default_loss(model, *batch)
        mx.eval(loss, ntok)
        total += float(loss.item()) * float(ntok.item())
        toks += float(ntok.item())
    model.train()
    return total / toks if toks else None


def _dataset(rows: list[dict[str, Any]], tokenizer: Any, mask_prompt: bool) -> Any:
    from mlx_lm.tuner.datasets import CacheDataset, ChatDataset

    return CacheDataset(ChatDataset(rows, tokenizer, chat_key="messages",
                                    mask_prompt=mask_prompt))


def adapter_config(job: dict[str, Any], lora_params: dict[str, Any]) -> dict[str, Any]:
    """mlx-lm compatible adapter_config.json (``mlx_lm.load(model, adapter_path=...)``)."""
    return {
        "model": job["model"],
        "fine_tune_type": "lora",
        "num_layers": int(job.get("num_layers", -1)),
        "lora_parameters": lora_params,
        "max_seq_length": int(job["max_seq_length"]),
        "mask_prompt": bool(job.get("mask_prompt", True)),
        "seed": int(job.get("seed", 0)),
        "zllf": {"alpha": float(job["scale"]) * int(job["rank"]),
                 "target_modules": job["target_modules"]},
    }


# ---------------------------------------------------------------- checkpoints
def save_checkpoint(ckpt_dir: Path, model: Any, optimizer: Any, step: int,
                    extra: dict[str, Any]) -> None:
    """Atomic: write ``ckpt.new`` then swap it in for ``ckpt`` (old copy kept until done)."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    new = ckpt_dir.with_name(ckpt_dir.name + ".new")
    old = ckpt_dir.with_name(ckpt_dir.name + ".old")
    shutil.rmtree(new, ignore_errors=True)
    new.mkdir(parents=True)
    mx.save_safetensors(str(new / "adapters.safetensors"),
                        dict(tree_flatten(model.trainable_parameters())))
    mx.save_safetensors(str(new / "optimizer.safetensors"), dict(tree_flatten(optimizer.state)))
    (new / "state.json").write_text(json.dumps({"step": step, **extra}))
    if ckpt_dir.exists():
        shutil.rmtree(old, ignore_errors=True)
        os.replace(ckpt_dir, old)
    os.replace(new, ckpt_dir)
    shutil.rmtree(old, ignore_errors=True)


def find_checkpoint(ckpt_dir: Path) -> Path | None:
    for d in (ckpt_dir, ckpt_dir.with_name(ckpt_dir.name + ".old")):
        if (d / "state.json").exists() and (d / "optimizer.safetensors").exists():
            return d
    return None


# ---------------------------------------------------------------- modes
def _load(job: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    import mlx.core as mx
    import numpy as np
    from mlx_lm import load

    seed = int(job.get("seed", 0))
    mx.random.seed(seed)
    np.random.seed(seed)
    if job.get("memory_limit_bytes"):
        mx.set_memory_limit(int(job["memory_limit_bytes"]))
    model, tokenizer, config = load(job["model"], return_config=True)
    return model, tokenizer, config


def _apply_lora(model: Any, job: dict[str, Any]) -> dict[str, Any]:
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model.freeze()
    params = lora_parameters(job, model)
    n_layers = int(job.get("num_layers", -1))
    if n_layers > len(model.layers):
        n_layers = -1
    linear_to_lora_layers(model, n_layers, params)
    return params


def run_dry(job: dict[str, Any]) -> dict[str, Any]:
    """Base-model losses (anchor/halluc/valid) + a few steps at max seq length, max rank."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    from mlx_lm.tuner.trainer import default_loss, grad_checkpoint

    t0 = time.time()
    model, tokenizer, config = _load(job)
    base_bytes = mx.get_active_memory()
    max_seq = int(job["max_seq_length"])
    eb = int(job.get("eval_batch_size", 4))
    base: dict[str, float | None] = {}
    for name in ("anchor", "halluc", "valid"):
        rows = _read_jsonl(job.get(f"{name}_file"))
        ds = _dataset(rows, tokenizer, True)
        base[name] = eval_loss(model, ds, eb, max_seq, job.get("max_eval_examples"))
    emit("base_losses", **{k: v for k, v in base.items()})

    params = _apply_lora(model, job)
    if job.get("grad_checkpoint"):
        grad_checkpoint(model.layers[0])
    opt = optim.AdamW(learning_rate=1e-5)
    loss_and_grad = nn.value_and_grad(model, default_loss)
    b = int(job["batch_size"])
    rng = np.random.default_rng(int(job.get("seed", 0)))
    vocab = int(config.get("vocab_size", 32000))
    mx.reset_peak_memory()
    times: list[float] = []
    steps = int(job.get("dry_run_steps", 3))
    for _ in range(steps):
        arr = mx.array(rng.integers(0, vocab, size=(b, max_seq + 1), dtype=np.int32))
        lengths = mx.array([[0, max_seq]] * b)
        tic = time.perf_counter()
        (loss, _), grads = loss_and_grad(model, arr, lengths)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        times.append(time.perf_counter() - tic)
    steady = times[1:] or times
    tok_s = b * max_seq / (sum(steady) / len(steady))
    return {
        "mode": "dry_run",
        "base_losses": base,
        "base_bytes": float(base_bytes),
        "peak_bytes": float(mx.get_peak_memory()),
        "tokens_per_s": tok_s,
        "step_seconds": times,
        "model_config": {k: config.get(k) for k in (
            "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "head_dim", "vocab_size") if k in config},
        "lora_keys": params["keys"],
        "seconds": time.time() - t0,
    }


def run_train(job: dict[str, Any]) -> dict[str, Any]:
    """LoRA training loop with eval, checkpoints and exact resume."""
    from functools import partial

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map, tree_unflatten
    from mlx_lm.tuner.trainer import default_loss, grad_checkpoint, iterate_batches

    t0 = time.time()
    model, tokenizer, _ = _load(job)
    max_seq = int(job["max_seq_length"])
    mask = bool(job.get("mask_prompt", True))
    train_rows = _read_jsonl(job["train_file"])
    train_set = _dataset(train_rows, tokenizer, mask)
    valid_set = _dataset(_read_jsonl(job.get("valid_file")), tokenizer, mask)
    b = min(int(job["batch_size"]), len(train_set))
    accum = int(job.get("grad_accumulation", 1))
    iters = int(job["iters"])  # micro-batches
    n_updates = max(1, iters // accum)
    eval_every = max(1, int(job.get("eval_every", iters)))
    save_every = max(1, int(job.get("save_every", 100)))
    save_seconds = float(job.get("save_seconds", 900))
    eb = int(job.get("eval_batch_size", 4))
    max_eval = job.get("max_eval_examples")

    lora_params = _apply_lora(model, job)
    if job.get("grad_checkpoint"):
        grad_checkpoint(model.layers[0])
    optimizer = optim.AdamW(learning_rate=build_schedule(job, n_updates),
                            weight_decay=float(job.get("weight_decay", 0.0)))

    out_dir = Path(job["out_dir"])
    ckpt_dir = out_dir / "ckpt"
    metrics_path = out_dir / "metrics.jsonl"
    start = 0
    history: dict[str, list[Any]] = {"train": [], "val": []}
    ck = find_checkpoint(ckpt_dir) if job.get("resume", True) else None
    if ck is not None:
        state = json.loads((ck / "state.json").read_text())
        model.load_weights(str(ck / "adapters.safetensors"), strict=False)
        optimizer.state = tree_unflatten(list(mx.load(str(ck / "optimizer.safetensors")).items()))
        optimizer.init(model.trainable_parameters())
        start = int(state["step"])
        history = state.get("history", history)
        emit("resumed", step=start)

    state_list = [model.state, optimizer.state, mx.random.state]
    loss_value_and_grad = nn.value_and_grad(model, default_loss)

    @partial(mx.compile, inputs=state_list, outputs=state_list)
    def step(batch: Any, lengths: Any, prev_grad: Any, do_update: bool) -> Any:
        (lvalue, toks), grad = loss_value_and_grad(model, batch, lengths)
        if prev_grad is not None:
            grad = tree_map(lambda x, y: x + y, grad, prev_grad)
        if do_update:
            if accum > 1:
                grad = tree_map(lambda x: x / accum, grad)
            optimizer.update(model, grad)
            grad = None
        return lvalue, toks, grad

    def log(rec: dict[str, Any]) -> None:
        with metrics_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        emit(rec.pop("event"), **rec)

    batches = iterate_batches(train_set, b, max_seq, loop=True, seed=int(job.get("seed", 0)))
    for _ in range(start):  # replay the deterministic data order up to the checkpoint
        next(batches)
    if start == 0 and len(valid_set):
        v = eval_loss(model, valid_set, eb, max_seq, max_eval)
        history["val"].append([0, v])
        log({"event": "eval", "step": 0, "progress": 0.0, "val_loss": v})

    model.train()
    mx.reset_peak_memory()
    grad_accum = None
    losses, ntoks, tsteps, ttime = 0.0, 0.0, 0, 0.0
    last_save = time.time()
    report_every = max(1, int(job.get("report_every", 5)))
    tok_s_hist: list[float] = []
    for it in range(start + 1, iters + 1):
        batch, lengths = next(batches)
        tic = time.perf_counter()
        lvalue, toks, grad_accum = step(batch, lengths, grad_accum, it % accum == 0)
        mx.eval(state_list, lvalue, toks, grad_accum)
        ttime += time.perf_counter() - tic
        losses += float(lvalue.item())
        ntoks += float(toks.item())
        tsteps += 1
        if it % report_every == 0 or it == iters:
            tl = losses / tsteps
            tps = ntoks / ttime if ttime > 0 else 0.0
            if it > report_every:  # first window includes compilation
                tok_s_hist.append(tps)
            history["train"].append([it, tl])
            log({"event": "train", "step": it, "train_loss": tl, "tokens_per_s": tps,
                 "learning_rate": float(optimizer.learning_rate.item()),
                 "peak_memory_gb": mx.get_peak_memory() / 1e9})
            losses, ntoks, tsteps, ttime = 0.0, 0.0, 0, 0.0
        if len(valid_set) and (it % eval_every == 0 or it == iters):
            v = eval_loss(model, valid_set, eb, max_seq, max_eval)
            history["val"].append([it, v])
            log({"event": "eval", "step": it, "progress": it / iters, "val_loss": v})
        due = it % save_every == 0 or time.time() - last_save > save_seconds
        # only at update boundaries (gradient accumulators are not checkpointed)
        if it < iters and due and grad_accum is None:
            save_checkpoint(ckpt_dir, model, optimizer, it, {"history": history})
            last_save = time.time()
            emit("checkpoint", step=it)
        if job.get("fail_at_step") == it and start == 0:  # test hook: simulate a hard crash
            os._exit(int(job.get("fail_exit_code", 137)))

    adapter_dir = Path(job["adapter_dir"])
    tmp = adapter_dir.with_name(adapter_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    mx.save_safetensors(str(tmp / "adapters.safetensors"),
                        dict(tree_flatten(model.trainable_parameters())))
    (tmp / "adapter_config.json").write_text(json.dumps(adapter_config(job, lora_params),
                                                        indent=2))
    shutil.rmtree(adapter_dir, ignore_errors=True)
    os.replace(tmp, adapter_dir)

    extra: dict[str, float | None] = {}
    for name in ("anchor", "halluc"):
        rows = _read_jsonl(job.get(f"{name}_file"))
        extra[name] = eval_loss(model, _dataset(rows, tokenizer, True), eb, max_seq, max_eval)
    vals = [v for _, v in history["val"] if v is not None]
    n_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    return {
        "mode": "train",
        "val_loss": vals[-1] if vals else None,
        "anchor_loss": extra["anchor"],
        "halluc_loss": extra["halluc"],
        "train_loss": history["train"][-1][1] if history["train"] else None,
        "history": history,
        "tokens_per_s": (sum(tok_s_hist) / len(tok_s_hist)) if tok_s_hist else None,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "trainable_params": int(n_params),
        "iters": iters,
        "batch_size": b,
        "grad_accumulation": accum,
        "resumed_from": start,
        "seconds": time.time() - t0,
        "lora_keys": lora_params["keys"],
    }


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    job = json.loads(Path(argv[0]).read_text())
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = run_dry(job) if job.get("mode") == "dry_run" else run_train(job)
    except Exception as exc:  # classify for the controller; message has no document text
        msg = f"{type(exc).__name__}: {exc}"[:500]
        if is_oom_message(msg):
            emit("oom", error=msg)
            return EXIT_OOM
        missing = ("LocalEntryNotFoundError", "RepositoryNotFound", "snapshot_download")
        if any(m in msg for m in missing) or (
                isinstance(exc, OSError) and "model" in msg.lower()):
            emit("model_missing", error=msg)
            return EXIT_MODEL_MISSING
        emit("error", error=msg)
        return EXIT_ERROR
    if result.get("val_loss") is not None and not math.isfinite(result["val_loss"]):
        result["val_loss"] = None
    _atomic_json(out_dir / "result.json", result)
    emit("done", seconds=result.get("seconds"))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
