"""MLX backend (Apple Silicon) for :class:`zllf.llm.base.LLMBackend`.

* Models load from the local Hugging Face cache only (repo id -> snapshot path); a missing
  model raises ``ZLLFError(MODEL_MISSING)``. Nothing is ever downloaded.
* Generation uses ``mlx_lm.batch_generate`` (continuous batching) when available, else a
  per-prompt ``mlx_lm.generate`` loop. Sampling via ``mlx_lm.sample_utils.make_sampler``;
  ``SamplingParams.seed`` seeds ``mx.random`` before each call.
* LoRA adapters (mlx-lm format: ``adapters.safetensors`` + ``adapter_config.json``) are
  hot-swapped on one cached base model: LoRA wrappers are inserted with mlx-lm's
  ``load_adapters`` and removed again by restoring the wrapped base modules, so the base
  weights are never modified and base outputs are unaffected after an adapter is removed.
* ``json_schema``: MLX has no constrained decoding here, so it is best effort: a
  schema-aware instruction, JSON extraction from the output, ``jsonschema`` validation and up
  to ``json_repair_retries`` greedy repair turns.

The pure helpers (JSON extraction/validation, prompt building) import no MLX, so they can be
unit-tested anywhere.
"""

from __future__ import annotations

import gc
import json
import logging
import re
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from zllf.core.errors import ErrorCode, ZLLFError
from zllf.core.schemas import Message
from zllf.llm.base import Chat, LLMBackend, SamplingParams

log = logging.getLogger(__name__)

ADAPTER_WEIGHTS = "adapters.safetensors"
ADAPTER_CONFIG = "adapter_config.json"

# ----------------------------------------------------------------------------- model lookup


def resolve_model_path(model_id: str) -> Path:
    """Local directory for ``model_id`` (a local path or an HF repo id in the local cache).

    Raises ``ZLLFError(MODEL_MISSING)`` if the model is not fully present. Never downloads.
    """
    local = Path(model_id).expanduser()
    if local.is_dir():
        path = local
    else:
        try:
            from huggingface_hub import snapshot_download
            from huggingface_hub.errors import LocalEntryNotFoundError
        except ImportError as exc:  # pragma: no cover - huggingface_hub ships with mlx-lm
            raise ZLLFError(ErrorCode.MODEL_MISSING, f"cannot resolve {model_id}") from exc
        try:
            path = Path(snapshot_download(model_id, local_files_only=True))
        except (LocalEntryNotFoundError, FileNotFoundError, ValueError, OSError) as exc:
            raise ZLLFError(
                ErrorCode.MODEL_MISSING, f"model {model_id!r} is not in the local cache"
            ) from exc
    _check_complete(path, model_id)
    return path


def _check_complete(path: Path, model_id: str) -> None:
    if not (path / "config.json").is_file():
        raise ZLLFError(ErrorCode.MODEL_MISSING, f"{model_id!r}: config.json missing")
    index = path / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        missing = [s for s in shards if not (path / s).is_file()]
        if missing:
            raise ZLLFError(
                ErrorCode.MODEL_MISSING, f"{model_id!r}: {len(missing)} weight shards missing"
            )
    elif not any(path.glob("*.safetensors")):
        raise ZLLFError(ErrorCode.MODEL_MISSING, f"{model_id!r}: no weights found")


def check_adapter_dir(adapter_path: Path) -> Path:
    """Validate an mlx-lm adapter directory; return its resolved path."""
    p = Path(adapter_path).expanduser().resolve()
    if not (p / ADAPTER_WEIGHTS).is_file() or not (p / ADAPTER_CONFIG).is_file():
        raise ZLLFError(
            ErrorCode.MODEL_MISSING,
            f"adapter at {p} needs {ADAPTER_WEIGHTS} and {ADAPTER_CONFIG}",
        )
    cfg = json.loads((p / ADAPTER_CONFIG).read_text(encoding="utf-8"))
    if cfg.get("fine_tune_type", "lora") == "full":
        raise ZLLFError(ErrorCode.MODEL_MISSING, "full fine-tune checkpoints are not adapters")
    return p


# ----------------------------------------------------------------------------- JSON helpers

_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_SMART_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})


def _close_brackets(text: str) -> str:
    """Append missing closing brackets/quotes to a truncated JSON prefix."""
    stack: list[str] = []
    in_str = False
    escape = False
    for ch in text:
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    out = text + ('"' if in_str else "")
    out = _TRAILING_COMMA.sub(r"\1", out.rstrip().rstrip(","))
    return out + "".join(reversed(stack))


def json_candidates(text: str) -> list[Any]:
    """All JSON objects/arrays decodable from ``text``, in order of appearance.

    Tolerates code fences, prose around the JSON, trailing commas, smart quotes and (as a
    last resort) truncated output with unclosed brackets.
    """
    decoder = json.JSONDecoder()
    found: list[Any] = []
    for variant in (text, _TRAILING_COMMA.sub(r"\1", text.translate(_SMART_QUOTES))):
        pos = 0
        while True:
            starts = [i for i in (variant.find("{", pos), variant.find("[", pos)) if i >= 0]
            if not starts:
                break
            start = min(starts)
            try:
                value, end = decoder.raw_decode(variant, start)
            except json.JSONDecodeError:
                pos = start + 1
                continue
            if isinstance(value, dict | list):
                found.append(value)
            pos = end
        if found:
            return found
    # truncated output: close what is open after the first bracket
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if starts:
        try:
            value = json.loads(_close_brackets(text[min(starts) :].translate(_SMART_QUOTES)))
        except json.JSONDecodeError:
            return []
        if isinstance(value, dict | list):
            return [value]
    return []


def schema_errors(value: Any, schema: dict[str, Any], limit: int = 3) -> list[str]:
    """Human-readable validation errors (empty list = valid)."""
    import jsonschema

    cls = jsonschema.validators.validator_for(schema)
    validator = cls(schema)
    errs = sorted(validator.iter_errors(value), key=lambda e: list(e.absolute_path))
    out = []
    for e in errs[:limit]:
        where = "/".join(str(p) for p in e.absolute_path) or "(root)"
        out.append(f"{where}: {e.message}")
    return out


@dataclass(frozen=True)
class JSONResult:
    text: str  # what to return to the caller
    valid: bool
    parsed: bool
    errors: tuple[str, ...] = ()


def extract_json(text: str, schema: dict[str, Any] | None = None) -> JSONResult:
    """Best JSON value in ``text``: the first schema-valid candidate, else the first parsed
    candidate (``valid=False``), else the raw text (``parsed=False``)."""
    candidates = json_candidates(text)
    if not candidates:
        return JSONResult(text, valid=False, parsed=False, errors=("no JSON value found",))
    first_errors: list[str] | None = None
    for value in candidates:
        errs = schema_errors(value, schema) if schema is not None else []
        if not errs:
            return JSONResult(json.dumps(value, ensure_ascii=False), valid=True, parsed=True)
        if first_errors is None:
            first_errors = errs
    return JSONResult(
        json.dumps(candidates[0], ensure_ascii=False),
        valid=False,
        parsed=True,
        errors=tuple(first_errors or ()),
    )


def schema_instruction(schema: dict[str, Any]) -> str:
    return (
        "Respond with a single JSON value that conforms to this JSON Schema. Output only the "
        "JSON, with no explanation and no code fences.\nJSON Schema:\n"
        + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    )


def with_schema_instruction(chat: Chat, schema: dict[str, Any]) -> list[Message]:
    """Copy of ``chat`` with the schema instruction appended to (or as) the system message."""
    instr = schema_instruction(schema)
    msgs = list(chat)
    if msgs and msgs[0].role == "system":
        return [Message(role="system", content=f"{msgs[0].content}\n\n{instr}"), *msgs[1:]]
    return [Message(role="system", content=instr), *msgs]


def repair_turns(chat: Sequence[Message], bad_output: str, errors: Sequence[str]) -> list[Message]:
    """Chat asking the model to fix its previous (invalid) JSON answer."""
    problems = "; ".join(errors) if errors else "the output was not valid JSON"
    return [
        *chat,
        Message(role="assistant", content=bad_output),
        Message(
            role="user",
            content=f"That answer is not valid against the schema ({problems}). "
            "Reply again with only the corrected JSON value.",
        ),
    ]


def apply_stop(text: str, stop: Sequence[str]) -> str:
    """Cut ``text`` at the earliest stop string."""
    cut = len(text)
    for s in stop:
        if s:
            i = text.find(s)
            if 0 <= i < cut:
                cut = i
    return text[:cut]


# ----------------------------------------------------------------------------- model cache


@dataclass
class _Handle:
    """One loaded base model shared by all backends using the same weights."""

    path: Path
    model: Any
    tokenizer: Any
    load_seconds: float
    refs: int = 0
    adapter: Path | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)


_HANDLES: dict[Path, _Handle] = {}
_HANDLES_LOCK = threading.Lock()


def _acquire(path: Path) -> _Handle:
    with _HANDLES_LOCK:
        handle = _HANDLES.get(path)
        if handle is None:
            from mlx_lm import load

            t0 = time.perf_counter()
            try:
                model, tokenizer = load(str(path))[:2]
            except FileNotFoundError as exc:
                raise ZLLFError(ErrorCode.MODEL_MISSING, f"cannot load {path.name}") from exc
            model.eval()
            handle = _Handle(path, model, tokenizer, time.perf_counter() - t0)
            _HANDLES[path] = handle
        handle.refs += 1
        return handle


def _release(handle: _Handle) -> None:
    import mlx.core as mx

    with _HANDLES_LOCK:
        handle.refs -= 1
        if handle.refs <= 0:
            _HANDLES.pop(handle.path, None)
            handle.model = None
            handle.tokenizer = None
    gc.collect()
    mx.clear_cache()


def _remove_lora(model: Any) -> int:
    """Replace every LoRA/DoRA wrapper by the base module it wraps. Returns the count."""
    from mlx.utils import tree_unflatten
    from mlx_lm.tuner.dora import DoRAEmbedding, DoRALinear
    from mlx_lm.tuner.lora import LoRAEmbedding, LoRALinear, LoRASwitchLinear

    linear_types = (LoRALinear, LoRASwitchLinear, DoRALinear)
    emb_types = (LoRAEmbedding, DoRAEmbedding)
    reset = []
    for name, module in model.named_modules():
        if isinstance(module, linear_types):
            reset.append((name, module.linear))
        elif isinstance(module, emb_types):
            reset.append((name, module.embedding))
    if reset:
        model.update_modules(tree_unflatten(reset))
    return len(reset)


def _set_adapter(handle: _Handle, adapter_path: Path | None) -> None:
    """Make ``adapter_path`` (or no adapter) active on the shared model."""
    target = check_adapter_dir(adapter_path) if adapter_path is not None else None
    if handle.adapter == target:
        return
    if handle.adapter is not None:
        _remove_lora(handle.model)
        handle.adapter = None
    if target is not None:
        from mlx_lm.tuner.utils import load_adapters

        try:
            load_adapters(handle.model, str(target))
        except Exception:
            _remove_lora(handle.model)  # never leave a half-applied adapter behind
            raise
        handle.adapter = target
    handle.model.eval()


# ----------------------------------------------------------------------------- backend


@dataclass
class GenerationStats:
    prompt_tokens: int = 0
    generation_tokens: int = 0
    seconds: float = 0.0
    peak_memory_gb: float = 0.0

    @property
    def generation_tps(self) -> float:
        return self.generation_tokens / self.seconds if self.seconds > 0 else 0.0


class MLXBackend(LLMBackend):
    """``LLMBackend`` on Apple MLX via ``mlx-lm``."""

    def __init__(
        self,
        model_id: str,
        max_seq_len: int = 2048,
        *,
        json_repair_retries: int = 2,
        loglik_token_budget: int = 4096,
    ) -> None:
        try:
            import mlx.core  # noqa: F401
            import mlx_lm  # noqa: F401
        except ImportError as exc:
            raise ZLLFError(ErrorCode.MODEL_MISSING, "mlx / mlx-lm is not installed") from exc
        self.model_id = model_id
        self.max_seq_len = max_seq_len
        self.json_repair_retries = json_repair_retries
        self.loglik_token_budget = loglik_token_budget
        self.model_path = resolve_model_path(model_id)
        self._handle: _Handle | None = _acquire(self.model_path)
        self.load_seconds = self._handle.load_seconds
        self.last_stats = GenerationStats()

    # ------------------------------------------------------------------ internals
    @property
    def handle(self) -> _Handle:
        if self._handle is None:
            raise RuntimeError("MLXBackend is closed")
        return self._handle

    @property
    def tokenizer(self) -> Any:
        return self.handle.tokenizer

    @property
    def model(self) -> Any:
        return self.handle.model

    def _template(self, chat: Sequence[Message], *, generation_prompt: bool = True) -> str:
        msgs = [{"role": m.role, "content": m.content} for m in chat]
        text = self.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=generation_prompt, tokenize=False
        )
        return str(text)

    def _encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def _prompt_ids(self, chat: Sequence[Message], budget: int) -> list[int]:
        """Templated prompt ids, shortening the longest message until it fits ``budget``."""
        msgs = list(chat)
        ids = self._encode(self._template(msgs))
        for _ in range(4):
            overflow = len(ids) - budget
            if overflow <= 0:
                return ids
            i = max(range(len(msgs)), key=lambda k: len(msgs[k].content))
            content_ids = self._encode(msgs[i].content)
            keep = max(0, len(content_ids) - overflow - 8)
            msgs[i] = Message(role=msgs[i].role, content=self.tokenizer.decode(content_ids[:keep]))
            ids = self._encode(self._template(msgs))
        return ids[-budget:] if len(ids) > budget else ids

    def _generate(self, chats: Sequence[Sequence[Message]], params: SamplingParams) -> list[str]:
        import mlx.core as mx
        import mlx_lm
        from mlx_lm.sample_utils import make_sampler

        if not chats:
            return []
        max_tokens = max(1, min(params.max_tokens, self.max_seq_len - 1))
        budget = max(self.max_seq_len - max_tokens, self.max_seq_len // 2)
        prompts = [self._prompt_ids(c, budget) for c in chats]
        top_p = params.top_p if 0.0 < params.top_p < 1.0 else 0.0
        sampler = make_sampler(temp=max(params.temperature, 0.0), top_p=top_p)
        mx.random.seed(params.seed)
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        batch_generate = getattr(mlx_lm, "batch_generate", None)
        if batch_generate is not None and len(prompts) > 1:
            resp = batch_generate(
                self.model, self.tokenizer, prompts, max_tokens=max_tokens, sampler=sampler
            )
            texts = list(resp.texts)
            n_gen = int(resp.stats.generation_tokens)
        else:
            texts = []
            n_gen = 0
            for ids in prompts:
                texts.append(
                    mlx_lm.generate(
                        self.model, self.tokenizer, ids, max_tokens=max_tokens, sampler=sampler
                    )
                )
                n_gen += len(self._encode(texts[-1]))
        self.last_stats = GenerationStats(
            prompt_tokens=sum(len(p) for p in prompts),
            generation_tokens=n_gen,
            seconds=time.perf_counter() - t0,
            peak_memory_gb=mx.get_peak_memory() / 1e9,
        )
        return [apply_stop(t, params.stop) for t in texts]

    # ------------------------------------------------------------------ public API
    def chat(
        self,
        batch: Sequence[Chat],
        params: SamplingParams = SamplingParams(),
        *,
        json_schema: dict[str, Any] | None = None,
        adapter_path: Path | None = None,
    ) -> list[str]:
        """Generate one completion per chat (see module docstring for JSON mode)."""
        with self.handle.lock:
            _set_adapter(self.handle, adapter_path)
            if json_schema is None:
                return self._generate(batch, params)
            return self._chat_json(batch, params, json_schema)

    def _chat_json(
        self, batch: Sequence[Chat], params: SamplingParams, schema: dict[str, Any]
    ) -> list[str]:
        chats = [with_schema_instruction(c, schema) for c in batch]
        raw = self._generate(chats, params)
        results = [extract_json(t, schema) for t in raw]
        greedy = SamplingParams(
            temperature=0.0, max_tokens=params.max_tokens, top_p=1.0, seed=params.seed
        )
        for _ in range(self.json_repair_retries):
            todo = [i for i, r in enumerate(results) if not r.valid]
            if not todo:
                break
            fixes = self._generate(
                [repair_turns(chats[i], raw[i], results[i].errors) for i in todo], greedy
            )
            for i, text in zip(todo, fixes, strict=True):
                new = extract_json(text, schema)
                # prefer valid > parsed > raw; keep the newest raw text for the next repair
                if new.valid or (new.parsed and not results[i].parsed):
                    results[i] = new
                raw[i] = text
        return [r.text for r in results]

    def loglikelihood(
        self,
        prompts: Sequence[Chat],
        completions: Sequence[str],
        *,
        adapter_path: Path | None = None,
    ) -> list[float]:
        """Sum of completion token log-probs given the chat-templated prompt (generation
        prompt included). The completion is tokenised together with the prompt and the
        boundary is the longest common prefix with the prompt's own tokenisation, so
        merges across the boundary are attributed to the completion."""
        if len(prompts) != len(completions):
            raise ValueError("prompts and completions must have the same length")
        rows: list[tuple[list[int], int]] = []
        for chat, completion in zip(prompts, completions, strict=True):
            prompt_text = self._template(chat)
            ctx = self._encode(prompt_text)
            full = self._encode(prompt_text + completion)
            n_ctx = _common_prefix_len(ctx, full)
            if n_ctx == 0:
                raise ValueError("empty prompt after templating")
            if len(full) > self.max_seq_len:  # keep the completion, drop prompt head
                drop = min(len(full) - self.max_seq_len, n_ctx - 1)
                full, n_ctx = full[drop:], n_ctx - drop
            rows.append((full, n_ctx))

        with self.handle.lock:
            _set_adapter(self.handle, adapter_path)
            return self._score(rows)

    def _score(self, rows: list[tuple[list[int], int]]) -> list[float]:
        import mlx.core as mx

        out = [0.0] * len(rows)
        order = sorted(range(len(rows)), key=lambda i: len(rows[i][0]))
        start = 0
        while start < len(order):
            end = start + 1
            longest = len(rows[order[start]][0])
            while end < len(order):
                cand = len(rows[order[end]][0])
                if cand * (end - start + 1) > self.loglik_token_budget:
                    break
                longest = cand
                end += 1
            idx = order[start:end]
            pad = 0
            batch = [rows[i][0] + [pad] * (longest - len(rows[i][0])) for i in idx]
            tokens = mx.array(batch)
            logits = self.model(tokens[:, :-1]).astype(mx.float32)
            logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            tok_lp = mx.take_along_axis(logp, tokens[:, 1:, None], axis=-1).squeeze(-1)
            positions = mx.arange(longest - 1)[None, :]
            lens = mx.array([len(rows[i][0]) for i in idx])[:, None]
            starts = mx.array([rows[i][1] for i in idx])[:, None]
            # target token t (position t-1 in tok_lp) counts iff n_ctx <= t < len
            mask = (positions + 1 >= starts) & (positions + 1 < lens)
            sums = (tok_lp * mask).sum(axis=-1)
            mx.eval(sums)
            for j, i in enumerate(idx):
                out[i] = float(sums[j].item())  # type: ignore[arg-type]
            del logits, logp, tok_lp
            start = end
        mx.clear_cache()
        return out

    def count_tokens(self, text: str) -> int:
        return len(self._encode(text))

    def close(self) -> None:
        """Release this backend's reference; weights are freed when no backend uses them."""
        if self._handle is not None:
            handle, self._handle = self._handle, None
            _release(handle)


def _flat_params(model: Any) -> list[tuple[str, Any]]:
    from mlx.utils import tree_flatten

    return list(tree_flatten(model.parameters()))  # type: ignore[arg-type]


def _common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


# ----------------------------------------------------------------------------- test helpers


def write_random_lora_adapter(
    backend: MLXBackend,
    out_dir: Path,
    *,
    rank: int = 4,
    num_layers: int = 2,
    scale: float = 20.0,
    seed: int = 0,
    std: float = 0.02,
) -> Path:
    """Write a random (non-zero) mlx-lm LoRA adapter for ``backend``'s model to ``out_dir``.

    For tests and smoke runs: it changes outputs measurably but carries no knowledge. The
    base model is restored afterwards.
    """
    import mlx.core as mx
    from mlx_lm.tuner.utils import linear_to_lora_layers

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lora_params = {"rank": rank, "scale": scale, "dropout": 0.0}
    with backend.handle.lock:
        _set_adapter(backend.handle, None)
        model = backend.model
        try:
            linear_to_lora_layers(model, num_layers, lora_params)
            mx.random.seed(seed)
            weights = {
                k: mx.random.normal(v.shape, dtype=mx.float32) * std
                for k, v in _flat_params(model)
                if k.rsplit(".", 1)[-1] in ("lora_a", "lora_b")
            }
            mx.save_safetensors(str(out_dir / ADAPTER_WEIGHTS), weights)
        finally:
            _remove_lora(model)
            model.eval()
    config = {
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "lora_parameters": lora_params,
        "model": backend.model_id,
    }
    (out_dir / ADAPTER_CONFIG).write_text(json.dumps(config, indent=2), encoding="utf-8")
    return out_dir
