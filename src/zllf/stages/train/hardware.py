"""Hardware profiler (FR-06) and the calibrated training-memory model (P2.1.2).

The profiler is cheap and side-effect free: it shells out to ``sysctl``/``vm_stat`` on
macOS, reads ``/proc/meminfo`` on Linux, and asks ``torch.cuda`` / NVML only if they are
already importable. It never imports MLX in the controller process except to read the
Metal device info (no model is loaded).
"""

from __future__ import annotations

import math
import os
import platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from zllf.core.config import RunConfig
from zllf.core.schemas import HardwareProfile

GB = 1024**3

# Hidden size / layer counts are only needed to estimate LoRA parameter counts before the
# dry run; the dry run then calibrates the activation term from a measured peak.
_ATTN_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_KEYS = ("gate_proj", "up_proj", "down_proj")


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _sysctl(name: str) -> str:
    return _run(["sysctl", "-n", name]).strip()


def _darwin_available_bytes() -> int | None:
    """free + inactive + speculative + purgeable pages (what macOS can hand out quickly)."""
    out = _run(["vm_stat"])
    if not out:
        return None
    m = re.search(r"page size of (\d+) bytes", out)
    page = int(m.group(1)) if m else 16384
    pages = 0
    for key in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
        km = re.search(rf"{key}:\s+(\d+)", out)
        if km:
            pages += int(km.group(1))
    return pages * page


def _linux_meminfo() -> tuple[int | None, int | None]:
    try:
        text = Path("/proc/meminfo").read_text()
    except OSError:
        return None, None
    vals: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            vals[parts[0].rstrip(":")] = int(parts[1]) * 1024
    return vals.get("MemTotal"), vals.get("MemAvailable")


def _metal_info() -> dict[str, Any]:
    """Metal device info via MLX (no model load). Empty dict when MLX is unavailable."""
    try:
        import mlx.core as mx

        fn = getattr(mx, "device_info", None) or mx.metal.device_info
        return dict(fn())
    except Exception:
        return {}


def _cuda_info() -> dict[str, Any] | None:
    """Best-effort CUDA probe (torch first, then NVML). None when no CUDA device."""
    try:
        import torch

        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            free, total = torch.cuda.mem_get_info(0)
            major, _minor = torch.cuda.get_device_capability(0)
            return {
                "name": torch.cuda.get_device_name(0),
                "n": n,
                "total": total,
                "free": free,
                "bf16": bool(torch.cuda.is_bf16_supported()),
                "fp8": major >= 9,
            }
    except Exception:
        pass
    try:
        import pynvml

        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        if n == 0:
            return None
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        name = pynvml.nvmlDeviceGetName(h)
        major, _ = pynvml.nvmlDeviceGetCudaComputeCapability(h)
        return {
            "name": name.decode() if isinstance(name, bytes) else str(name),
            "n": n,
            "total": int(mem.total),
            "free": int(mem.free),
            "bf16": major >= 8,
            "fp8": major >= 9,
        }
    except Exception:
        return None


def profile_hardware(config: RunConfig, disk_path: Path | None = None) -> HardwareProfile:
    """Detect the machine and choose backend + quantisation for this run."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    cpu_cores = os.cpu_count() or 1
    disk_free = shutil.disk_usage(disk_path or Path.cwd()).free

    accelerator: str = "cpu"
    device_name = platform.processor() or machine
    n_devices = 1
    bf16 = False
    fp8 = False
    if system == "darwin":
        device_name = _sysctl("machdep.cpu.brand_string") or device_name
        total = int(_sysctl("hw.memsize") or 0)
        avail = _darwin_available_bytes() or total // 2
        if machine == "arm64":
            accelerator = "mps"
            bf16 = True  # Apple GPUs from M2 on support bf16; MLX emulates it otherwise
        plat = f"darwin-{machine}"
    else:
        total_l, avail_l = _linux_meminfo()
        total = total_l or 0
        avail = avail_l or total // 2
        cuda = _cuda_info()
        plat = f"{system}-{machine}"
        if cuda:
            accelerator = "cuda"
            device_name = cuda["name"]
            n_devices = cuda["n"]
            total, avail = cuda["total"], cuda["free"]
            bf16, fp8 = cuda["bf16"], cuda["fp8"]
            plat = f"{system}-cuda"

    backend = config.models.backend
    model = config.models.student.lower()
    if "4bit" in model or "q4" in model:
        quant = "4bit"
    elif "8bit" in model or "q8" in model:
        quant = "8bit"
    elif backend == "mlx" or (accelerator == "cuda" and total < 40 * GB):
        quant = "4bit"  # QLoRA-style when memory is tight
    else:
        quant = "none" if accelerator == "cuda" else "4bit"
    return HardwareProfile(
        platform=plat,
        accelerator=accelerator,  # type: ignore[arg-type]
        device_name=device_name,
        n_devices=n_devices,
        memory_total_gb=round(total / GB, 2),
        memory_available_gb=round(avail / GB, 2),
        cpu_cores=cpu_cores,
        disk_free_gb=round(disk_free / GB, 2),
        bf16=bf16,
        fp8=fp8,
        backend=backend,
        quantization=quant,  # type: ignore[arg-type]
    )


def metal_guidance(profile: HardwareProfile) -> dict[str, Any]:
    """Recommended MLX memory limits. MLX defaults its limit to 1.5x the recommended working
    set; we cap it at the available memory so the OS (and other jobs) are not starved."""
    info = _metal_info() if profile.accelerator == "mps" else {}
    rec = int(info.get("max_recommended_working_set_size", 0) or 0)
    avail = int(profile.memory_available_gb * GB)
    limit = min(x for x in (rec or avail, avail) if x > 0) if (rec or avail) else 0
    return {
        "max_recommended_working_set_gb": round(rec / GB, 2) if rec else None,
        "memory_limit_gb": round(limit / GB, 2) if limit else None,
        "device_info": {k: v for k, v in info.items() if isinstance(v, (int, str))},
    }


# ---------------------------------------------------------------- memory model
@dataclass
class ModelShape:
    """Coarse architecture facts needed for LoRA parameter counts."""

    hidden_size: int = 896
    intermediate_size: int = 4864
    num_layers: int = 24
    kv_dim: int = 128  # num_key_value_heads * head_dim

    @classmethod
    def from_model_config(cls, cfg: dict[str, Any]) -> ModelShape:
        h = int(cfg.get("hidden_size", 896))
        heads = int(cfg.get("num_attention_heads", 14))
        kv_heads = int(cfg.get("num_key_value_heads", heads))
        head_dim = int(cfg.get("head_dim", h // max(heads, 1)))
        return cls(
            hidden_size=h,
            intermediate_size=int(cfg.get("intermediate_size", 4 * h)),
            num_layers=int(cfg.get("num_hidden_layers", 24)),
            kv_dim=kv_heads * head_dim,
        )

    def lora_params_per_rank(self, target_modules: str, num_layers: int) -> int:
        """LoRA parameters per unit rank: sum over adapted linears of (in + out)."""
        h, i, kv = self.hidden_size, self.intermediate_size, self.kv_dim
        attn = (h + h) + (h + kv) + (h + kv) + (h + h)
        mlp = (h + i) + (h + i) + (i + h)
        per_layer = {"attention": attn, "mlp": mlp, "all": attn + mlp}[target_modules]
        layers = self.num_layers if num_layers < 0 else min(num_layers, self.num_layers)
        return per_layer * layers


def find_model_config(model_id: str) -> dict[str, Any]:
    """Read ``config.json`` of a model from a local dir or the local HF cache (no network)."""
    import json

    p = Path(model_id)
    candidates = [p / "config.json"] if p.exists() else []
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
    snap_root = hub / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if snap_root.exists():
        candidates += sorted(snap_root.glob("*/config.json"))
    for c in candidates:
        try:
            data: dict[str, Any] = json.loads(c.read_text())
            return data
        except (OSError, ValueError):
            continue
    return {}


@dataclass
class MemoryModel:
    """peak(r, b, L, M) ≈ base + act_per_token · b · L · ckpt_factor + lora_bytes(r, M).

    ``base`` is the resident model (+ allocator slack) and ``act_per_token`` the activation
    bytes per token, both calibrated from one dry-run measurement at max sequence length.
    LoRA bytes = params · 16 (fp32 weight + grad + two Adam moments).
    """

    base_bytes: float
    act_bytes_per_token: float
    shape: ModelShape
    num_layers: int = -1
    ckpt_factor: float = 0.35  # activation fraction kept with gradient checkpointing
    lora_bytes_per_param: float = 16.0

    def predict(
        self, rank: int, batch_size: int, seq_len: int, target_modules: str,
        grad_checkpoint: bool = False,
    ) -> float:
        act = self.act_bytes_per_token * batch_size * seq_len
        if grad_checkpoint:
            act *= self.ckpt_factor
        lora = self.shape.lora_params_per_rank(target_modules, self.num_layers) * rank
        return self.base_bytes + act + lora * self.lora_bytes_per_param

    @classmethod
    def calibrate(
        cls, *, base_bytes: float, peak_bytes: float, rank: int, batch_size: int,
        seq_len: int, target_modules: str, shape: ModelShape, num_layers: int = -1,
    ) -> MemoryModel:
        lora = shape.lora_params_per_rank(target_modules, num_layers) * rank * 16.0
        act = max(peak_bytes - base_bytes - lora, 0.0) / max(batch_size * seq_len, 1)
        return cls(base_bytes=base_bytes, act_bytes_per_token=act, shape=shape,
                   num_layers=num_layers)

    @classmethod
    def heuristic(cls, model_bytes: float, shape: ModelShape, num_layers: int = -1) -> MemoryModel:
        """Uncalibrated prior: ~ 2·hidden·layers·(bytes/activation≈2)·12 per token."""
        act = 24.0 * shape.hidden_size * shape.num_layers * 2.0
        return cls(base_bytes=model_bytes * 1.2, act_bytes_per_token=act, shape=shape,
                   num_layers=num_layers)

    def fit_batch(
        self, *, rank: int, batch_size: int, grad_accumulation: int, seq_len: int,
        target_modules: str, limit_bytes: float,
    ) -> tuple[int, int, bool] | None:
        """Largest micro-batch (halving, accumulation doubled) that fits ``limit_bytes``;
        enables gradient checkpointing as last resort. None if nothing fits."""
        b, acc = batch_size, grad_accumulation
        while True:
            if self.predict(rank, b, seq_len, target_modules) <= limit_bytes:
                return b, acc, False
            if b == 1:
                break
            b, acc = max(1, b // 2), acc * 2
        if self.predict(rank, 1, seq_len, target_modules, grad_checkpoint=True) <= limit_bytes:
            return 1, acc, True
        return None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["base_gb"] = round(self.base_bytes / GB, 3)
        d["act_mb_per_token"] = round(self.act_bytes_per_token / 1024**2, 4)
        return d


def gb(x: float) -> float:
    return round(x / GB, 3) if math.isfinite(x) else x
