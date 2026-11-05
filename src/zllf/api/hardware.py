"""Hardware profile: the train stage's profiler when present, else a minimal local probe."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

from zllf.core.config import load_config
from zllf.core.schemas import HardwareProfile

GB = 1024**3


def hardware_profile(hardware: str, disk_path: Path, config_dir: Path | None = None
                     ) -> HardwareProfile:
    config = (load_config(hardware=hardware, config_dir=config_dir) if config_dir
              else load_config(hardware=hardware))
    try:
        from zllf.stages.train.hardware import profile_hardware

        return profile_hardware(config, disk_path=disk_path)
    except ImportError:
        return _probe(config.models.backend, disk_path)


def _sysctl(name: str) -> str:
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True,
                              timeout=5, check=False).stdout.strip()
    except OSError:
        return ""


def _probe(backend: str, disk_path: Path) -> HardwareProfile:
    system, machine = platform.system().lower(), platform.machine().lower()
    total = 0
    if system == "darwin":
        total = int(_sysctl("hw.memsize") or 0)
        device = _sysctl("machdep.cpu.brand_string") or machine
    else:
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError):
            total = 0
        device = platform.processor() or machine
    disk_path.mkdir(parents=True, exist_ok=True)
    return HardwareProfile(
        platform=f"{system}-{machine}",
        accelerator="mps" if system == "darwin" and machine == "arm64" else "cpu",
        device_name=device,
        memory_total_gb=round(total / GB, 2),
        memory_available_gb=round(total / GB / 2, 2),
        cpu_cores=os.cpu_count() or 1,
        disk_free_gb=round(shutil.disk_usage(disk_path).free / GB, 2),
        backend=backend,
    )
