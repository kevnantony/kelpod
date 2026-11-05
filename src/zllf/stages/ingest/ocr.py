"""Local OCR engines for scanned pages and images (FR-02 basic). Never downloads models.

Engines, in ``auto`` order:
1. ``rapidocr`` (ONNX, Python dep of Docling) — only if its model files are already on disk;
   RapidOCR would otherwise try to download them on first use.
2. ``tesseract`` CLI — if the binary is on ``PATH`` (system package, traineddata local).
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL.Image import Image

_TESS_LANG = {"en": "eng", "de": "deu", "fr": "fra", "es": "spa", "it": "ita", "nl": "nld"}


@dataclass(frozen=True)
class OcrEngine:
    name: str  # "rapidocr" | "tesseract"
    languages: tuple[str, ...]  # engine-specific codes in use


@functools.cache
def _rapidocr_models_present() -> bool:
    """RapidOCR package, its bundled ONNX det/rec models and onnxruntime are all present."""
    try:
        import importlib.util

        spec = importlib.util.find_spec("rapidocr")
    except (ImportError, ValueError):
        return False
    if spec is None or not spec.submodule_search_locations:
        return False
    if importlib.util.find_spec("onnxruntime") is None:  # rapidocr's default inference engine
        return False
    pkg = Path(next(iter(spec.submodule_search_locations)))
    onnx = [p.name.lower() for p in (pkg / "models").glob("*.onnx")] if (pkg / "models").is_dir() \
        else []
    has_det = any("det" in n for n in onnx)
    has_rec = any("rec" in n for n in onnx)
    return has_det and has_rec


@functools.cache
def _tesseract_languages() -> tuple[str, ...]:
    exe = shutil.which("tesseract")
    if exe is None:
        return ()
    try:
        proc = subprocess.run([exe, "--list-langs"], capture_output=True, text=True, timeout=20,
                              check=False)
    except (OSError, subprocess.SubprocessError):
        return ()
    lines = (proc.stdout or proc.stderr).splitlines()
    return tuple(ln.strip() for ln in lines[1:] if ln.strip() and " " not in ln.strip())


def resolve_engine(choice: str, languages: tuple[str, ...]) -> OcrEngine | None:
    """Pick an available OCR engine or return ``None`` (OCR impossible offline)."""
    if choice in ("auto", "rapidocr") and _rapidocr_models_present():
        return OcrEngine("rapidocr", ("ch_en",))
    if choice in ("auto", "tesseract"):
        installed = _tesseract_languages()
        wanted = tuple(_TESS_LANG[lang] for lang in languages if _TESS_LANG.get(lang) in installed)
        if not wanted and "eng" in installed:
            wanted = ("eng",)
        if wanted:
            return OcrEngine("tesseract", wanted)
    return None


def engine_status(languages: tuple[str, ...]) -> dict[str, Any]:
    """Diagnostic info for the ingest report (no document data)."""
    installed = _tesseract_languages()
    return {
        "rapidocr_usable": _rapidocr_models_present(),
        "tesseract_languages": list(installed),
        "missing_tesseract_languages": [
            _TESS_LANG[lang] for lang in languages
            if lang in _TESS_LANG and _TESS_LANG[lang] not in installed
        ] if installed else [],
    }


@functools.cache
def _rapidocr() -> Any:
    from rapidocr import RapidOCR

    return RapidOCR()


def _ocr_rapid(image: Image) -> str:
    import numpy as np

    result = _rapidocr()(np.asarray(image.convert("RGB")))
    txts = list(getattr(result, "txts", None) or [])
    boxes = getattr(result, "boxes", None)
    if boxes is None or not txts:
        return "\n".join(txts)
    # reading order: top-to-bottom, then left-to-right; merge boxes on the same line
    items = sorted(
        ((float(min(p[1] for p in b)), float(min(p[0] for p in b)),
          float(max(p[1] for p in b)) - float(min(p[1] for p in b)), t)
         for b, t in zip(boxes, txts, strict=False)),
    )
    lines: list[list[tuple[float, str]]] = []
    last_y, last_h = -1e9, 0.0
    for y, x, h, t in items:
        if lines and abs(y - last_y) < 0.5 * max(h, last_h):
            lines[-1].append((x, t))
        else:
            lines.append([(x, t)])
            last_y, last_h = y, h
    return "\n".join(" ".join(t for _, t in sorted(line)) for line in lines)


def _ocr_tesseract(image: Image, languages: tuple[str, ...], timeout_s: float) -> str:
    exe = shutil.which("tesseract")
    if exe is None:
        raise RuntimeError("tesseract not found")
    with tempfile.TemporaryDirectory(prefix="zllf-ocr-") as tmp:
        src = Path(tmp) / "page.png"
        image.convert("L").save(src)
        env = {**os.environ, "OMP_THREAD_LIMIT": "1"}
        proc = subprocess.run(
            [exe, str(src), "stdout", "-l", "+".join(languages), "--psm", "3"],
            capture_output=True, text=True, timeout=timeout_s, check=False, env=env,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"tesseract exit code {proc.returncode}")
    return proc.stdout


def ocr_image(image: Image, engine: OcrEngine, timeout_s: float = 120.0) -> str:
    """Run OCR on one page image; returns raw text (normalise afterwards)."""
    if engine.name == "rapidocr":
        return _ocr_rapid(image)
    return _ocr_tesseract(image, engine.languages, timeout_s)
