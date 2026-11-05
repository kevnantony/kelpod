"""Versioned Jinja2 prompt templates for the teacher (``prompts/*.j2``).

Every teacher prompt is a pair ``<name>.system.j2`` (static instructions, shared by all
calls so backends can prefix-cache it) and ``<name>.user.j2`` (starts with the passage).
The sha256 of every template file is written to ``prompts_manifest.json`` so a run can be
traced to the exact prompt wording.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from zllf.core.hashing import sha256_file
from zllf.core.schemas import Message

PROMPT_DIR = Path(__file__).parent / "prompts"


@lru_cache(maxsize=1)
def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(PROMPT_DIR)),
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=False,
    )


def render(template: str, **variables: Any) -> str:
    """Render ``prompts/<template>`` and strip surrounding whitespace."""
    return _env().get_template(template).render(**variables).strip()


def teacher_chat(name: str, **variables: Any) -> list[Message]:
    """System + user chat for the teacher prompt ``name`` (e.g. ``"quotes"``)."""
    return [
        Message(role="system", content=render(f"{name}.system.j2", **variables)),
        Message(role="user", content=render(f"{name}.user.j2", **variables)),
    ]


def prompts_manifest() -> dict[str, str]:
    """Template file name -> sha256 of its contents (sorted)."""
    return {p.name: sha256_file(p) for p in sorted(PROMPT_DIR.glob("*.j2"))}
