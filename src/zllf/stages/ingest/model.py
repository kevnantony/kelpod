"""Intermediate parse result shared by all loaders (picklable, crosses process boundaries)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

BlockType = Literal["heading", "paragraph", "list", "table", "caption", "other"]


@dataclass
class ParsedBlock:
    """One structural block; converted to :class:`zllf.core.schemas.Block` by the stage."""

    type: BlockType
    text: str
    page: int | None = None
    level: int | None = None
    table_markdown: str | None = None


@dataclass
class ParsedDocument:
    blocks: list[ParsedBlock] = field(default_factory=list)
    n_pages: int = 0
    n_ocr_pages: int = 0
    parser: str = ""
    # machine-readable warning codes only (never document text)
    warnings: list[str] = field(default_factory=list)

    @property
    def n_chars(self) -> int:
        return sum(len(b.text) for b in self.blocks)


@dataclass
class ParseOutcome:
    """Result of parsing one file in a worker."""

    rel_path: str
    status: Literal["parsed", "failed"]
    doc: ParsedDocument | None = None
    error: str | None = None  # "<ErrorCode>: <reason>" — reason never contains document text
    seconds: float = 0.0
