"""n-best readings from a CTC recogniser (RapidOCR / PP-OCR).

RapidOCR decodes the recogniser output greedily (arg-max per frame) and returns one string. The
softmax matrix behind it holds more: a ``1`` that was nearly a ``7``, a comma that was nearly a
dot or nothing. :func:`ctc_nbest` runs a CTC prefix beam search over that matrix and returns the
``k`` most probable labellings, so the dimension-chain decoder (:mod:`kelpod.ocr.chains`) can pick
the reading that fits its chain instead of the one the recogniser liked best in isolation.

:class:`NBestDecoder` wraps RapidOCR's ``CTCLabelDecode`` so the n-best list rides along with every
result row (``[box, text, score, alternatives]``) without a second recogniser pass.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

import numpy as np

_DIGIT = re.compile(r"[0-9]")


def _logsumexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def ctc_nbest(
    probs: np.ndarray,
    charset: list[str],
    *,
    k: int = 5,
    beam: int = 12,
    prune: float = 2e-3,
    blank: int = 0,
) -> list[tuple[str, float]]:
    """CTC prefix beam search over ``probs`` (frames x classes, softmax rows).

    Returns up to ``k`` ``(text, probability)`` pairs, most probable first; the probability is the
    sum over all alignments of that labelling. Classes below ``prune`` in a frame are skipped
    (the frame's best class is always kept), which keeps the search to a few classes per frame.
    """
    T = probs.shape[0]
    # prefix (tuple of class ids) -> (log p ending in blank, log p ending in non-blank)
    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, -math.inf)}
    for t in range(T):
        row = probs[t]
        cand = np.flatnonzero(row >= prune)
        best = int(row.argmax())
        if best not in cand:
            cand = np.append(cand, best)
        logp = {int(c): math.log(max(float(row[c]), 1e-12)) for c in cand}
        if blank not in logp:
            logp[blank] = math.log(max(float(row[blank]), 1e-12))
        nxt: dict[tuple[int, ...], list[float]] = defaultdict(lambda: [-math.inf, -math.inf])
        for prefix, (pb, pnb) in beams.items():
            total = _logsumexp(pb, pnb)
            # stay on the same prefix via blank
            e = nxt[prefix]
            e[0] = _logsumexp(e[0], total + logp[blank])
            last = prefix[-1] if prefix else None
            for c, lp in logp.items():
                if c == blank:
                    continue
                if c == last:
                    # repeat without blank in between collapses; after a blank it extends
                    e[1] = _logsumexp(e[1], pnb + lp)
                    ext = nxt[prefix + (c,)]
                    ext[1] = _logsumexp(ext[1], pb + lp)
                else:
                    ext = nxt[prefix + (c,)]
                    ext[1] = _logsumexp(ext[1], total + lp)
        ranked = sorted(nxt.items(), key=lambda kv: -_logsumexp(kv[1][0], kv[1][1]))
        beams = {p: (v[0], v[1]) for p, v in ranked[:beam]}
    out = []
    for prefix, (pb, pnb) in sorted(beams.items(), key=lambda kv: -_logsumexp(*kv[1])):
        text = "".join(charset[c] for c in prefix if 0 <= c < len(charset))
        out.append((text, math.exp(_logsumexp(pb, pnb))))
    # different class ids can map to the same string (e.g. two space classes): merge
    merged: dict[str, float] = {}
    for text, p in out:
        merged[text] = merged.get(text, 0.0) + p
    return sorted(merged.items(), key=lambda kv: -kv[1])[:k]


class NBestDecoder:
    """Drop-in wrapper for RapidOCR's ``CTCLabelDecode``: same greedy result, plus an n-best
    list as an extra tuple element for strings that contain a digit (dimension candidates)."""

    def __init__(self, base, *, k: int = 5) -> None:
        self.base = base
        self.k = k
        self.character = base.character

    def __getattr__(self, name):  # anything else RapidOCR asks for
        return getattr(self.base, name)

    def __call__(self, preds: np.ndarray, return_word_box: bool = False, **kwargs):
        res = self.base(preds, return_word_box, **kwargs)
        out = []
        for i, r in enumerate(res):
            text = str(r[0])
            alts: list[tuple[str, float]] = []
            if _DIGIT.search(text) and len(text) <= 16:
                alts = ctc_nbest(np.asarray(preds[i], np.float64), self.character, k=self.k)
            out.append((*r, alts))
        return out


__all__ = ["NBestDecoder", "ctc_nbest"]
