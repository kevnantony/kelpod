#!/usr/bin/env python3
"""Collect real floor-plan images (German, DIN 1356-1 style) as test data.

Sources (official APIs only; no HTML scraping, no CAPTCHA workarounds):
  openverse  https://api.openverse.org/v1/images/   (no key)
  commons    Wikimedia Commons Action API           (no key)
  google     Programmable Search JSON API           (only with GOOGLE_CSE_KEY + GOOGLE_CSE_CX;
                                                     100 free queries/day, counted and enforced)

Usage:
  uv run python scripts/collect_plans.py --out DIR --max-per-query N \
      [--sources openverse,commons,google] [--queries FILE]

Output layout (default DIR = ~/kelpod-data/web, never inside the repo):
  DIR/<source>/<sha16>.<ext>    images that passed the plan filter (rejected ones are only logged)
  DIR/manifest.jsonl            one JSON line per accepted image
  DIR/rejected.jsonl            one JSON line per rejected image
  DIR/seen_urls.txt             every image URL already tried (resume)
  DIR/google_quota.json         per-day Google query counter

Plan filter (cheap heuristics, thresholds are the constants below):
  1. min side >= MIN_SIDE px
  2. >= MIN_WHITE_FRAC near-white pixels (all channels >= WHITE_LEVEL)
  3. mean HSV saturation < MAX_MEAN_SAT (line drawing, not a photo)
  4. line structure: of all dark pixels, >= MIN_LINE_RATIO lie in long horizontal or vertical
     runs (morphological opening with a 1xK / Kx1 element, K = LINE_RUN_FRAC * analysis size),
     and the dark fraction is within [MIN_DARK_FRAC, MAX_DARK_FRAC].
Resume-safe: image URLs, sha256 and dHashes already in the manifests are skipped.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

USER_AGENT = "kelpod-collector/0.1 (research; contact via GitHub repo)"

# ---- politeness / limits
MIN_INTERVAL_S = 1.0  # between requests per host
HOST_INTERVAL_S = {"upload.wikimedia.org": 4.0}  # Wikimedia answers 429 to faster original-file downloads
TIMEOUT_S = 15
RETRIES = 3  # attempts, i.e. at most 2 retries
BREAKER_N = 3  # consecutive failed API calls before a source is skipped for the rest of the run
BACKOFF_S = 2.0
MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 60_000_000
GOOGLE_DAILY_QUOTA = 100
MAX_PAGES = 5  # result pages per (query, source)

# ---- plan filter thresholds
MIN_SIDE = 400
MIN_WHITE_FRAC = 0.55
WHITE_LEVEL = 230
MAX_MEAN_SAT = 0.15
ANALYSIS_SIDE = 1200  # longest side the analysis is run at
DARK_LEVEL = 140  # gray below this counts as ink
LINE_RUN_FRAC = 0.04  # opening length as fraction of analysis longest side
MIN_LINE_RATIO = 0.30
MIN_DARK_FRAC = 0.003
MAX_DARK_FRAC = 0.30
DHASH_MAX_DIST = 4
MIN_DIR_FRAC = 0.25  # each of horizontal / vertical must hold this share of the line pixels

ALLOWED_EXT = {"jpeg": "jpg", "jpg": "jpg", "png": "png", "gif": "gif", "webp": "webp", "bmp": "bmp"}

DEFAULT_QUERIES = [
    "Grundriss Erdgeschoss Bemaßung",
    "Bestandsplan Grundriss",
    "Grundriss Obergeschoss Maße",
    "Bauzeichnung Grundriss",
    "Grundriss DIN 1356",
    "Bauantrag Grundriss Erdgeschoss",
    "Grundriss Einfamilienhaus Maße",
    "Grundriss Obergeschoss Bemaßung",
    "Werkplan Grundriss",
    "Grundriss Wohnung Maßkette",
    "floor plan German dimensions Grundriss",
]


# ---------------------------------------------------------------- licences

def classify_licence(lic: str) -> str:
    """'ok' (CC0 / PD / CC BY / CC BY-SA), 'nc' (has NC or ND), 'unverified' or 'no'."""
    t = lic.lower().strip()
    if t.startswith("unverified"):
        return "unverified"
    tok = [x for x in re.split(r"[^a-z0-9]+", t) if x]
    if "nc" in tok or "nd" in tok:
        return "nc"
    if tok[:1] in (["cc0"], ["pdm"], ["pd"]) or t.startswith("public domain"):
        return "ok"
    if tok[:2] == ["cc", "by"] or tok[:1] == ["by"]:
        return "ok"  # by, by-sa, cc by 4.0, cc by-sa 3.0 (anything else after BY was caught above)
    return "no"


# ---------------------------------------------------------------- data

@dataclass
class Candidate:
    source: str
    query: str
    image_url: str
    page_url: str = ""
    title: str = ""
    licence: str = "unknown"
    licence_url: str = ""
    author: str = ""
    width: int = 0
    height: int = 0
    size: int = 0  # bytes if the API says so, else 0
    mime: str = ""


# ---------------------------------------------------------------- HTTP

class Http:
    """Polite urllib client: UA, timeouts, retries with backoff, >= 1 s between requests per host."""

    def __init__(self, min_interval: float = MIN_INTERVAL_S, sleep=time.sleep, clock=time.monotonic):
        self.min_interval = min_interval
        self._sleep, self._clock = sleep, clock
        self._last: dict[str, float] = {}
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}

    def _throttle(self, url: str) -> None:
        host = urllib.parse.urlsplit(url).netloc
        wait = self._last.get(host, -1e9) + max(self.min_interval, HOST_INTERVAL_S.get(host, 0)) - self._clock()
        if wait > 0:
            self._sleep(wait)
        self._last[host] = self._clock()

    def get(self, url: str, params: dict | None = None, max_bytes: int | None = None,
            headers: dict | None = None) -> bytes:
        if params:
            url = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
        last_err: Exception | None = None
        for attempt in range(RETRIES):
            self._throttle(url)
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
            try:
                with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                    if max_bytes is not None:
                        cl = r.headers.get("Content-Length")
                        if cl and cl.isdigit() and int(cl) > max_bytes:
                            raise TooLarge(url)
                        data = r.read(max_bytes + 1)
                        if len(data) > max_bytes:
                            raise TooLarge(url)
                        return data
                    return r.read()
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code not in (429, 500, 502, 503, 504):
                    raise
                ra = e.headers.get("Retry-After") if e.headers else None
                delay = min(float(ra), 15) if ra and ra.isdigit() else max(BACKOFF_S * 2 ** attempt, 10.0 if e.code == 429 else 0)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last_err = e
                delay = BACKOFF_S * 2 ** attempt
            if attempt < RETRIES - 1:
                self._sleep(delay)
        assert last_err is not None
        raise last_err

    def get_json(self, url: str, params: dict | None = None) -> dict:
        return json.loads(self.get(url, params).decode("utf-8"))

    def robots_allowed(self, url: str) -> bool:
        parts = urllib.parse.urlsplit(url)
        base = f"{parts.scheme}://{parts.netloc}"
        if base not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                rp.parse(self.get(base + "/robots.txt", max_bytes=1_000_000).decode("utf-8", "replace").splitlines())
                self._robots[base] = rp
            except urllib.error.HTTPError as e:
                self._robots[base] = None if e.code < 500 else _DENY  # 4xx: no robots.txt -> allowed
            except Exception:
                self._robots[base] = _DENY  # unreachable robots.txt: be conservative
        rp = self._robots[base]
        if rp is _DENY:
            return False
        return True if rp is None else rp.can_fetch(USER_AGENT, url)


class TooLarge(Exception):
    pass


_DENY = urllib.robotparser.RobotFileParser()  # sentinel


# ---------------------------------------------------------------- source parsers

def _int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _ext_ok(mime: str, url: str) -> bool:
    if mime:
        return mime.lower().split("/")[-1] in ALLOWED_EXT
    return url.lower().rsplit(".", 1)[-1].split("?")[0] in ALLOWED_EXT


def _strip_html(s: str) -> str:
    out, intag = [], False
    for ch in s or "":
        if ch == "<":
            intag = True
        elif ch == ">":
            intag = False
        elif not intag:
            out.append(ch)
    return "".join(out).strip()


def parse_openverse(data: dict, query: str) -> list[Candidate]:
    out = []
    for r in data.get("results", []):
        url = r.get("url")
        if not url:
            continue
        lic = r.get("license") or ""
        ver = r.get("license_version") or ""
        out.append(Candidate(
            source="openverse", query=query, image_url=url,
            page_url=r.get("foreign_landing_url") or "", title=r.get("title") or "",
            licence=(f"{lic} {ver}".strip() or "unknown"), licence_url=r.get("license_url") or "",
            author=r.get("creator") or "", width=_int(r.get("width")), height=_int(r.get("height")),
            size=_int(r.get("filesize")), mime=(r.get("filetype") and "image/" + r["filetype"]) or "",
        ))
    return [c for c in out if _ext_ok(c.mime, c.image_url)]


def parse_commons(data: dict, query: str) -> list[Candidate]:
    out = []
    pages = (data.get("query") or {}).get("pages") or {}
    for p in sorted(pages.values(), key=lambda p: p.get("index", 0)):
        infos = p.get("imageinfo") or []
        if not infos:
            continue
        ii = infos[0]
        meta = ii.get("extmetadata") or {}

        def mv(k):
            return _strip_html((meta.get(k) or {}).get("value", ""))

        c = Candidate(
            source="commons", query=query, image_url=ii.get("url", ""),
            page_url=ii.get("descriptionurl", ""), title=p.get("title", "").removeprefix("File:"),
            licence=mv("LicenseShortName") or mv("License") or "unknown", licence_url=mv("LicenseUrl"),
            author=mv("Artist"), width=_int(ii.get("width")), height=_int(ii.get("height")),
            size=_int(ii.get("size")), mime=ii.get("mime", ""),
        )
        if c.image_url and _ext_ok(c.mime, c.image_url):
            out.append(c)
    return out


def parse_google(data: dict, query: str) -> list[Candidate]:
    out = []
    for it in data.get("items", []) or []:
        img = it.get("image") or {}
        url = it.get("link")
        if not url:
            continue
        c = Candidate(
            source="google", query=query, image_url=url, page_url=img.get("contextLink", ""),
            title=it.get("title", ""),
            licence="unverified (Google rights filter: cc_publicdomain|cc_attribute|cc_sharealike)", width=_int(img.get("width")),
            height=_int(img.get("height")), size=_int(img.get("byteSize")), mime=it.get("mime", ""),
        )
        if _ext_ok(c.mime, url):
            out.append(c)
    return out


# ---------------------------------------------------------------- source fetchers

OPENVERSE_URL = "https://api.openverse.org/v1/images/"
COMMONS_URL = "https://commons.wikimedia.org/w/api.php"
GOOGLE_URL = "https://www.googleapis.com/customsearch/v1"


def fetch_openverse(http, query: str, page: int) -> list[Candidate]:
    data = http.get_json(OPENVERSE_URL, {"q": query, "page": page, "page_size": 20})
    return parse_openverse(data, query)


def fetch_commons(http, query: str, page: int) -> list[Candidate]:
    params = {
        "action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
        "gsrsearch": query, "gsrlimit": 20, "gsroffset": 20 * (page - 1),
        "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata",
    }
    return parse_commons(http.get_json(COMMONS_URL, params), query)


class GoogleQuota:
    """Persisted per-day query counter (UTC date)."""

    def __init__(self, path: Path, limit: int = GOOGLE_DAILY_QUOTA, today=None):
        self.path, self.limit = path, limit
        self.today = today or (lambda: dt.datetime.now(dt.timezone.utc).date().isoformat())
        self.count = 0
        try:
            d = json.loads(path.read_text())
            if d.get("date") == self.today():
                self.count = int(d.get("count", 0))
        except (OSError, ValueError):
            pass

    def remaining(self) -> int:
        return max(0, self.limit - self.count)

    def use(self) -> bool:
        """Count one query; False (and no count) if the quota is exhausted."""
        if self.count >= self.limit:
            return False
        self.count += 1
        self.path.write_text(json.dumps({"date": self.today(), "count": self.count}))
        return True


def make_google_fetcher(key: str, cx: str, quota: GoogleQuota):
    def fetch(http, query: str, page: int) -> list[Candidate]:
        if not quota.use():
            raise QuotaExhausted()
        params = {"key": key, "cx": cx, "q": query, "searchType": "image", "num": 10,
                  "start": 1 + 10 * (page - 1), "lr": "lang_de",
                  "rights": "cc_publicdomain,cc_attribute,cc_sharealike"}
        return parse_google(http.get_json(GOOGLE_URL, params), query)
    return fetch


class QuotaExhausted(Exception):
    pass


# ---------------------------------------------------------------- image analysis

def dhash(img: Image.Image) -> int:
    g = np.asarray(img.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _long_runs(b: np.ndarray, k: int) -> np.ndarray:
    """Mask of pixels of boolean image b lying in horizontal runs of length >= k (opening 1xk)."""
    h, w = b.shape
    if w < k:
        return np.zeros_like(b)
    c = np.concatenate([np.zeros((h, 1), np.int32), np.cumsum(b, axis=1, dtype=np.int32)], axis=1)
    starts = (c[:, k:] - c[:, :-k]) == k  # (h, w-k+1)
    d = np.zeros((h, w + 1), np.int32)
    d[:, : w - k + 1] += starts
    d[:, k: w + 1] -= starts
    return np.cumsum(d, axis=1)[:, :w] > 0


def plan_scores(img: Image.Image) -> dict:
    """Heuristic scores used by the plan filter."""
    w, h = img.size
    scale = ANALYSIS_SIDE / max(w, h)
    if scale < 1:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba)
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    mx, mn = rgb.max(axis=2), rgb.min(axis=2)
    white = float((mn >= WHITE_LEVEL).mean())
    sat = float(np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0).mean())
    gray = rgb.mean(axis=2)
    dark = gray < DARK_LEVEL
    dark_frac = float(dark.mean())
    n_dark = int(dark.sum())
    k = max(12, int(LINE_RUN_FRAC * max(dark.shape)))
    if n_dark:
        hm, vm = _long_runs(dark, k) & dark, (_long_runs(dark.T, k).T) & dark
        n_h, n_v = int(hm.sum()), int(vm.sum())
        line_ratio = float((hm | vm).sum() / n_dark)
        tot = max(1, n_h + n_v)
        dir_balance = min(n_h, n_v) / tot
    else:
        line_ratio = dir_balance = 0.0
    return {"white_frac": round(white, 4), "mean_sat": round(sat, 4),
            "dark_frac": round(dark_frac, 4), "line_ratio": round(line_ratio, 4),
            "dir_balance": round(dir_balance, 4)}


def plan_filter(img: Image.Image) -> tuple[bool, dict, str]:
    """Return (passed, scores, reason). reason is '' when passed."""
    scores = {"width": img.size[0], "height": img.size[1]}
    if min(img.size) < MIN_SIDE:
        return False, scores, f"too_small(min side {min(img.size)} < {MIN_SIDE})"
    scores.update(plan_scores(img))
    if scores["white_frac"] < MIN_WHITE_FRAC:
        return False, scores, f"not_white_enough({scores['white_frac']} < {MIN_WHITE_FRAC})"
    if scores["mean_sat"] >= MAX_MEAN_SAT:
        return False, scores, f"too_colourful({scores['mean_sat']} >= {MAX_MEAN_SAT})"
    if not MIN_DARK_FRAC <= scores["dark_frac"] <= MAX_DARK_FRAC:
        return False, scores, f"dark_fraction_out_of_range({scores['dark_frac']})"
    if scores["line_ratio"] < MIN_LINE_RATIO:
        return False, scores, f"few_straight_lines({scores['line_ratio']} < {MIN_LINE_RATIO})"
    if scores["dir_balance"] < MIN_DIR_FRAC:
        return False, scores, f"not_two_orthogonal_directions({scores['dir_balance']} < {MIN_DIR_FRAC})"
    scores["filter_score"] = round(min(1.0, scores["line_ratio"]) * 0.5 + scores["dir_balance"], 4)
    return True, scores, ""


# ---------------------------------------------------------------- collector

@dataclass
class Collector:
    out: Path
    http: object
    max_new: int
    allow_nc: bool = False
    max_per_query: int = 0  # 0 = unlimited
    ref_dirs: tuple = ()
    licence_skipped: dict = field(default_factory=dict)
    query_count: dict = field(default_factory=dict)
    log: callable = print
    seen_urls: set = field(default_factory=set)
    seen_sha: set = field(default_factory=set)
    hashes: list = field(default_factory=list)
    accepted: dict = field(default_factory=dict)  # source -> count (this run)
    rejected: dict = field(default_factory=dict)
    fetched: dict = field(default_factory=dict)
    total_accepted: int = 0

    def __post_init__(self):
        self.out.mkdir(parents=True, exist_ok=True)
        self.load_reference(self.ref_dirs)
        self.manifest = self.out / "manifest.jsonl"
        self.rej_manifest = self.out / "rejected.jsonl"
        seen_file = self.out / "seen_urls.txt"
        if seen_file.exists():
            self.seen_urls.update(seen_file.read_text().splitlines())
        for path, is_acc in ((self.manifest, True), (self.rej_manifest, False)):
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                self.seen_urls.add(r.get("image_url"))
                self.seen_sha.add(r.get("sha256"))
                if is_acc and r.get("dhash") is not None:
                    self.hashes.append(int(r["dhash"], 16) if isinstance(r["dhash"], str) else r["dhash"])
                if is_acc:
                    self.total_accepted += 1

    def load_reference(self, dirs) -> None:
        """Hash images already collected elsewhere so they are not collected again."""
        for d in dirs:
            for f in sorted(Path(d).expanduser().rglob("*")):
                if not f.is_file() or f.suffix.lower().lstrip(".") not in ALLOWED_EXT:
                    continue
                try:
                    data = f.read_bytes()
                    self.seen_sha.add(hashlib.sha256(data).hexdigest())
                    with Image.open(io.BytesIO(data)) as im:
                        if im.size[0] * im.size[1] > MAX_PIXELS:
                            continue
                        self.hashes.append(dhash(im))
                except Exception:
                    continue

    @property
    def done(self) -> bool:
        return sum(self.accepted.values()) >= self.max_new

    def is_dup_hash(self, h: int) -> bool:
        return any(hamming(h, o) <= DHASH_MAX_DIST for o in self.hashes)

    def process(self, c: Candidate) -> str:
        """Download, filter, store one candidate. Returns a status string."""
        if c.image_url in self.seen_urls:
            return "seen"
        self.seen_urls.add(c.image_url)
        with (self.out / "seen_urls.txt").open("a") as f:
            f.write(c.image_url + "\n")
        if c.size and c.size > MAX_BYTES:
            return "too_large"
        cls = classify_licence(c.licence)
        if cls == "no" or (cls == "nc" and not self.allow_nc):
            self.licence_skipped[c.licence] = self.licence_skipped.get(c.licence, 0) + 1
            return "licence"
        if not self.http.robots_allowed(c.image_url):
            return "robots"
        try:
            data = self.http.get(c.image_url, max_bytes=MAX_BYTES)
        except TooLarge:
            return "too_large"
        except Exception as e:  # network / HTTP error: skip this image
            self.log(f"  download failed {c.image_url}: {e}")
            return "error"
        self.fetched[c.source] = self.fetched.get(c.source, 0) + 1
        sha = hashlib.sha256(data).hexdigest()
        if sha in self.seen_sha:
            return "dup_sha"
        try:
            img = Image.open(io.BytesIO(data))
            fmt = (img.format or "").lower()
            if img.size[0] * img.size[1] > MAX_PIXELS:
                return "too_many_pixels"
            img.load()
        except Exception:
            return "not_an_image"
        ext = ALLOWED_EXT.get(fmt)
        if not ext:
            return "bad_format"
        self.seen_sha.add(sha)
        dh = dhash(img)
        if self.is_dup_hash(dh):
            return "dup_dhash"
        ok, scores, reason = plan_filter(img)
        rec = {
            "source": c.source, "query": c.query, "image_url": c.image_url, "page_url": c.page_url,
            "title": c.title, "licence": c.licence, "licence_url": c.licence_url, "author": c.author,
            "width": img.size[0], "height": img.size[1], "sha256": sha, "dhash": f"{dh:016x}",
            "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "filter": scores, "filter_score": scores.get("filter_score"),
            "licence_class": cls,
        }
        name = f"{sha[:16]}.{ext}"
        if ok:
            sdir = self.out / c.source
            sdir.mkdir(parents=True, exist_ok=True)
            (sdir / name).write_bytes(data)
            rec["file"] = f"{c.source}/{name}"
            with self.manifest.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.hashes.append(dh)
            self.accepted[c.source] = self.accepted.get(c.source, 0) + 1
            self.total_accepted += 1
            return "accepted"
        rec["reject_reason"] = reason  # rejected images are only logged, never stored
        with self.rej_manifest.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.rejected[c.source] = self.rejected.get(c.source, 0) + 1
        return "rejected"

    def run(self, fetchers: dict, queries: list[str], max_pages: int = MAX_PAGES,
            breaker_n: int = BREAKER_N) -> None:
        dead: set[str] = set()
        fails: dict[str, int] = {}
        for page in range(1, max_pages + 1):
            for q in queries:
                for name, fetch in fetchers.items():
                    if self.done:
                        return
                    if name in dead:
                        continue
                    try:
                        cands = fetch(self.http, q, page)
                    except QuotaExhausted:
                        self.log(f"[{name}] daily quota exhausted, stopping this source")
                        dead.add(name)
                        continue
                    except Exception as e:
                        fails[name] = fails.get(name, 0) + 1
                        self.log(f"[{name}] query failed ({q!r} p{page}): {e}")
                        if fails[name] >= breaker_n:
                            self.log(f"[{name}] circuit breaker: {breaker_n} consecutive failures, "
                                     "skipping this source for the rest of the run")
                            dead.add(name)
                        continue
                    fails[name] = 0
                    self.log(f"[{name}] {q!r} p{page}: {len(cands)} candidates")
                    for c in cands:
                        if self.done:
                            return
                        key = (name, q)
                        if self.max_per_query and self.query_count.get(key, 0) >= self.max_per_query:
                            break
                        if self.process(c) == "accepted":
                            self.query_count[key] = self.query_count.get(key, 0) + 1
            if len(dead) == len(fetchers):
                return


# ---------------------------------------------------------------- CLI

def load_queries(path: str | None) -> list[str]:
    if not path:
        return list(DEFAULT_QUERIES)
    qs = [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines()]
    return [q for q in qs if q and not q.startswith("#")]


def build_fetchers(sources: list[str], out: Path, env=os.environ) -> dict:
    fetchers = {}
    for s in sources:
        if s == "openverse":
            fetchers[s] = fetch_openverse
        elif s == "commons":
            fetchers[s] = fetch_commons
        elif s == "google":
            key, cx = env.get("GOOGLE_CSE_KEY"), env.get("GOOGLE_CSE_CX")
            if not (key and cx):
                print("[google] skipped: set GOOGLE_CSE_KEY and GOOGLE_CSE_CX to enable it")
                continue
            fetchers[s] = make_google_fetcher(key, cx, GoogleQuota(out / "google_quota.json"))
        else:
            print(f"unknown source {s!r}", file=sys.stderr)
    return fetchers


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(Path.home() / "kelpod-data" / "web"))
    ap.add_argument("--max", type=int, default=10**9, help="stop after N new accepted plan images in total")
    ap.add_argument("--max-per-query", type=int, default=50, help="accepted images per (source, query)")
    ap.add_argument("--allow-nc", action="store_true",
                    help="also keep NC/ND licences (educational repo only; marked licence_class=nc)")
    ap.add_argument("--ref-dirs", default=str(Path.home() / "kelpod-data/real"),
                    help="comma-separated dirs whose images count as already collected")
    ap.add_argument("--sources", default="openverse,commons,google")
    ap.add_argument("--queries", help="text file, one query per line")
    a = ap.parse_args(argv)
    out = Path(a.out).expanduser()
    repo = Path(__file__).resolve().parent.parent
    if repo in (out.resolve(), *out.resolve().parents):
        print("refusing to write images into the repo; choose --out outside it", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    fetchers = build_fetchers([s.strip() for s in a.sources.split(",") if s.strip()], out)
    if not fetchers:
        print("no usable sources", file=sys.stderr)
        return 1
    col = Collector(out=out, http=Http(), max_new=a.max, allow_nc=a.allow_nc,
                    max_per_query=a.max_per_query,
                    ref_dirs=tuple(d for d in a.ref_dirs.split(",") if d))
    col.run(fetchers, load_queries(a.queries))
    print("\n== summary ==")
    for s in fetchers:
        print(f"{s}: downloaded {col.fetched.get(s, 0)}, accepted {col.accepted.get(s, 0)}, "
              f"rejected by filter {col.rejected.get(s, 0)}")
    print(f"skipped by licence: {col.licence_skipped}")
    print(f"accepted this run: {sum(col.accepted.values())} (total in {out}: {col.total_accepted})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
