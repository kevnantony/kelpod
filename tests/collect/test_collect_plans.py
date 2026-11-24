"""Tests for scripts/collect_plans.py. No network: all HTTP is faked."""
import importlib.util
import io
import json
import sys
import urllib.error
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

_spec = importlib.util.spec_from_file_location(
    "collect_plans", Path(__file__).resolve().parents[2] / "scripts" / "collect_plans.py")
cp = importlib.util.module_from_spec(_spec)
sys.modules["collect_plans"] = cp
_spec.loader.exec_module(cp)


def plan_image(w=900, h=700, shift=0):
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    d.rectangle([50 + shift, 50, w - 50, h - 50], outline="black", width=4)
    d.line([50 + shift, h // 2, w - 50, h // 2], fill="black", width=3)
    d.line([w // 2 + shift, 50, w // 2 + shift, h - 50], fill="black", width=3)
    d.line([w // 4, h // 2, w // 4, h - 50], fill="black", width=3)
    return im


def noise_image(w=900, h=700, seed=0):
    a = np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(a)


def png_bytes(im):
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


class FakeHttp:
    def __init__(self, files=None):
        self.files = files or {}
        self.calls = []

    def robots_allowed(self, url):
        return True

    def get(self, url, params=None, max_bytes=None, headers=None):
        self.calls.append(url)
        return self.files[url]


# ---- parsers

def test_parse_openverse():
    data = {"results": [
        {"url": "https://x/a.png", "foreign_landing_url": "https://x/page", "title": "Grundriss",
         "license": "by-sa", "license_version": "4.0", "license_url": "https://cc/by-sa", "creator": "Ann",
         "width": 800, "height": 600, "filetype": "png"},
        {"url": "https://x/b.svg", "license": "cc0", "filetype": "svg"},
        {"title": "no url"},
    ]}
    c = cp.parse_openverse(data, "q")
    assert len(c) == 1
    assert (c[0].source, c[0].licence, c[0].author, c[0].width) == ("openverse", "by-sa 4.0", "Ann", 800)
    assert c[0].page_url == "https://x/page"


def test_parse_commons():
    data = {"query": {"pages": {
        "1": {"index": 2, "title": "File:B.jpg", "imageinfo": [{"url": "https://u/B.jpg", "mime": "image/jpeg",
              "width": 1000, "height": 700, "size": 5, "descriptionurl": "https://c/B",
              "extmetadata": {"LicenseShortName": {"value": "CC BY 3.0"},
                              "LicenseUrl": {"value": "https://cc/3"},
                              "Artist": {"value": "<a href='x'>Bob</a>"}}}]},
        "2": {"index": 1, "title": "File:A.pdf", "imageinfo": [{"url": "https://u/A.pdf", "mime": "application/pdf"}]},
        "3": {"index": 3, "title": "File:C.png"},
    }}}
    c = cp.parse_commons(data, "q")
    assert len(c) == 1
    assert c[0].title == "B.jpg" and c[0].licence == "CC BY 3.0" and c[0].author == "Bob"


def test_parse_google():
    data = {"items": [
        {"link": "https://g/a.jpg", "title": "t", "mime": "image/jpeg",
         "image": {"contextLink": "https://g/p", "width": 900, "height": 600, "byteSize": 1234}},
        {"title": "nolink"},
    ]}
    c = cp.parse_google(data, "q")
    assert len(c) == 1 and c[0].licence.startswith("unverified") and c[0].page_url == "https://g/p"


# ---- hashing / filter

def test_dhash_similarity():
    a = plan_image()
    b = a.copy()
    ImageDraw.Draw(b).point((3, 3), "black")
    assert cp.hamming(cp.dhash(a), cp.dhash(b)) <= cp.DHASH_MAX_DIST
    assert cp.hamming(cp.dhash(a), cp.dhash(noise_image())) > cp.DHASH_MAX_DIST


def test_filter_accepts_plan_rejects_noise_and_small():
    ok, scores, reason = cp.plan_filter(plan_image())
    assert ok, (scores, reason)
    ok, _, reason = cp.plan_filter(noise_image())
    assert not ok and reason
    ok, _, reason = cp.plan_filter(plan_image(300, 300))
    assert not ok and reason.startswith("too_small")
    # coloured drawing is rejected by saturation / white fraction
    col = Image.new("RGB", (800, 800), (200, 60, 60))
    ok, _, _ = cp.plan_filter(col)
    assert not ok


def test_long_runs():
    b = np.zeros((3, 10), bool)
    b[0, 1:7] = True
    b[1, 2:4] = True
    m = cp._long_runs(b, 5)
    assert m[0, 1:7].all() and not m[1].any() and not m[0, 0] and not m[0, 7]


# ---- collector

def test_collector_dedupe_filter_resume(tmp_path):
    plan = plan_image()
    near = plan.copy()
    ImageDraw.Draw(near).point((5, 5), "black")  # different bytes, same dhash
    files = {
        "https://x/1.png": png_bytes(plan),
        "https://x/2.png": png_bytes(plan),          # same sha
        "https://x/3.png": png_bytes(near),          # same dhash
        "https://x/4.png": png_bytes(noise_image()),  # rejected
        "https://x/5.png": png_bytes(plan_image(shift=60)),  # distinct plan
    }
    cands = [cp.Candidate("openverse", "q", u, title=u, licence="cc0") for u in files]
    http = FakeHttp(files)
    col = cp.Collector(out=tmp_path, http=http, max_new=10, log=lambda *_: None)
    status = [col.process(c) for c in cands]
    assert status[0] == "accepted"
    assert status[1] == "dup_sha"
    assert status[2] == "dup_dhash"
    assert status[3] == "rejected"
    assert status[4] in ("accepted", "dup_dhash")
    lines = [json.loads(l) for l in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert lines[0]["sha256"] and lines[0]["dhash"] and lines[0]["licence"] == "cc0"
    assert "line_ratio" in lines[0]["filter"]
    assert (tmp_path / "openverse").is_dir() and not (tmp_path / "rejected").exists()  # rejects only logged
    assert len((tmp_path / "rejected.jsonl").read_text().splitlines()) == 1
    assert "filter_score" in lines[0]
    # resume: a new collector skips everything already seen without downloading
    http2 = FakeHttp(files)
    col2 = cp.Collector(out=tmp_path, http=http2, max_new=10, log=lambda *_: None)
    assert all(col2.process(c) == "seen" for c in cands)
    assert http2.calls == []


def test_collector_skips_too_large_and_robots(tmp_path):
    class NoRobots(FakeHttp):
        def robots_allowed(self, url):
            return False
    col = cp.Collector(out=tmp_path, http=NoRobots(), max_new=1, log=lambda *_: None)
    assert col.process(cp.Candidate("commons", "q", "https://x/a.png", licence="cc0")) == "robots"
    col = cp.Collector(out=tmp_path / "b", http=FakeHttp(), max_new=1, log=lambda *_: None)
    assert col.process(cp.Candidate("commons", "q", "https://x/b.png", licence="cc0", size=cp.MAX_BYTES + 1)) == "too_large"


def test_run_stops_at_max_and_round_robins(tmp_path):
    files = {f"https://x/{i}.png": png_bytes(plan_image(shift=i * 40)) for i in range(3)}

    def fetch_a(http, q, page):
        return [cp.Candidate("openverse", q, u, licence="cc0") for u in files]
    col = cp.Collector(out=tmp_path, http=FakeHttp(files), max_new=1, log=lambda *_: None)
    col.run({"openverse": fetch_a}, ["q"], max_pages=2)
    assert sum(col.accepted.values()) == 1


# ---- google quota

def test_google_quota_counts_and_stops(tmp_path):
    q = cp.GoogleQuota(tmp_path / "q.json", limit=3, today=lambda: "2030-01-01")
    assert [q.use() for _ in range(4)] == [True, True, True, False]
    assert q.remaining() == 0
    # persisted for the same day, reset on a new day
    assert cp.GoogleQuota(tmp_path / "q.json", limit=3, today=lambda: "2030-01-01").remaining() == 0
    assert cp.GoogleQuota(tmp_path / "q.json", limit=3, today=lambda: "2030-01-02").remaining() == 3


def test_google_fetcher_stops_when_quota_hit(tmp_path):
    quota = cp.GoogleQuota(tmp_path / "q.json", limit=1, today=lambda: "2030-01-01")
    fetch = cp.make_google_fetcher("k", "cx", quota)

    class J:
        def get_json(self, url, params=None):
            assert params["searchType"] == "image" and params["num"] == 10
            return {"items": []}
    assert fetch(J(), "q", 1) == []
    try:
        fetch(J(), "q", 2)
        assert False
    except cp.QuotaExhausted:
        pass


def test_google_skipped_without_env(tmp_path, capsys):
    f = cp.build_fetchers(["google", "openverse"], tmp_path, env={})
    assert list(f) == ["openverse"]
    assert "skipped" in capsys.readouterr().out
    assert cp.build_fetchers(["google"], tmp_path, env={"GOOGLE_CSE_KEY": "k"}) == {}
    assert cp.build_fetchers(["google"], tmp_path, env={"GOOGLE_CSE_CX": "c"}) == {}
    f = cp.build_fetchers(["google"], tmp_path, env={"GOOGLE_CSE_KEY": "k", "GOOGLE_CSE_CX": "c"})
    assert "google" in f


# ---- http politeness

def test_http_throttle_and_retry(monkeypatch):
    t = [0.0]
    sleeps = []
    h = cp.Http(sleep=lambda s: (sleeps.append(s), t.__setitem__(0, t[0] + s)), clock=lambda: t[0])
    h._throttle("https://a.org/x")
    h._throttle("https://a.org/y")
    assert sleeps and sleeps[0] >= 0.99
    sleeps.clear()
    h._throttle("https://b.org/x")  # other host: no wait
    assert not sleeps

    n = {"c": 0}

    class R:
        headers = {}
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self, *a): return b"ok"

    def fake_urlopen(req, timeout):
        n["c"] += 1
        assert req.get_header("User-agent") == cp.USER_AGENT
        if n["c"] < 3:
            raise urllib.error.URLError("boom")
        return R()
    monkeypatch.setattr(cp.urllib.request, "urlopen", fake_urlopen)
    assert h.get("https://c.org/") == b"ok" and n["c"] == 3


# ---- licences, reference dedupe, per-query cap, robots

def test_classify_licence():
    for ok in ["cc0", "pdm", "by", "by-sa", "CC BY 4.0", "CC BY-SA 3.0", "CC0", "Public domain", "PD-old"]:
        assert cp.classify_licence(ok) == "ok", ok
    for nc in ["by-nc", "by-nd", "by-nc-sa", "CC BY-NC-ND 4.0", "CC BY-ND 2.0"]:
        assert cp.classify_licence(nc) == "nc", nc
    for no in ["unknown", "", "All rights reserved", "GFDL"]:
        assert cp.classify_licence(no) == "no", no


def test_licence_filter_in_collector(tmp_path):
    files = {"https://x/a.png": png_bytes(plan_image()), "https://x/b.png": png_bytes(plan_image(shift=70))}
    col = cp.Collector(out=tmp_path, http=FakeHttp(files), max_new=9, log=lambda *_: None)
    assert col.process(cp.Candidate("openverse", "q", "https://x/a.png", licence="by-nc")) == "licence"
    assert col.process(cp.Candidate("openverse", "q", "https://x/b.png", licence="unknown")) == "licence"
    assert col.licence_skipped == {"by-nc": 1, "unknown": 1}
    col2 = cp.Collector(out=tmp_path / "nc", http=FakeHttp(files), max_new=9, allow_nc=True, log=lambda *_: None)
    assert col2.process(cp.Candidate("openverse", "q", "https://x/a.png", licence="by-nc")) == "accepted"
    rec = json.loads((tmp_path / "nc" / "manifest.jsonl").read_text())
    assert rec["licence_class"] == "nc" and rec["licence"] == "by-nc"
    assert rec["file"].startswith("openverse/")


def test_reference_dirs_dedupe(tmp_path):
    ref = tmp_path / "ref" / "sub"
    ref.mkdir(parents=True)
    plan_image().save(ref / "old.png")
    near = plan_image()
    ImageDraw.Draw(near).point((5, 5), "black")
    files = {"https://x/n.png": png_bytes(near), "https://x/s.png": png_bytes(plan_image())}
    col = cp.Collector(out=tmp_path / "o", http=FakeHttp(files), max_new=9, ref_dirs=(str(tmp_path / "ref"),),
                       log=lambda *_: None)
    assert col.process(cp.Candidate("commons", "q", "https://x/n.png", licence="cc0")) == "dup_dhash"
    col.seen_urls.clear()
    assert col.process(cp.Candidate("commons", "q", "https://x/s.png", licence="cc0")) == "dup_sha"


def test_max_per_query(tmp_path):
    files = {f"https://x/{i}.png": png_bytes(plan_image(shift=i * 45)) for i in range(4)}
    col = cp.Collector(out=tmp_path, http=FakeHttp(files), max_new=99, max_per_query=1, log=lambda *_: None)
    col.run({"openverse": lambda h, q, p: [cp.Candidate("openverse", q, u, licence="cc0") for u in files]},
            ["q"], max_pages=1)
    assert sum(col.accepted.values()) == 1


def test_pixel_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(cp, "MAX_PIXELS", 1000)
    col = cp.Collector(out=tmp_path, http=FakeHttp({"https://x/a.png": png_bytes(plan_image())}), max_new=1,
                       log=lambda *_: None)
    assert col.process(cp.Candidate("commons", "q", "https://x/a.png", licence="cc0")) == "too_many_pixels"


def test_robots_parsing(monkeypatch):
    h = cp.Http(sleep=lambda s: None)
    robots = b"User-agent: *\nDisallow: /private/\n"
    monkeypatch.setattr(h, "get", lambda url, params=None, max_bytes=None, headers=None: robots)
    assert h.robots_allowed("https://a.org/pub/x.png")
    assert not h.robots_allowed("https://a.org/private/x.png")

    h2 = cp.Http(sleep=lambda s: None)

    def boom(url, params=None, max_bytes=None, headers=None):
        raise OSError("down")
    monkeypatch.setattr(h2, "get", boom)
    assert not h2.robots_allowed("https://b.org/x.png")  # unreachable robots.txt: conservative


def test_filter_needs_two_directions():
    im = Image.new("RGB", (900, 700), "white")
    d = ImageDraw.Draw(im)
    for y in range(60, 650, 40):
        d.line([50, y, 850, y], fill="black", width=3)  # horizontal only
    ok, _, reason = cp.plan_filter(im)
    assert not ok and reason.startswith("not_two_orthogonal")
