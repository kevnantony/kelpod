#!/usr/bin/env python3
"""Fetch a small PUBLIC demo corpus for the clean-energy / critical-infrastructure use case.

Build-host only (needs ``ZLLF_ALLOW_ONLINE=1``); the appliance never runs this. Output goes to
``bench/corpora/data/<name>/`` (gitignored) with a ``sources.json`` recording, per file, the
source URL, sha256, title, date and licence note. See ``bench/corpora/README.md``.

Sources are US federal regulator documents published in the Federal Register. They are works
of the US Government and not subject to copyright in the US (17 U.S.C. § 105). Listing uses the
public Federal Register API (https://www.federalregister.gov/developers/documentation/api/v1);
PDFs are downloaded from govinfo.gov (GPO), which serves them for bulk/automated access.

Examples::

    ZLLF_ALLOW_ONLINE=1 uv run python scripts/fetch_demo_corpus.py              # nrc, 30 PDFs
    ZLLF_ALLOW_ONLINE=1 uv run python scripts/fetch_demo_corpus.py --source ferc --limit 10
    uv run python scripts/fetch_demo_corpus.py --verify --name nrc-fr          # offline

Stdlib only, so it also runs with a bare ``python3``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "bench" / "corpora" / "data"
FR_API = "https://www.federalregister.gov/api/v1/documents.json"
USER_AGENT = "zllf-fetch-demo-corpus/0.1 (offline LoRA appliance; public-domain demo corpus)"
LICENCE_NOTE = (
    "US Government work published in the Federal Register; not subject to copyright in the "
    "United States (17 U.S.C. 105). Source: Federal Register API / govinfo.gov (GPO)."
)
ALLOWED_PDF_HOSTS = {"www.govinfo.gov", "govinfo.gov"}

# name -> (default corpus name, Federal Register agency slugs, document types)
SOURCES: dict[str, tuple[str, list[str], list[str]]] = {
    # Nuclear Regulatory Commission: final + proposed rules (reactor safety, spent fuel,
    # emergency preparedness, cybersecurity of nuclear facilities ...).
    "nrc": ("nrc-fr", ["nuclear-regulatory-commission"], ["RULE", "PRORULE"]),
    # Federal Energy Regulatory Commission: grid reliability standards (NERC CIP, etc.).
    "ferc": ("ferc-fr", ["federal-energy-regulatory-commission"], ["RULE", "PRORULE"]),
    # Department of Energy: efficiency standards, energy programmes.
    "doe": ("doe-fr", ["energy-department"], ["RULE", "PRORULE"]),
}


def _get(url: str, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data: bytes = resp.read()
        return data


def list_documents(
    agencies: list[str], types: list[str], *, min_pages: int, max_pages: int, want: int
) -> list[dict[str, Any]]:
    """Newest-first Federal Register documents within the page-length window."""
    params: list[tuple[str, str]] = [("per_page", "100"), ("order", "newest")]
    params += [("conditions[agencies][]", a) for a in agencies]
    params += [("conditions[type][]", t) for t in types]
    for field in (
        "document_number",
        "title",
        "type",
        "publication_date",
        "pdf_url",
        "html_url",
        "page_length",
        "agencies",
    ):
        params.append(("fields[]", field))
    url: str | None = FR_API + "?" + urllib.parse.urlencode(params)
    picked: list[dict[str, Any]] = []
    pages = 0
    while url and len(picked) < want and pages < 10:
        payload = json.loads(_get(url))
        pages += 1
        for doc in payload.get("results", []):
            n = doc.get("page_length") or 0
            pdf = doc.get("pdf_url") or ""
            if not (min_pages <= n <= max_pages) or not pdf:
                continue
            if doc["document_number"].startswith("C"):  # corrections: tiny, low value
                continue
            if urllib.parse.urlparse(pdf).hostname not in ALLOWED_PDF_HOSTS:
                continue
            picked.append(doc)
            if len(picked) >= want:
                break
        url = payload.get("next_page_url")
    return picked


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(args: argparse.Namespace) -> int:
    if os.environ.get("ZLLF_ALLOW_ONLINE") != "1":
        print(
            "refusing to touch the network: set ZLLF_ALLOW_ONLINE=1 (build host only)",
            file=sys.stderr,
        )
        return 2
    default_name, agencies, types = SOURCES[args.source]
    name = args.name or default_name
    out = DATA_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    sources_path = out / "sources.json"
    existing = json.loads(sources_path.read_text()) if sources_path.is_file() else {}
    have = {d["file"]: d for d in existing.get("documents", [])}

    docs = list_documents(
        agencies, types, min_pages=args.min_pages, max_pages=args.max_pages, want=args.limit
    )
    print(f"[list] {len(docs)} documents from {', '.join(agencies)}")
    records: list[dict[str, Any]] = []
    for i, doc in enumerate(docs, 1):
        fname = f"{doc['document_number']}.pdf"
        path = out / fname
        if path.is_file() and fname in have:
            data = path.read_bytes()
            if sha256_bytes(data) == have[fname]["sha256"]:
                records.append(have[fname])
                print(f"[{i:>3}/{len(docs)}] cached {fname}")
                continue
        data = _get(doc["pdf_url"])
        if not data.startswith(b"%PDF"):
            print(f"[{i:>3}/{len(docs)}] SKIP {fname}: not a PDF", file=sys.stderr)
            continue
        path.write_bytes(data)
        records.append(
            {
                "file": fname,
                "sha256": sha256_bytes(data),
                "size": len(data),
                "title": doc["title"],
                "type": doc.get("type"),
                "document_number": doc["document_number"],
                "publication_date": doc["publication_date"],
                "pages": doc.get("page_length"),
                "agencies": [a.get("name") for a in doc.get("agencies", []) if a.get("name")],
                "source_url": doc["pdf_url"],
                "landing_url": doc.get("html_url"),
                "licence": "US-Government-Work (public domain in the US)",
            }
        )
        kib = len(data) / 1024
        print(f"[{i:>3}/{len(docs)}] {fname} ({kib:.0f} KiB, {doc.get('page_length')} p)")
        time.sleep(args.delay)

    manifest = {
        "name": name,
        "source": args.source,
        "description": "Public Federal Register documents for the ZLLF clean-energy demo",
        "licence_note": LICENCE_NOTE,
        "listing_api": FR_API,
        "query": {
            "agencies": agencies,
            "types": types,
            "page_length": [args.min_pages, args.max_pages],
            "limit": args.limit,
        },
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),  # noqa: UP017 (py3.9)
        "documents": records,
    }
    sources_path.write_text(json.dumps(manifest, indent=2) + "\n")
    total = sum(r["size"] for r in records)
    print(f"[done] {len(records)} PDFs, {total / 2**20:.1f} MiB -> {out.relative_to(REPO_ROOT)}")
    return 0 if records else 1


def verify(args: argparse.Namespace) -> int:
    name = args.name or SOURCES[args.source][0]
    sources_path = DATA_ROOT / name / "sources.json"
    if not sources_path.is_file():
        print(f"no {sources_path}", file=sys.stderr)
        return 1
    bad = 0
    for rec in json.loads(sources_path.read_text())["documents"]:
        path = sources_path.parent / rec["file"]
        if not path.is_file() or sha256_bytes(path.read_bytes()) != rec["sha256"]:
            print(f"[BAD] {rec['file']}")
            bad += 1
    print(f"[verify] {name}: {'OK' if not bad else f'{bad} bad'}")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--source", choices=sorted(SOURCES), default="nrc")
    p.add_argument("--name", help="output dir name under bench/corpora/data (default per source)")
    p.add_argument("--limit", type=int, default=30, help="number of PDFs (default 30)")
    p.add_argument("--min-pages", type=int, default=3)
    p.add_argument("--max-pages", type=int, default=40)
    p.add_argument("--delay", type=float, default=0.5, help="seconds between downloads")
    p.add_argument("--verify", action="store_true", help="offline: re-hash against sources.json")
    args = p.parse_args(argv)
    return verify(args) if args.verify else fetch(args)


if __name__ == "__main__":
    raise SystemExit(main())
