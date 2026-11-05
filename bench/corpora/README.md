# Demo corpora (public documents only)

PRD §13.2: no dataset is committed unless its licence allows redistribution. This directory
holds **scripts and notes only**; fetched files go to `bench/corpora/data/<name>/`, which is
gitignored (and blocked by the pre-commit PDF hook). Fetch on the internet-connected build host,
then carry the folder to the appliance like any other import.

## Clean energy / critical infrastructure ("AI for Good", Mode A)

```bash
ZLLF_ALLOW_ONLINE=1 uv run python scripts/fetch_demo_corpus.py                  # 30 NRC PDFs
ZLLF_ALLOW_ONLINE=1 uv run python scripts/fetch_demo_corpus.py --limit 60
ZLLF_ALLOW_ONLINE=1 uv run python scripts/fetch_demo_corpus.py --source ferc --limit 20
uv run python scripts/fetch_demo_corpus.py --verify --source nrc                  # offline
uv run zllf run --corpus bench/corpora/data/nrc-fr --hardware toy --auto-approve
```

| `--source` | Output dir | Content | Licence |
|---|---|---|---|
| `nrc` (default) | `data/nrc-fr/` | US Nuclear Regulatory Commission final and proposed rules (reactor licensing, spent-fuel casks, emergency preparedness, fees) as published in the Federal Register | US Government work, public domain in the US (17 U.S.C. § 105) |
| `ferc` | `data/ferc-fr/` | Federal Energy Regulatory Commission rules (grid reliability standards, e.g. NERC CIP) | same |
| `doe` | `data/doe-fr/` | US Department of Energy rules (efficiency standards, programmes) | same |

How it works: the document list comes from the public
[Federal Register API](https://www.federalregister.gov/developers/documentation/api/v1)
(`/api/v1/documents.json`, filtered by agency and type `RULE`/`PRORULE`, newest first, 3–40
pages, corrections skipped). PDFs are downloaded from `www.govinfo.gov` (US GPO). Each output
folder has a `sources.json` with, per file: source URL, landing page, sha256, size, title,
document number, publication date, page count and licence. Default `--limit` is 30 PDFs
(≈ 5–10 MB).

Verified working on 2026-11-05: `--source nrc --limit 5` downloaded 5 PDFs (1.3 MiB, 3–14
pages each), `--verify` passed.

### US NRC Licensee Event Reports / Information Notices — not automated

The PRD names NRC Licensee Event Reports (LERs) and inspection reports. They are also US
Government works, but as of 2026-11-05 they cannot be fetched by a script in a polite way:

- The legacy ADAMS search API host `adams.nrc.gov` no longer resolves; the replacement
  ADAMS Public Search API (`adams-api.nrc.gov`) requires a registered subscription key.
- Direct PDF links on `www.nrc.gov/docs/MLxxxx/…pdf` (e.g. those listed on
  `https://www.nrc.gov/reading-rm/doc-collections/gen-comm/info-notices/2024/index.html`)
  returned HTTP 403 / timeouts to non-browser clients; we do not try to work around that.
- `lersearch.inl.gov` (LER search) is an interactive web app without a documented bulk API.

If you want LERs: register for an ADAMS API key and add a source to the script, or download a
handful of PDFs by hand in a browser into `data/nrc-ler/` and write a `sources.json` with the
ADAMS accession numbers (ML…) as source IDs.

## Rules

- Only public-domain or redistribution-permitting sources. Record the licence per file.
- Never commit `data/`. Large corpora (`ref-500`, `ref-1000p`) are built the same way.
- The fetch scripts are the only corpus code allowed to use the network and refuse to run
  without `ZLLF_ALLOW_ONLINE=1`.
