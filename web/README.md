# ZLLF web UI

React + TypeScript single-page app for the Zero-Leak LoRA Factory. It talks to the FastAPI
backend described in [`../docs/API.md`](../docs/API.md) and is served by it from `web/dist`.

Everything is bundled locally: no web fonts, CDN scripts or analytics. The built app works with
the network unplugged.

Stack: Vite, React 19, React Router, TanStack Query, Tailwind CSS v4 (system font stack),
Vitest + Testing Library, ESLint + Prettier.

## Commands

```bash
cd web
npm install          # once (needs the npm registry or a local mirror)

npm run dev          # dev server on http://localhost:5173, proxies /api → http://127.0.0.1:8000
npm run dev:mock     # same, but with an in-memory fake backend (no Python needed)
npm run build        # type-check + production build into web/dist
npm run build:mock   # demo build with the fake backend into web/dist-mock
npm run lint         # eslint + prettier --check
npm run format       # prettier --write
npm run test         # vitest (watch); `npm run test -- --run` for a single run
```

## Mock mode

`VITE_MOCK=1` (used by `dev:mock`) swaps the API transport for `src/mock/backend.ts`, an
in-memory implementation of every endpoint in `docs/API.md`. It simulates a full run: documents
get parsed, the run moves through the six steps, pauses at the privacy-filter review until you
approve, produces training attempts, a scorecard with quality checks, and a deployable model with
a working playground.

- State is kept in `sessionStorage`; run `zllfMock.reset()` in the browser console to start over.
- Name a workspace with the word **"fail"** in it to get a scorecard where a quality check fails
  (Deploy stays disabled).
- The mock is loaded through a dynamic `import()` behind `import.meta.env.VITE_MOCK`, so a normal
  `npm run build` does not contain any mock code.

## Layout

```
src/
  api/        types.ts (mirrors docs/API.md), client.ts (fetch/XHR/SSE transport, errors), hooks.ts (TanStack Query)
  mock/       backend.ts (fake API + run simulation), install.ts (wires it into the client in mock mode)
  lib/        plain-language copy, formatting, schema templates + validation, gold-set parsing, folder drop helpers
  components/ UI primitives (ui.tsx), StageTimeline, ScorecardView, RedactionReview, DropZone, SchemaEditor, …
  pages/      one file per screen (setup, login, workspaces, documents, goal, run monitor, review, scorecard,
              playground, models, system)
```

All user-facing names for steps, quality checks, metrics and states live in `src/lib/copy.ts`
(PRD §7.3: no ML jargon on the default path; ready for translation).

## Notes for backend developers

- Errors must use `{"error": {"code": "E_…", "message": "…"}}`; `message` is shown to the user as is,
  so write it as "what happened + what to do".
- Uploads are sent in batches (≤25 files / 64 MB per request) as multipart `files[]` with a
  matching `paths[]` entry per file (relative path for folder uploads, otherwise the file name).
- The run monitor relies on SSE `progress` events (`stage`, `done`, `total`, `message`) and on
  `stage_started` / `stage_completed` / `stage_failed` / `awaiting_approval` / `cancelled`. It also
  understands `trial_completed`, `approved`, `resumed` and `run_completed` if the backend emits them.
- Report downloads are fetched with the session cookie and saved via a blob, so
  `/api/runs/{id}/report.pdf|json` only needs to return the file.
