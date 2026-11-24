# Kelpod sandbox on Streamlit Community Cloud

Free hosting with uploads, no card. Deploy once:

1. Sign in at <https://share.streamlit.io> with GitHub.
2. **Create app** → "Deploy a public app from GitHub":
   - Repository: `kevnantony/kelpod`
   - Branch: `claude/kelpod-mvp-fa1abc` (or `main` once this folder is merged)
   - Main file path: `deploy/streamlit/streamlit_app.py`
   - App URL: e.g. `kelpod`
   - Advanced settings → Python **3.12**
3. Deploy. The first build takes a few minutes.

The app imports Kelpod from the repo checkout it is deployed from (`src/` on `sys.path`, also for
the worker via `PYTHONPATH`); `requirements.txt` lists only third-party packages. Deploy the app
from `main`: every push to `main` then redeploys the sandbox.
System libraries for OpenCV come from `packages.txt` at the repo root (Streamlit Cloud only reads it
there).

It uses the classical segmenter (no model weights) at 160 dpi, runs one conversion at a time in a
short-lived child process, and deletes uploads right after converting. Measured on the Mac: the
page idles at about 150 MB; a conversion peaks at about 0.93 GB (8-room synthetic plan). Run locally:
`uv run --with streamlit streamlit run deploy/streamlit/streamlit_app.py`.
