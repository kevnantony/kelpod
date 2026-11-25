"""Kelpod sandbox on Streamlit Community Cloud: upload a floor-plan scan, get a closed IFC model.

Entry point for share.streamlit.io (main file: ``deploy/streamlit/streamlit_app.py``). Runs the
same pipeline as ``kelpod convert`` with the classical segmenter (no model weights needed). One
conversion runs at a time, in a short-lived child process, to stay inside the free tier's memory;
uploads live in a temporary folder that is deleted right after the conversion.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

# Kelpod comes from the repo checkout this app is deployed from (not a pinned install), so every
# push to the deployed branch also updates the pipeline. The worker gets the same path.
SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(SRC)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
)

import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

WORKER = Path(__file__).with_name("kelpod_worker.py")

st.set_page_config(page_title="Kelpod sandbox", page_icon="🏠", layout="wide")

MAX_MB = 25
SAMPLES = {"Sample plan A (3 rooms)": 0, "Sample plan B (4 rooms)": 1, "Sample plan C (8 rooms)": 4}


@st.cache_resource
def _lock() -> threading.Lock:
    return threading.Lock()


def _in_child(fn_name: str, *args):
    """Run kelpod_worker.<fn_name> in a fresh Python process (one at a time); return its result."""
    with _lock(), tempfile.TemporaryDirectory() as d:
        args_path, out_path = Path(d) / "args.pkl", Path(d) / "out.pkl"
        args_path.write_bytes(pickle.dumps(args))
        proc = subprocess.run([sys.executable, str(WORKER), fn_name, str(args_path), str(out_path)],
                              capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            lines = proc.stderr.strip().splitlines()
            raise RuntimeError(lines[-1] if lines else "worker failed")
        return pickle.loads(out_path.read_bytes())


@st.cache_data(show_spinner=False)
def _sample_pdf(seed: int) -> bytes:
    return _in_child("sample_pdf", seed)


@st.cache_data(show_spinner=False, max_entries=8, ttl=3600)
def _convert(data: bytes, suffix: str, schema: str, plan_scale: float, height: float) -> dict:
    """Cached by content, so repeating a plan costs nothing."""
    return _in_child("convert_bytes", data, suffix, schema, plan_scale, height)


st.title("Kelpod sandbox")
st.markdown(
    "Upload a scanned German floor plan (*Bestandsplan*) and Kelpod turns it into a closed 3D "
    "building model (IFC) for energy software such as HottCAD. "
    "[Source code](https://github.com/kevnantony/kelpod) · open-source, educational project."
)
st.info(
    "**Sandbox.** Tested on synthetic plans only; results on real scans are not validated, and an "
    "expert must check every model. Uploads are processed in memory and deleted right after. "
    "Do not upload plans you are not allowed to share.",
    icon="ℹ️",
)

left, right = st.columns([3, 2])
with left:
    upload = st.file_uploader(f"Floor plan (PDF, PNG, JPG or TIFF, up to {MAX_MB} MB)",
                              type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"])
with right:
    sample = st.selectbox("…or try a sample plan", ["None", *SAMPLES], index=0)
    with st.expander("Options"):
        schema = st.radio("IFC schema", ["IFC4", "IFC2X3"], horizontal=True)
        plan_scale = st.number_input("Plan scale 1:", min_value=10, max_value=1000, value=100,
                                     step=10, help="Used until the dimension chains set the scale.")
        height = st.number_input("Room height (m)", min_value=2.0, max_value=5.0, value=2.60,
                                 step=0.05)

if upload is not None:
    data, suffix, name = upload.getvalue(), Path(upload.name).suffix.lower(), upload.name
    if len(data) > MAX_MB * 1024 * 1024:
        st.error(f"The file is larger than {MAX_MB} MB. Export the plan at a lower resolution.")
        st.stop()
elif sample != "None":
    with st.spinner("Drawing a sample plan…"):
        data, suffix, name = _sample_pdf(SAMPLES[sample]), ".pdf", f"{sample}.pdf"
else:
    st.caption("Upload a plan or pick a sample to start.")
    st.stop()

key = hashlib.sha256(data).hexdigest()[:12]
try:
    with st.spinner("Converting: clean → segment → read dimensions → fit → IFC…"):
        out = _convert(data, suffix, schema, float(plan_scale), float(height))
except Exception as exc:  # noqa: BLE001 - show any pipeline failure to the user
    st.error(f"Kelpod could not convert this file ({type(exc).__name__}). Scans with very "
             "unusual drawing styles are not supported yet.")
    st.stop()

s = out["stats"]
if out.get("segmenter") == "classical":
    st.info("This sandbox uses the classical segmenter: the learned model's weights are not "
            "deployed here yet, so results are below what Kelpod's main branch reaches.")
if out["closed"]:
    st.success(f"**Closed model.** {s['rooms']} rooms, {s['walls']} walls, {s['doors']} doors and "
               f"{s['windows']} windows; every wall end meets another wall.")
else:
    st.warning(f"**Not fully closed.** {out['dangling']} loose wall end(s). Rooms that could not "
               "be enclosed are missing from the model; an expert would fix these.")

cols = st.columns(5)
cols[0].metric("Rooms", s["rooms"])
joined = out["node_closure"]
cols[1].metric("Wall ends joined", f"{joined:.0%}" if joined is not None else "–")
from_dims = out["scale_source"] == "dimensions"
cols[2].metric("Scale", "dims" if from_dims else f"1:{plan_scale}",
               help="'dims': scale read from the dimension chains; else the plan scale you set.")
cols[3].metric("Time", f"{out['total']} s")
cols[4].metric("IFC re-opened", "yes" if out["reopened"] else "no")

plan_col, model_col = st.columns(2)
with plan_col:
    st.caption("Input: your plan")
    st.image(out["plan_png"], use_container_width=True)
with model_col:
    st.caption("Output: IFC model in 3D (drag to orbit, scroll to zoom)")
    components.html(out["viewer"], height=560, scrolling=False)

d1, d2 = st.columns([1, 3])
d1.download_button("Download IFC", out["ifc"], file_name=f"{Path(name).stem}.ifc",
                   mime="application/x-step", type="primary")
with d2.expander("Details"):
    rooms = zip(out["rooms"], out["areas"], strict=True)
    st.write("**Rooms:** " + ", ".join(f"{n} ({a} m²)" for n, a in rooms))
    st.write("**Time per step (s):** " + ", ".join(f"{k} {v}" for k, v in out["timings"].items()))
    for issue in out["issues"]:
        st.write(f"- {issue}")
    st.caption(f"File fingerprint {key}. Nothing is stored after the page is closed.")
