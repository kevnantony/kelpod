"""Side-by-side demo page: the input plan next to an interactive 3D view of the IFC model.

``write_viewer(plan_image, ifc_path, out_html)`` tessellates every IFC product with
``ifcopenshell.geom`` and writes one self-contained HTML file (three.js from a CDN, model data and
plan image inline). Open it in any browser; drag to orbit, scroll to zoom.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import ifcopenshell
import ifcopenshell.geom
import numpy as np

COLOURS = {
    "IfcWall": "#d9d4cc",
    "IfcWallStandardCase": "#d9d4cc",
    "IfcWindow": "#7fb7e6",
    "IfcDoor": "#a9764a",
    "IfcSlab": "#9a958e",
    "IfcSpace": "#6cc08b",
}
SKIP = {"IfcOpeningElement"}


def ifc_meshes(ifc_path: str | Path) -> list[dict]:
    """World-coordinate triangle meshes per product: ``{cls, name, verts, faces, colour}``."""
    model = ifcopenshell.open(str(ifc_path))
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    meshes = []
    products = [p for p in model.by_type("IfcProduct") if p.Representation and p.is_a() not in SKIP]
    for product in products:
        try:
            shape = ifcopenshell.geom.create_shape(settings, product)
        except Exception:  # noqa: BLE001 - a product without buildable geometry is skipped
            continue
        verts = np.round(np.asarray(shape.geometry.verts, dtype=float), 4).tolist()
        faces = list(shape.geometry.faces)
        if not faces:
            continue
        cls = product.is_a()
        name = product.LongName if cls == "IfcSpace" and product.LongName else (product.Name or "")
        meshes.append({"cls": cls, "name": name, "verts": verts, "faces": faces,
                       "colour": COLOURS.get(cls, "#cccccc")})
    return meshes


def _image_data_url(path: str | Path, max_side: int = 1400) -> str:
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium

        page = pdfium.PdfDocument(str(p))[0]
        img = np.asarray(page.render(scale=150 / 72).to_pil().convert("L"))
    else:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    f = min(1.0, max_side / max(h, w))
    if f < 1.0:
        img = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def write_viewer(
    plan: str | Path, ifc_path: str | Path, out_html: str | Path, *, title: str = "Kelpod demo",
    note: str = "", stats: dict | None = None,
) -> Path:
    """Write the demo page and return its path."""
    data = {"meshes": ifc_meshes(ifc_path), "stats": stats or {}}
    html = (_TEMPLATE.replace("__TITLE__", title)
            .replace("__NOTE__", note)
            .replace("__IMG__", _image_data_url(plan))
            .replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    out = Path(out_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f6f5f2;--fg:#1d1d1b;--muted:#6b6a66;--card:#fff;--line:#e3e1dc}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe8;--muted:#a3a19c;--card:#1f1f1d;--line:#33322f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
header{padding:14px 18px;border-bottom:1px solid var(--line)}h1{margin:0;font-size:17px}
header p{margin:4px 0 0;color:var(--muted)}
main{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px;height:calc(100vh - 74px)}
@media (max-width:800px){main{grid-template-columns:1fr;height:auto}.pane{height:70vh}}
.pane{background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden;position:relative;display:flex;flex-direction:column}
.pane h2{margin:0;padding:8px 12px;font-size:13px;font-weight:600;border-bottom:1px solid var(--line)}
.plan{flex:1;display:flex;align-items:center;justify-content:center;background:#fff}
.plan img{max-width:100%;max-height:100%;object-fit:contain}
#view{flex:1;min-height:0}
.legend{position:absolute;left:10px;bottom:10px;font-size:12px;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:6px 8px}
.legend span{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 8px;vertical-align:-1px}
.stats{position:absolute;right:10px;top:42px;font-size:12px;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:6px 8px;white-space:pre}
</style></head><body>
<header><h1>__TITLE__</h1><p>__NOTE__</p></header>
<main>
<section class="pane"><h2>Input plan (PDF scan)</h2><div class="plan"><img src="__IMG__" alt="input floor plan"></div></section>
<section class="pane"><h2>IFC model in 3D (drag to orbit, scroll to zoom)</h2><div id="view"></div>
<div class="legend"><span style="background:#d9d4cc"></span>walls<span style="background:#7fb7e6"></span>windows<span style="background:#a9764a"></span>doors<span style="background:#6cc08b"></span>rooms</div>
<div class="stats" id="stats"></div></section>
</main>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
const DATA = __DATA__;
const el = document.getElementById("view");
const s = DATA.stats; document.getElementById("stats").textContent =
  Object.entries(s).map(([k,v])=>k+": "+v).join("\n") || "";
const scene = new THREE.Scene();
const dark = matchMedia("(prefers-color-scheme: dark)").matches;
scene.background = new THREE.Color(dark ? 0x1f1f1d : 0xfafaf8);
const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 500);
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(devicePixelRatio); el.appendChild(renderer.domElement);
scene.add(new THREE.HemisphereLight(0xffffff, 0x888877, 1.1));
const sun = new THREE.DirectionalLight(0xffffff, 1.4); sun.position.set(-8, 12, 15); scene.add(sun);
const group = new THREE.Group(); group.rotation.x = -Math.PI/2; scene.add(group); // IFC z-up -> three y-up
const labels = [];
for (const m of DATA.meshes) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(m.verts, 3));
  g.setIndex(m.faces); g.computeVertexNormals();
  const space = m.cls === "IfcSpace";
  const mat = new THREE.MeshStandardMaterial({color:m.colour, roughness:0.85, transparent: space || m.cls==="IfcWindow",
    opacity: space ? 0.18 : (m.cls==="IfcWindow" ? 0.6 : 1), side:THREE.DoubleSide, depthWrite: !space});
  const mesh = new THREE.Mesh(g, mat); group.add(mesh);
  if (!space) group.add(new THREE.LineSegments(new THREE.EdgesGeometry(g, 30),
    new THREE.LineBasicMaterial({color: dark ? 0x55534f : 0x8a877f})));
  if (space && m.name) { g.computeBoundingBox(); labels.push([m.name, g.boundingBox]); }
}
for (const [name, bb] of labels) {
  const c = document.createElement("canvas"); c.width = 256; c.height = 64;
  const ctx = c.getContext("2d"); ctx.font = "600 30px system-ui"; ctx.textAlign = "center";
  ctx.fillStyle = dark ? "#ecebe8" : "#1d1d1b"; ctx.fillText(name, 128, 42);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c), depthTest:false}));
  const ctr = new THREE.Vector3(); bb.getCenter(ctr); sp.position.set(ctr.x, ctr.y, bb.max.z + 0.4);
  sp.scale.set(2.4, 0.6, 1); group.add(sp);
}
const box = new THREE.Box3().setFromObject(group); const ctr = box.getCenter(new THREE.Vector3());
const size = box.getSize(new THREE.Vector3()).length();
const controls = new OrbitControls(camera, renderer.domElement); controls.target.copy(ctr);
let fitted = false;
function fit(){ // distance so the whole building fits the narrower side of the panel
  const vfov = camera.fov * Math.PI / 180, hfov = 2 * Math.atan(Math.tan(vfov/2) * camera.aspect);
  const d = 0.5 * size / Math.tan(Math.min(vfov, hfov) / 2);
  const dir = new THREE.Vector3(-0.2, 0.75, 0.63).normalize();
  camera.position.copy(ctr).addScaledVector(dir, d); controls.update();
}
function resize(){ const w = el.clientWidth, h = el.clientHeight; renderer.setSize(w, h); camera.aspect = w/h; camera.updateProjectionMatrix(); if (!fitted && w > 0) { fit(); fitted = true; } }
new ResizeObserver(resize).observe(el); resize();
renderer.setAnimationLoop(()=>{ controls.update(); renderer.render(scene, camera); });
</script></body></html>
"""
