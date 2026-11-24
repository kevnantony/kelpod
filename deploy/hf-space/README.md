---
title: Kelpod
emoji: 🏠
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: other
short_description: Old German floor-plan scans to closed IFC models (sandbox)
---

# Kelpod sandbox API

Backend for the sandbox at <https://kevinantony.de/kelpod>. Upload a floor plan (PDF, PNG, JPG,
TIFF) to `POST /convert` and get a closed IFC4 model back. Educational, open-source project:
<https://github.com/kevnantony/kelpod>.

- `GET /health` · `POST /convert` (multipart `file`, optional `schema`, `plan_scale`, …)
- `GET /jobs/{id}` · `GET /jobs/{id}/model.ifc` · `GET /jobs/{id}/meshes.json` · `GET /jobs/{id}/plan.png`
- API docs: `/docs`

**Sandbox only.** This is a public demo on shared hardware: uploads are deleted after 15 minutes,
but do not upload plans you are not allowed to share. The pipeline has been tested on synthetic
plans only; results on real scans are not validated, and an expert must check every model.

Source of this Space: `deploy/hf-space/` in the GitHub repo.
