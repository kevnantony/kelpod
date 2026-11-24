# Credits

Kelpod (this repo) is an open-source, educational project. It builds on the work below. Every
library, model, dataset or piece of code we use is listed here with its licence and link; we follow
each licence's terms. Add an entry whenever you add something.

## Libraries

| Project | Licence | Link |
|---|---|---|
| NumPy | BSD-3-Clause | https://numpy.org |
| pydantic | MIT | https://github.com/pydantic/pydantic |
| OpenCV (opencv-python-headless) | Apache-2.0 | https://opencv.org |
| scikit-image | BSD-3-Clause | https://scikit-image.org |
| pypdfium2 (PDFium) | Apache-2.0 / BSD-3-Clause | https://github.com/pypdfium2-team/pypdfium2 |
| Pillow | MIT-CMU (HPND) | https://python-pillow.org |
| Shapely | BSD-3-Clause | https://github.com/shapely/shapely |
| SciPy | BSD-3-Clause | https://scipy.org |
| NetworkX | BSD-3-Clause | https://networkx.org |
| IfcOpenShell | LGPL-3.0 | https://ifcopenshell.org |
| RapidOCR (ONNX Runtime, PaddleOCR models) | Apache-2.0 | https://github.com/RapidAI/RapidOCR |
| Typer | MIT | https://github.com/fastapi/typer |
| FastAPI / Uvicorn | MIT / BSD-3-Clause | https://fastapi.tiangolo.com |
| Starlette | BSD-3-Clause | https://github.com/encode/starlette |
| python-multipart | Apache-2.0 | https://github.com/Kludex/python-multipart |

## Models and datasets

Used by the learned segmentation backend (`training/`, `src/kelpod/segment/learned.py`).

| Project | What we use | Licence | Link |
|---|---|---|---|
| CubiCasa5k dataset | 4200 train plans (images + SVG labels) for fine-tuning; val/test splits for evaluation | CC BY-NC 4.0 (Zenodo record: CC BY-NC-SA 4.0) | https://github.com/CubiCasa/CubiCasa5k, https://zenodo.org/records/2613548 |
| CubiCasa5k model (`hg_furukawa_original` + released weights) | zero-shot baseline; network code vendored in `training/kelpod_train/third_party/cubicasa_hg.py` | CC BY-NC 4.0 | https://github.com/CubiCasa/CubiCasa5k |
| Swiss Dwellings v3.0.0 | wall/opening/room geometry re-rendered in DIN 1356-1 style | CC BY 4.0 | https://zenodo.org/records/7788422 |
| ResPlan | vector plans re-rendered in DIN 1356-1 style | data CC BY 4.0, code MIT | https://github.com/m-agour/ResPlan |
| ImageNet-pretrained ResNet-34 encoder (torchvision weights via segmentation_models_pytorch) | encoder initialisation | BSD-3-Clause (torchvision) | https://pytorch.org/vision/stable/models.html |

### Training-only libraries (`training/pyproject.toml`)

| Project | Licence | Link |
|---|---|---|
| PyTorch / torchvision | BSD-3-Clause | https://pytorch.org |
| segmentation_models_pytorch | MIT | https://github.com/qubvel-org/segmentation_models.pytorch |
| timm | Apache-2.0 | https://github.com/huggingface/pytorch-image-models |
| pandas | BSD-3-Clause | https://pandas.pydata.org |
| svgpathtools | MIT | https://github.com/mathandy/svgpathtools |
| gdown | MIT | https://github.com/wkentaro/gdown |

## Real plan sources

Real floor-plan scans for the real-scan benchmark (RealBench-DE dev set and the sealed test-real
split). The images are stored outside the repo (`~/kelpod-data/real/`, `~/kelpod-data/sealed/`)
and never committed. Licence, author, source link and hash for each page are listed in
`~/kelpod-data/real/sources.csv`.

| Source | What we use | Licence | Link |
|---|---|---|---|
| Landesarchiv Baden-Württemberg, Abt. Staatsarchiv Ludwigsburg, Bestand K 412 IV (railway building plans, Württemberg/Baden, c. 1860-1960), via Wikimedia Commons | 49 floor-plan pages, one per station building | CC BY 4.0 (credit: "Landesarchiv Baden-Württemberg, Staatsarchiv Ludwigsburg K 412 IV <signature>") | https://commons.wikimedia.org/wiki/Category:Staatsarchiv_Ludwigsburg_Findbuch_K_412_IV, https://www.landesarchiv-bw.de |
| Landesarchiv Berlin, B Rep 202 (Bauakte Regentenstraße 19, Kayser & von Großheim, 1893), via Wikimedia Commons | 3 pages | Public domain (PD-old) | https://commons.wikimedia.org/wiki/Category:Wohnhaus_Regentenstraße_19_(Berlin) |
| Stadtarchiv Magdeburg (Bauakten Poststraße), via Wikimedia Commons | 3 pages | Public domain (PD-old) | https://commons.wikimedia.org/wiki/Category:Images_from_Stadtarchiv_Magdeburg |
| Historic building plans on Wikimedia Commons (Bonn Baugesuche, Magdeburg Eichamt, Leipzig Bauakten, München Hildebrandhaus, Berlin Villa Lehfeldt) | 9 pages | Public domain (PD-old); Villa Lehfeldt scan CC BY-SA 4.0 | per file in `sources.csv`; e.g. https://commons.wikimedia.org/wiki/Category:Floor_plans_of_buildings_in_Bonn |
| Museum für Hamburgische Geschichte, plan collection (Landhaus Rücker), via Wikimedia Commons | 2 pages | CC BY-SA 3.0 DE | https://commons.wikimedia.org/wiki/Category:Images_from_the_Museum_für_Hamburgische_Geschichte_Pläne |
| Wien Museum Online Sammlung (Otto Wagner, Karl Stigler, Hubert Gessner plans) | 5 pages | CC0 1.0 (credit the author and the Wien Museum, as the museum asks) | https://sammlung.wienmuseum.at |
