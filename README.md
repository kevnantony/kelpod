# Kelpod

Turns old German floor-plan scans (*Bestandspläne*: PDF, PNG, JPG, TIFF) into topologically closed
IFC 2x3 / IFC4 models that energy software such as HottCAD can import.

Status: MVP in progress, tested on synthetic plans only. See [KELPOD_CONTEXT.md](KELPOD_CONTEXT.md)
for the product and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the pipeline.

```bash
uv sync --all-extras
uv run kelpod convert plan.pdf -o model.ifc
uv run pytest
```
