import numpy as np

from kelpod.contracts import (
    ExportResult,
    PlanGraph,
    RasterPage,
    m_to_px,
    px_to_m,
)


def test_px_m_roundtrip():
    p = (123.0, 45.0)
    m = px_to_m(p, 0.0127, 1000)
    back = m_to_px(m, 0.0127, 1000)
    assert np.allclose(back, p)
    assert m[1] > 0  # y up


def test_raster_page_scale():
    img = np.full((10, 20), 255, np.uint8)
    page = RasterPage(image=img, ink=img < 128, dpi=200, source="x.png")
    assert page.width == 20 and page.height == 10
    assert abs(page.m_per_px_from_scale - 0.0254 / 200 * 100) < 1e-12
    assert "image" not in page.model_dump()


def test_export_result_alias_and_graph_defaults():
    r = ExportResult(path="a.ifc", schema="IFC4", reopened_ok=True)
    assert r.schema_ == "IFC4"
    g = PlanGraph(m_per_px=0.01, scale_source="default", image_height_px=100)
    assert g.walls == [] and g.report is None
