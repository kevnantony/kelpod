from kelpod.contracts import Node, PlanGraph, Room, Wall, WallKind
from kelpod.export import build_model, write_ifc
from kelpod.viz import ifc_meshes, write_viewer


def _box_graph() -> PlanGraph:
    pts = [(0, 0), (6, 0), (6, 4), (0, 4)]
    nodes = [Node(id=i, x=x, y=y) for i, (x, y) in enumerate(pts)]
    walls = [Wall(id=i, start=i, end=(i + 1) % 4, thickness_m=0.365, kind=WallKind.EXTERIOR)
             for i in range(4)]
    inner = [(0.1825, 0.1825), (5.8175, 0.1825), (5.8175, 3.8175), (0.1825, 3.8175)]
    room = Room(id=0, polygon=inner, name="Wohnen", area_m2=5.635 * 3.635)
    return PlanGraph(nodes=nodes, walls=walls, rooms=[room], m_per_px=0.0127,
                     scale_source="ground_truth", image_height_px=400)


def test_viewer_has_meshes_and_page(tmp_path):
    import cv2
    import numpy as np

    ifc = tmp_path / "m.ifc"
    write_ifc(build_model(_box_graph()), ifc)
    meshes = ifc_meshes(ifc)
    classes = {m["cls"] for m in meshes}
    assert "IfcSpace" in classes and ({"IfcWall", "IfcWallStandardCase"} & classes)
    png = tmp_path / "plan.png"
    cv2.imwrite(str(png), np.full((40, 60), 255, np.uint8))
    html = write_viewer(png, ifc, tmp_path / "v.html", note="test").read_text()
    assert "three.module.js" in html and "Wohnen" in html and "__DATA__" not in html
