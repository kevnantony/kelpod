"""Historic room names: hyphen joining, exterior spaces."""

from __future__ import annotations

from kelpod.contracts import TextItem
from kelpod.ocr.names import join_hyphenated, match_historic_name, pretty
from kelpod.roomnames import is_exterior


def _it(text, x, y, w=60, h=12):
    return TextItem(text=text, quad=[(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def test_hyphen_with_abbreviation_tail_keeps_hyphen():
    out = join_hyphenated([_it("SPEISE-", 100, 100), _it("S.", 120, 114, w=15)])
    assert [i.text for i in out] == ["SPEISE-S."]


def test_hyphen_with_full_word_drops_hyphen():
    out = join_hyphenated([_it("FEST-", 100, 100), _it("RÄUME", 100, 116)])
    assert out[0].text == "FESTRÄUME"


def test_far_or_unrelated_lines_are_not_joined():
    items = [_it("FEST-", 100, 100), _it("RÄUME", 100, 300), _it("KÜCHE", 400, 112)]
    assert len(join_hyphenated(items)) == 3


def test_no_hyphen_no_change():
    items = [_it("ANKLEIDE", 100, 100), _it("Z.", 120, 114)]
    assert join_hyphenated(items) == items


def test_pretty_and_match_historic():
    assert pretty("SPEISE-S.") == "Speise-S."
    assert match_historic_name("SCHLAFZ.") == "Schlafz."
    assert match_historic_name("Frühstz.") == "Frühstz."
    assert match_historic_name("4,35") is None
    assert match_historic_name("xyzzy") is None


def test_exterior_names():
    for n in ("GARTEN", "Vorhof", "Altan", "Terrasse", "Balkon", "Loggia", "Hof"):
        assert is_exterior(n), n
    for n in ("Wintergarten", "Lichthof", "Küche", None, "Speise-S."):
        assert not is_exterior(n), n


def test_exterior_room_is_left_out_of_the_model():
    from kelpod.contracts import Node, PlanGraph, Room, Wall
    from kelpod.export import build_model

    sq = [(0, 0), (4, 0), (4, 4), (0, 4)]
    nodes = [Node(id=i, x=x, y=y) for i, (x, y) in enumerate(sq)]
    walls = [Wall(id=i, start=i, end=(i + 1) % 4, thickness_m=0.3) for i in range(4)]
    g = PlanGraph(
        nodes=nodes,
        walls=walls,
        rooms=[Room(id=1, polygon=sq, name="Küche", area_m2=16), Room(id=2, polygon=sq, name="Garten", exterior=True, area_m2=16)],
        m_per_px=0.01,
        scale_source="default",
        image_height_px=100,
    )
    names = [s.name for s in build_model(g).storeys[0].spaces]
    assert names == ["Küche"]


def test_fuzzy_repairs_small_scan_noise():
    assert match_historic_name("Framdenz.") == "Fremdenz."
    assert match_historic_name("Schlatr.") == "Schlafz."
    assert match_historic_name("qqqqqqq") is None
