import pytest

from kelpod.ocr.parse import (
    is_brh_keyword,
    is_opening_label,
    match_room_name,
    parse_area,
    parse_length_m,
    parse_number,
    parse_sill_height,
)


@pytest.mark.parametrize(
    ("text", "metres"),
    [
        ("3,51", 3.51),
        ("11,24", 11.24),
        ("0,885", 0.885),
        ("1,01", 1.01),
        ("2,5", 2.5),
        ("88,5", 0.885),  # centimetres with a half
        ("36,5", 0.365),
        ("76", 0.76),  # plain integers are centimetres
        ("24", 0.24),
        ("1,01⁵", 1.015),  # superscript millimetres
        ("88⁵", 0.885),
        ("885", 0.885),  # superscript lost by OCR
        ("O,885", 0.885),  # O -> 0
        ("l,26", 1.26),  # l -> 1
        ("3,5l", 3.51),
        ("3.51", 3.51),  # dot instead of comma
        (" 3 , 51 ", 3.51),
        ("4,26.", 4.26),
    ],
)
def test_parse_length(text, metres):
    assert parse_length_m(text) == pytest.approx(metres)


@pytest.mark.parametrize("text", ["", "Bad", "IS", "1:100", "BRH 0,90", "a,b", "14,52 m²", "M", "0"])
def test_parse_length_rejects(text):
    assert parse_length_m(text) is None


def test_parse_number_flags():
    n = parse_number("O,885")
    assert n is not None and n.fixed and n.has_comma and n.decimal_digits == 3
    n = parse_number("1,01⁵")
    assert n is not None and n.superscript and not n.fixed


@pytest.mark.parametrize(
    ("text", "metres"),
    [
        ("BRH 0,90", 0.90),
        ("BRH 87,5", 0.875),
        ("Brh.=90", 0.90),
        ("BRH=1,10", 1.10),
        ("BRH: 87,5 cm", 0.875),
        ("brh 0,9", 0.9),
        ("8RH 0,9O", 0.90),  # OCR confusions
        ("BRH 88⁵", 0.885),
    ],
)
def test_parse_sill_height(text, metres):
    assert parse_sill_height(text) == pytest.approx(metres)


@pytest.mark.parametrize("text", ["Bad", "3,51", "BRH", "Wohnen 14,52 m²"])
def test_parse_sill_height_rejects(text):
    assert parse_sill_height(text) is None


def test_brh_keyword():
    assert is_brh_keyword("BRH")
    assert is_brh_keyword("Brh.=")
    assert not is_brh_keyword("BRH 0,90")
    assert not is_brh_keyword("Bad")


@pytest.mark.parametrize(
    ("text", "area", "before"),
    [
        ("14,52 m²", 14.52, ""),
        ("14,52 m2", 14.52, ""),
        ("14,52m²", 14.52, ""),
        ("7,9 qm", 7.9, ""),
        ("12,3 M2", 12.3, ""),
        ("Wohnen 14,52 m²", 14.52, "Wohnen"),
        ("Bad: 4,10 m2", 4.10, "Bad"),
    ],
)
def test_parse_area(text, area, before):
    res = parse_area(text)
    assert res is not None
    assert res[0] == pytest.approx(area)
    assert res[1] == before


@pytest.mark.parametrize("text", ["14,52", "Wohnen", "m²", "BRH 0,90"])
def test_parse_area_rejects(text):
    assert parse_area(text) is None


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("Wohnen", "Wohnen"),
        ("WOHNEN", "Wohnen"),
        ("Küche", "Küche"),
        ("Kuche", "Küche"),  # umlaut lost by OCR
        ("Kueche", "Küche"),
        ("Schiafen", "Schlafen"),  # l read as i
        ("Bad", "Bad"),
        ("WC", "WC"),
        ("Flur", "Flur"),
        ("Abst.", "Abst."),
        ("Kind 2", "Kind 2"),
        ("Zi. 1", "Zi. 1"),
        ("Büro", "Büro"),
        ("Buero", "Büro"),
    ],
)
def test_match_room_name(text, name):
    assert match_room_name(text) == name


@pytest.mark.parametrize("text", ["3,51", "BRH 0,90", "Bal", "Haus", "", "x"])
def test_match_room_name_rejects(text):
    assert match_room_name(text) is None


def test_opening_label():
    assert is_opening_label("88,5/2,01")
    assert is_opening_label("1,01 / 1,385")
    assert not is_opening_label("3,51")
