import xml.etree.ElementTree as ET

import pytest
from hypothesis import given

from regexsmith.dsl import parse
from regexsmith.railroad import Box, Loop, Seq, Skip, Stack, layout, to_svg, to_text

from .strategies import trees


def test_literal_runs_merge_into_one_box():
    assert layout(parse(r"com\d")) == Seq((Box("com", True), Box(r"\d", False)))


def test_layout_of_quantifiers():
    assert layout(parse("a?")) == Skip(Box("a", True))
    assert layout(parse("a+")) == Loop(Box("a", True), "", False)
    assert layout(parse("a*")) == Loop(Box("a", True), "", True)
    assert layout(parse("a{2,4}")) == Loop(Box("a", True), "2-4x", False)
    assert layout(parse("(a|bc)")) == Stack((Box("a", True), Box("bc", True)))


def test_terminal_diagram_of_a_real_regex():
    assert to_text(parse(r"[a-z]+@(com|org)?")) == "\n".join(
        [
            '●─┬─[a-z]─┬─"@"─┬─────────┬─●',
            '  └───<───┘     └┬─"com"─┬┘',
            '                 └─"org"─┘',
        ]
    )


@pytest.mark.parametrize("regex", ["a", r"\d{1,3}(\.\d{1,3}){3}", "(ab|c)*d?", "x|y|z"])
def test_every_row_of_the_terminal_diagram_lines_up(regex):
    lines = to_text(parse(regex)).splitlines()
    assert any("●" in line for line in lines)
    assert sum(line.count("●") for line in lines) == 2


@given(trees)
def test_svg_is_well_formed_for_any_regex(tree):
    root = ET.fromstring(to_svg(tree, "caption <&>"))
    assert root.tag.endswith("svg")


def test_svg_escapes_text():
    assert "&lt;" in to_svg(parse("<"))
