import re

import pytest
from hypothesis import given, settings

from regexsmith.dsl import parse, to_regex
from regexsmith.semantics import compare, matches, may_backtrack_badly

from .strategies import texts, trees


@settings(max_examples=300)
@given(trees, texts)
def test_derivatives_agree_with_the_re_module(tree, text):
    assert matches(tree, text) == bool(re.fullmatch(to_regex(tree), text))


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("(a|b)+", "(a+b*)+|(b+a*)+"),
        ("a?a?", "aa|a?"),
        (r"\d", "[0-9]"),
        ("aa*", "a+"),
        ("(ab)*a", "a(ba)*"),
        ("a{2,3}", "aaa?"),
    ],
)
def test_equivalent_regexes(a, b):
    assert compare(parse(a), parse(b)).equal is True


@pytest.mark.parametrize(
    ("a", "b"),
    [("a+", "a*"), ("[a-z]", r"\w"), ("(ab)+", "(ab)*"), ("a{2}", "a{3}"), (".", r"\w")],
)
def test_different_regexes_produce_a_real_witness(a, b):
    left, right = parse(a), parse(b)
    result = compare(left, right)
    assert result.equal is False
    assert matches(left, result.witness) != matches(right, result.witness)


@given(trees, trees)
def test_comparison_is_consistent_with_matching(a, b):
    result = compare(a, b)
    if result.equal is False:
        assert matches(a, result.witness) != matches(b, result.witness)


@given(trees)
def test_a_regex_equals_itself(tree):
    assert compare(tree, tree).equal is True


def test_state_limit_reports_unknown_instead_of_guessing():
    assert compare(parse("(a|b)*a(a|b){8}"), parse("(a|b)*b(a|b){8}"), max_states=5).equal is None


@pytest.mark.parametrize(
    ("text", "risky"),
    [
        (r"(a+b?)+", True),
        (r"(\d|[0-9a-f])+", True),
        (r"(a{1,3}b)*", True),
        (r"([a-z]+-)*[a-z]+", True),
        (r"[a-z]+-\d+", False),
        (r"(ab)+", False),
        (r"(a{3}b)+", False),
        (r"(a|b){3}", False),
    ],
)
def test_backtracking_risk(text, risky):
    assert may_backtrack_badly(parse(text)) is risky
