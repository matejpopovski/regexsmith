import re

import pytest
from hypothesis import given

from regexsmith import dsl
from regexsmith.dsl import (
    ALPHABET,
    HOLE,
    Node,
    UnsupportedRegex,
    from_tokens,
    has_hole,
    lit,
    overapprox,
    parse,
    preorder,
    size,
    to_regex,
    underapprox,
)
from regexsmith.field import FIELD_TARGETS
from regexsmith.grammar import is_canonical
from regexsmith.semantics import compare, matches

from .strategies import texts, trees


@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        ("a", ["lit:a"]),
        (r"\d+", ["q:+", "cls:digit"]),
        ("ab", ["cat", "lit:a", "lit:b"]),
        ("a|b|cd", ["alt", "lit:a", "alt", "lit:b", "cat", "lit:c", "lit:d"]),
    ],
)
def test_preorder_tokens(text, tokens):
    assert preorder(parse(text)) == tokens


@pytest.mark.parametrize(
    "text",
    [
        "ab|cd",
        "(ab|cd)e",
        "(ab)+",
        "a(b|cd)?",
        r"\d{1,3}(\.\d{1,3}){3}",
        r"[ab\-]",
        r"[0-9a-f]{8}",
        r"\$\(x\)",
        r"[\-\]]",
    ],
)
def test_render_is_inverse_of_parse(text):
    assert to_regex(parse(text)) == text


@pytest.mark.parametrize(
    "text",
    ["^a", "a$", "(?=a)", "a{7}", "[^a]", "a**", "*a", "[d-b]", "(a", "a)", "é", "a||b", r"\s"],
)
def test_syntax_outside_the_grammar_is_rejected(text):
    with pytest.raises(UnsupportedRegex):
        parse(text)


def test_other_ranges_expand_to_their_members():
    assert to_regex(parse("[b-d]")) == "[bcd]"


def test_quantifier_on_a_quantifier_is_outside_the_grammar():
    with pytest.raises(UnsupportedRegex):
        parse("(a+)+")


@pytest.mark.parametrize("char", list(ALPHABET))
def test_every_character_round_trips(char):
    node = lit(char)
    pattern = to_regex(node)
    assert re.fullmatch(pattern, char)
    assert parse(pattern) == node
    assert re.fullmatch(to_regex(dsl.alt(node, lit("a" if char != "a" else "b"))), char)


@given(trees)
def test_rendered_regex_has_the_same_language_as_its_tree(tree):
    assert compare(parse(to_regex(tree)), tree).equal is True


@given(trees)
def test_tokens_rebuild_the_tree(tree):
    assert from_tokens(preorder(tree)) == tree


@given(trees)
def test_size_is_token_count(tree):
    assert size(tree) == len(preorder(tree))


def test_truncated_tokens_leave_holes():
    tree = from_tokens(["cat", "lit:a"])
    assert tree == Node("cat", (lit("a"), Node(HOLE)))
    assert has_hole(tree)


@pytest.mark.parametrize(
    ("tokens", "over", "under"),
    [
        (["cat", "lit:a"], "a.*", None),
        (["alt", "lit:a"], "a|.*", "a"),
        (["alt", "cat", "lit:a", "lit:b"], "ab|.*", "ab"),
        (["q:+"], ".*", None),
        (["q:+", "alt", "lit:a"], ".*", "a+"),
    ],
)
def test_sketch_approximations(tokens, over, under):
    tree = from_tokens(tokens)
    assert to_regex(overapprox(tree)) in (over, ".*")
    inner = underapprox(tree)
    assert (None if inner is None else to_regex(inner)) == under


@given(trees, texts)
def test_approximations_bracket_every_completion(tree, text):
    """The property that makes pruning safe: no sketch can exclude its own completion."""
    tokens = preorder(tree)
    accepted = matches(tree, text)
    for cut in range(1, len(tokens)):
        sketch = from_tokens(tokens[:cut])
        if accepted:
            assert matches(overapprox(sketch), text)
        inner = underapprox(sketch)
        if inner is not None and matches(inner, text):
            assert accepted


def test_field_targets_fit_the_grammar():
    for name, text in FIELD_TARGETS:
        assert is_canonical(parse(text)), name
