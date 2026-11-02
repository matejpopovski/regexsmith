import random
import re

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings

from regexsmith.baseline import Enumerator, _Infixes, enumerate_regex
from regexsmith.dsl import CAT, parse, size, to_regex
from regexsmith.grammar import is_canonical
from regexsmith.semantics import compare, matches
from regexsmith.tasks import sample_task

from .strategies import texts, trees


def _vector(tree, infixes):
    pattern = re.compile(to_regex(tree))
    return np.array([[bool(pattern.fullmatch(s)) for s in infixes.items]])


@settings(max_examples=120, deadline=None)
@given(trees, trees, st.lists(texts, min_size=1, max_size=4))
def test_vector_operations_match_the_real_engine(a, b, strings):
    infixes = _Infixes(strings, [])
    left, right = _vector(a, infixes), _vector(b, infixes)
    assert np.array_equal(
        infixes.concat(left, right), _vector(parse(f"({to_regex(a)})({to_regex(b)})"), infixes)
    )


@settings(max_examples=60, deadline=None)
@given(trees, st.lists(texts, min_size=1, max_size=4))
def test_star_matches_the_real_engine(tree, strings):
    infixes = _Infixes(strings, [])
    pattern = re.compile(f"(?:{to_regex(tree)})*")
    expected = np.array([[bool(pattern.fullmatch(s)) for s in infixes.items]])
    assert np.array_equal(infixes.star(_vector(tree, infixes)), expected)


@pytest.mark.parametrize(
    ("positives", "negatives", "expected_size"),
    [
        (["a"], ["b"], 1),
        (["1", "22", "333"], ["", "a"], 2),
        (["ab1", "cd2"], ["ab", "1"], 2),
        (["aa", "aaaa"], ["a", "aaa"], 4),
    ],
)
def test_finds_a_smallest_solution(positives, negatives, expected_size):
    outcome = enumerate_regex(positives, negatives, budget=10.0)
    assert outcome.regex is not None
    assert size(outcome.regex) == expected_size
    assert all(matches(outcome.regex, s) for s in positives)
    assert not any(matches(outcome.regex, s) for s in negatives)


def test_observational_equivalence_prunes_hard():
    task = sample_task(random.Random(2), 5)
    outcome = enumerate_regex(task.positives, task.negatives, budget=0.5)
    assert outcome.distinct < outcome.candidates / 3


def test_stays_inside_the_grammar():
    rng = random.Random(0)
    for _ in range(25):
        task = sample_task(rng, 2)
        outcome = enumerate_regex(task.positives, task.negatives, budget=0.5)
        if outcome.regex is not None:
            assert is_canonical(outcome.regex)
            assert outcome.regex.tok != CAT or outcome.regex.kids[0].tok != CAT


def test_recovers_the_target_when_examples_pin_it_down():
    rng = random.Random(3)
    task = sample_task(rng, 1)
    outcome = enumerate_regex(task.positives, task.negatives, budget=2.0)
    assert compare(outcome.regex, parse(task.target)).equal is True


def test_respects_the_budget():
    rng = random.Random(1)
    task = sample_task(rng, 5)
    outcome = Enumerator(task.positives, task.negatives, budget=0.3).run()
    assert outcome.elapsed < 1.5
