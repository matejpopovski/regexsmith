import random

import pytest

from regexsmith.dsl import parse, size, to_regex
from regexsmith.field import FIELD_TARGETS
from regexsmith.grammar import is_canonical
from regexsmith.semantics import matches
from regexsmith.tasks import (
    MAX_STRING,
    TIERS,
    Task,
    is_heldout,
    make_task,
    sample_string,
    sample_target,
    sample_task,
)


@pytest.mark.parametrize("tier", list(TIERS))
def test_targets_land_in_their_tier(tier):
    rng = random.Random(tier)
    low, high = TIERS[tier]
    for _ in range(100):
        target = sample_target(rng, tier)
        assert low <= size(target) <= high
        assert is_canonical(target)
        assert parse(to_regex(target)) == target


@pytest.mark.parametrize("tier", list(TIERS))
def test_examples_are_labelled_correctly(tier):
    rng = random.Random(tier)
    for _ in range(25):
        task = sample_task(rng, tier, n_hidden=10)
        target = parse(task.target)
        for text in task.positives + task.hidden_positives:
            assert matches(target, text)
            assert len(text) <= MAX_STRING
        for text in task.negatives + task.hidden_negatives:
            assert not matches(target, text)
        assert not set(task.positives) & set(task.hidden_positives)
        assert not set(task.negatives) & set(task.hidden_negatives)


def test_splits_are_disjoint_by_target():
    rng = random.Random(0)
    for _ in range(200):
        assert not is_heldout(to_regex(sample_target(rng, 2)))
    for _ in range(40):
        assert is_heldout(to_regex(sample_target(rng, 2, heldout=True)))


def test_field_targets_are_never_sampled():
    field = {to_regex(parse(text)) for _, text in FIELD_TARGETS}
    rng = random.Random(0)
    assert not any(to_regex(sample_target(rng, 1 + i % 3)) in field for i in range(3000))


def test_generation_is_deterministic():
    assert sample_task(random.Random(5), 3) == sample_task(random.Random(5), 3)


def test_task_json_round_trip():
    task = sample_task(random.Random(1), 2, n_hidden=3)
    assert Task.from_json(task.to_json()) == task


def test_classes_show_their_rare_members():
    rng = random.Random(0)
    node = parse(r"\w{12}")
    assert any("_" in sample_string(rng, node) for _ in range(20))


def test_small_languages_keep_every_positive_visible():
    task = make_task(parse("c?"), random.Random(0), n_hidden=20)
    assert set(task.positives) == {"", "c"}


def test_a_positive_set_of_only_the_empty_string_is_refused():
    target = parse("(ab)?")
    for seed in range(40):
        task = make_task(target, random.Random(seed), n_pos=1)
        assert task is None or task.positives == ("ab",)
