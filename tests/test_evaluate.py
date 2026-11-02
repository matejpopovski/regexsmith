import json
import random

import pytest

from regexsmith.dsl import parse
from regexsmith.evaluate import (
    build_benchmark,
    grade,
    load_tasks,
    run_enumerative,
    run_neural,
    save_tasks,
    summarize,
    wilson,
)
from regexsmith.field import FIELD_TARGETS
from regexsmith.tasks import TIERS, Task, is_heldout, sample_task

TASK = Task(
    target=r"\d+",
    tier=1,
    positives=("1", "22"),
    negatives=("a", ""),
    hidden_positives=("333",),
    hidden_negatives=("4a",),
)


def test_exact_answer_gets_every_grade():
    grades = grade(TASK, parse("[0-9]+"))
    assert grades["consistent"]
    assert grades["generalizes"]
    assert grades["exact"]
    assert grades["witness"] is None


def test_overfit_answer_is_consistent_only():
    grades = grade(TASK, parse("1|22"))
    assert grades["consistent"]
    assert not grades["generalizes"]
    assert not grades["exact"]
    assert grades["witness"] is not None


def test_generalizing_is_not_exact():
    grades = grade(TASK, parse(r"\d{1,3}"))
    assert grades["consistent"]
    assert grades["generalizes"]
    assert not grades["exact"]


def test_inconsistent_answer_earns_nothing_even_if_right_on_hidden_strings():
    grades = grade(TASK, parse(r"\d"))
    assert not grades["consistent"]
    assert not grades["generalizes"]


def test_no_answer():
    assert grade(TASK, None)["regex"] is None


def test_wilson_interval():
    low, high = wilson(50, 100)
    assert 0.40 < low < 0.41
    assert 0.59 < high < 0.60
    assert wilson(0, 10)[0] == 0.0
    assert wilson(0, 0) == (0.0, 0.0)


def test_benchmark_is_held_out_and_includes_the_field_suite(tmp_path):
    tasks = build_benchmark(per_tier=3, seed=1)
    assert len(tasks) == 3 * len(TIERS) + len(FIELD_TARGETS)
    assert all(is_heldout(t.target) for t in tasks if t.tier)
    assert all(t.hidden_positives or t.tier <= 2 for t in tasks)
    path = tmp_path / "tasks.jsonl"
    save_tasks(tasks, path)
    assert load_tasks(path) == tasks


def test_benchmark_is_reproducible():
    assert build_benchmark(2, seed=3) == build_benchmark(2, seed=3)


def test_enumerative_records_are_confirmed_by_the_real_engine():
    tasks = [sample_task(random.Random(i), 1, n_hidden=5, heldout=True) for i in range(3)]
    records = run_enumerative(tasks, budget=1.0, jobs=1)
    assert all(r["confirmed"] for r in records if r["regex"])


@pytest.mark.parametrize("method", ["top1", "guided"])
def test_neural_records_have_the_same_shape(tiny_model, method):
    tasks = [sample_task(random.Random(0), 1, 4, 4, n_hidden=5, heldout=True)]
    record = run_neural(tasks, method, 0.5, tiny_model, log=lambda _: None)[0]
    record.update(method=method, tier=1)
    json.dumps(record)
    assert summarize([record])[method]["1"]["exact"]["total"] == 1
