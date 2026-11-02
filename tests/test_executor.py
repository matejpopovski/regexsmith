import time

import pytest

from regexsmith.executor import (
    BOTH,
    MUST_ACCEPT,
    MUST_REJECT,
    BudgetExceeded,
    RegexRunner,
    Verdict,
)

EVIL = r"(a|a)*b"
SLOW_INPUT = "a" * 40


def test_verdicts_follow_the_modes():
    with RegexRunner() as runner:
        verdicts = runner.check(
            [("a+", BOTH), ("a*", BOTH), ("a*", MUST_ACCEPT), ("b", MUST_REJECT), ("(", BOTH)],
            ["a", "aa"],
            ["", "b"],
        )
    assert verdicts == [Verdict.PASS, Verdict.FAIL, Verdict.PASS, Verdict.FAIL, Verdict.ERROR]


def test_catastrophic_backtracking_is_cut_off():
    with RegexRunner(timeout=0.2) as runner:
        started = time.perf_counter()
        verdict = runner.check([(EVIL, BOTH)], [SLOW_INPUT], [])[0]
        elapsed = time.perf_counter() - started
    assert verdict is Verdict.TIMEOUT
    assert elapsed < 2.0
    assert runner.timeouts == 1


def test_the_rest_of_a_batch_survives_a_hang():
    queries = [("a+", BOTH), (EVIL, BOTH), ("a{40}", BOTH), (EVIL, BOTH), ("b", BOTH)]
    with RegexRunner(timeout=0.2) as runner:
        verdicts = runner.check(queries, [SLOW_INPUT], ["b"])
    assert verdicts == [
        Verdict.PASS,
        Verdict.TIMEOUT,
        Verdict.PASS,
        Verdict.TIMEOUT,
        Verdict.FAIL,
    ]


def test_a_shorter_limit_applies_to_one_call():
    with RegexRunner(timeout=30.0) as runner:
        started = time.perf_counter()
        verdict = runner.check([(EVIL, BOTH)], [SLOW_INPUT], [], timeout=0.1)[0]
    assert verdict is Verdict.TIMEOUT
    assert time.perf_counter() - started < 5.0


def test_the_overall_deadline_stops_a_batch_of_slow_patterns():
    queries = [(EVIL + "c" * i, BOTH) for i in range(20)]
    with RegexRunner(timeout=0.2) as runner:
        started = time.perf_counter()
        with pytest.raises(BudgetExceeded):
            runner.check(queries, [SLOW_INPUT], [], deadline=started + 0.5)
        assert time.perf_counter() - started < 1.5
        assert runner.check([("a+", BOTH)], ["a"], [])[0] is Verdict.PASS


def test_batches_larger_than_a_pipe_buffer():
    queries = [("x" * 500 + str(i), BOTH) for i in range(2000)]
    with RegexRunner() as runner:
        verdicts = runner.check(queries, ["y"], [])
    assert verdicts == [Verdict.FAIL] * len(queries)


def test_close_is_idempotent_and_reopens():
    runner = RegexRunner()
    assert runner.check([("a", BOTH)], ["a"], [])[0] is Verdict.PASS
    runner.close()
    runner.close()
    assert runner.check([("a", BOTH)], ["a"], [])[0] is Verdict.PASS
    runner.close()
