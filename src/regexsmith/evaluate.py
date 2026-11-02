"""The benchmark: every method, the same tasks, the same wall-clock budget per task.

Three grades, from weakest to strongest, because the gap between them is the story:

``consistent``   Classifies every visible example correctly. Any search can reach this
                 by memorising the positives, so on its own it proves little.
``generalizes``  Also classifies all hidden strings, which no method ever sees.
``exact``        Accepts precisely the target's language, decided by a DFA equivalence
                 check rather than by sampling.

Grading uses the derivative engine, so a pathological answer cannot stall the grader.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import random
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from .baseline import enumerate_regex
from .dsl import Node, parse, to_regex
from .executor import BOTH, RegexRunner, Verdict
from .field import FIELD_TARGETS
from .semantics import Algebra
from .tasks import TIERS, Task, make_task, sample_task

FIELD_TIER = 0
METHODS = ("top1", "filter", "guided", "enumerative", "enumerative-10x")


def build_benchmark(per_tier: int, seed: int = 2026) -> list[Task]:
    """Test tasks from the held-out split, plus the field suite as its own tier."""
    rng = random.Random(seed)
    tasks = [
        sample_task(rng, tier, n_hidden=20, heldout=True) for tier in TIERS for _ in range(per_tier)
    ]
    for name, regex in FIELD_TARGETS:
        task = None
        while task is None:
            task = make_task(parse(regex), rng, n_hidden=20, tier=FIELD_TIER, name=name)
        tasks.append(task)
    return tasks


def load_tasks(path: Path) -> list[Task]:
    return [Task.from_json(line) for line in path.read_text().splitlines()]


def save_tasks(tasks: Iterable[Task], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(task.to_json() + "\n" for task in tasks))


def grade(task: Task, answer: Node | None) -> dict:
    grades = {
        "regex": None,
        "consistent": False,
        "generalizes": False,
        "exact": False,
        "witness": None,
        "catastrophic": False,
    }
    if answer is None:
        return grades
    algebra = Algebra()
    term = algebra.compile(answer)
    consistent = all(algebra.matches(term, s) for s in task.positives) and not any(
        algebra.matches(term, s) for s in task.negatives
    )
    hidden_right = all(algebra.matches(term, s) for s in task.hidden_positives) and not any(
        algebra.matches(term, s) for s in task.hidden_negatives
    )
    comparison = algebra.compare(term, algebra.compile(parse(task.target)))
    grades.update(
        regex=to_regex(answer),
        consistent=consistent,
        generalizes=consistent and hidden_right,
        exact=consistent and comparison.equal is True,
        witness=comparison.witness,
    )
    return grades


def _confirm_with_re(task: Task, answer: Node | None, runner: RegexRunner) -> bool:
    """Every reported solution must also pass the real engine in the time limit."""
    if answer is None:
        return False
    verdict = runner.check([(to_regex(answer), BOTH)], task.positives, task.negatives)[0]
    return verdict is Verdict.PASS


def _enumerate_one(args: tuple[Task, float]) -> dict:
    task, budget = args
    outcome = enumerate_regex(task.positives, task.negatives, budget)
    with RegexRunner() as runner:
        confirmed = _confirm_with_re(task, outcome.regex, runner)
    record = grade(task, outcome.regex)
    record.update(
        time=outcome.elapsed if outcome.regex is not None else None,
        elapsed=outcome.elapsed,
        candidates=outcome.candidates,
        distinct=outcome.distinct,
        max_cost=outcome.max_cost_finished,
        confirmed=confirmed,
    )
    return record


def run_enumerative(tasks: list[Task], budget: float, jobs: int) -> list[dict]:
    work = [(task, budget) for task in tasks]
    if jobs <= 1:
        return [_enumerate_one(item) for item in work]
    with multiprocessing.get_context("spawn").Pool(jobs) as pool:
        return pool.map(_enumerate_one, work, chunksize=1)


def run_neural(
    tasks: list[Task],
    method: str,
    budget: float,
    model,
    log: Callable[[str], None] = print,
) -> list[dict]:
    from .search import SearchConfig, synthesize

    config = SearchConfig(mode=method, budget=budget)
    records = []
    with RegexRunner() as runner:
        for index, task in enumerate(tasks):
            result = synthesize(model, task.positives, task.negatives, config, runner)
            record = grade(task, result.best)
            record.update(
                time=result.time_to_first if method != "top1" else result.elapsed,
                elapsed=result.elapsed,
                width=result.widest_pass,
                pruned=result.pruned,
                queries=result.queries,
                timeouts=result.timeouts,
                confirmed=_confirm_with_re(task, result.best, runner),
            )
            records.append(record)
            if (index + 1) % 50 == 0:
                log(f"  {method}: {index + 1}/{len(tasks)}")
    return records


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """A 95% interval that behaves near 0 and 1, where most of these rates live."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(centre - spread, 0.0), min(centre + spread, 1.0))


@dataclass(frozen=True, slots=True)
class Rate:
    successes: int
    total: int

    def to_dict(self) -> dict:
        low, high = wilson(self.successes, self.total)
        rate = self.successes / self.total if self.total else 0.0
        return {
            "successes": self.successes,
            "total": self.total,
            "rate": round(rate, 4),
            "ci95": [round(low, 4), round(high, 4)],
        }


def summarize(records: list[dict]) -> dict:
    summary: dict = {}
    for method in sorted({r["method"] for r in records}):
        own = [r for r in records if r["method"] == method]
        per_tier = {}
        for tier in sorted({r["tier"] for r in own}):
            rows = [r for r in own if r["tier"] == tier]
            times = sorted(r["time"] for r in rows if r["regex"] is not None)
            per_tier[str(tier)] = {
                grade_name: Rate(sum(r[grade_name] for r in rows), len(rows)).to_dict()
                for grade_name in ("consistent", "generalizes", "exact")
            }
            per_tier[str(tier)]["median_seconds_when_answered"] = (
                round(times[len(times) // 2], 4) if times else None
            )
            per_tier[str(tier)]["timeouts"] = sum(r.get("timeouts", 0) for r in rows)
        summary[method] = per_tier
    return summary


def write_results(records: list[dict], out_dir: Path, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "records.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    summary = {"meta": meta, "methods": summarize(records)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def run_benchmark(
    tasks: list[Task],
    methods: list[str],
    budget: float,
    checkpoint: Path,
    device: str,
    jobs: int,
    out_dir: Path,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Run methods, merging into any results already in ``out_dir``."""
    existing: list[dict] = []
    records_path = out_dir / "records.jsonl"
    if records_path.exists():
        existing = [json.loads(line) for line in records_path.read_text().splitlines()]
        existing = [r for r in existing if r["method"] not in methods]

    records = list(existing)
    model = None
    for method in methods:
        started = time.perf_counter()
        log(f"{method}: {len(tasks)} tasks at {budget:g} s")
        if method.startswith("enumerative"):
            scale = 10 if method.endswith("10x") else 1
            rows = run_enumerative(tasks, budget * scale, jobs)
        else:
            if model is None:
                from .train import load_model

                model = load_model(checkpoint, device)
            rows = run_neural(tasks, method, budget, model, log)
        for index, (task, row) in enumerate(zip(tasks, rows, strict=True)):
            row.update(
                method=method, task=index, tier=task.tier, target=task.target, name=task.name
            )
        records.extend(rows)
        log(f"{method}: finished in {time.perf_counter() - started:.0f} s")
        meta = {"budget_seconds": budget, "tasks": len(tasks), "device": device}
        write_results(records, out_dir, meta)
    return records
