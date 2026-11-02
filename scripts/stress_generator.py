"""Generate tasks flat out and report the slowest, to show the data pipeline cannot stall.

A 3 second alarm around each task turns a hang into a failure with the regex attached.
Writes results/generator_stress.json.

    uv run python scripts/stress_generator.py
"""

from __future__ import annotations

import json
import multiprocessing
import random
import signal
import time
from pathlib import Path

from regexsmith.dsl import to_regex
from regexsmith.tasks import make_task, sample_target

SECONDS = 90
PROCESSES = 6
TASK_LIMIT = 3


class TooSlow(Exception):
    pass


def _alarm(*_args: object) -> None:
    raise TooSlow


def work(seed: int) -> dict:
    signal.signal(signal.SIGALRM, _alarm)
    rng = random.Random(seed)
    started = time.time()
    tasks = 0
    slowest = (0.0, "")
    while time.time() - started < SECONDS:
        tier = rng.choice((3, 4, 5, 5))
        target = sample_target(rng, tier)
        before = time.perf_counter()
        signal.setitimer(signal.ITIMER_REAL, TASK_LIMIT)
        try:
            make_task(target, rng, rng.randint(4, 10), rng.randint(4, 10))
        except TooSlow:
            return {"hung_on": to_regex(target), "tasks": tasks}
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        tasks += 1
        slowest = max(slowest, (time.perf_counter() - before, to_regex(target)))
    return {"tasks": tasks, "slowest_seconds": round(slowest[0], 4), "slowest_target": slowest[1]}


def main() -> None:
    with multiprocessing.get_context("spawn").Pool(PROCESSES) as pool:
        workers = pool.map(work, range(PROCESSES))
    report = {
        "tiers": [3, 4, 5],
        "tasks": sum(w["tasks"] for w in workers),
        "hangs": sum("hung_on" in w for w in workers),
        "slowest_seconds": max(w.get("slowest_seconds", 0.0) for w in workers),
        "workers": workers,
    }
    out = Path(__file__).resolve().parents[1] / "results" / "generator_stress.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
