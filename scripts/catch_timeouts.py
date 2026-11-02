"""Replay the benchmark tasks where guided search hit the regex timeout, and name the patterns.

Writes results/timeout_examples.json. The patterns are real candidates from the search,
which is the case for having the executor at all.

    uv run python scripts/catch_timeouts.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from regexsmith.cli import DEFAULT_CHECKPOINT
from regexsmith.evaluate import load_tasks
from regexsmith.executor import RegexRunner, Verdict
from regexsmith.search import SearchConfig, synthesize
from regexsmith.train import load_model

ROOT = Path(__file__).resolve().parents[1]


class RecordingRunner(RegexRunner):
    def __init__(self) -> None:
        super().__init__()
        self.hung: list[str] = []

    def check(self, queries, *args, **kwargs):
        verdicts = super().check(queries, *args, **kwargs)
        self.hung += [
            pattern
            for (pattern, _), verdict in zip(queries, verdicts, strict=True)
            if verdict is Verdict.TIMEOUT
        ]
        return verdicts


def main() -> None:
    torch.set_num_threads(1)
    records = [json.loads(line) for line in (ROOT / "results/records.jsonl").open()]
    replay = [r for r in records if r["method"] == "guided" and r["timeouts"]]
    tasks = load_tasks(ROOT / "data/benchmark.jsonl")
    model = load_model(DEFAULT_CHECKPOINT, "cpu")
    report = []
    for record in replay:
        task = tasks[record["task"]]
        with RecordingRunner() as runner:
            synthesize(model, task.positives, task.negatives, SearchConfig(budget=5.0), runner)
        report.append(
            {
                "task": record["task"],
                "name": task.name,
                "target": task.target,
                "timeouts_in_benchmark": record["timeouts"],
                "patterns": list(dict.fromkeys(runner.hung))[:5],
            }
        )
    (ROOT / "results/timeout_examples.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
