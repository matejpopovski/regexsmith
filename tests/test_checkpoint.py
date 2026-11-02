"""The repository ships a model, so a clean clone can solve tasks without training."""

from regexsmith.cli import DEFAULT_CHECKPOINT
from regexsmith.dsl import parse
from regexsmith.executor import RegexRunner
from regexsmith.model import count_parameters
from regexsmith.search import SearchConfig, synthesize
from regexsmith.semantics import compare
from regexsmith.train import load_model


def test_checkpoint_is_small_enough_to_commit():
    assert DEFAULT_CHECKPOINT.stat().st_size < 20 * 1024 * 1024


def test_shipped_model_solves_an_easy_task_on_cpu():
    model = load_model(DEFAULT_CHECKPOINT, "cpu")
    assert count_parameters(model) > 1_000_000
    positives = ["2024-01-31", "1999-12-01", "2031-07-15", "1987-03-09", "2005-11-22"]
    negatives = ["2024-1-31", "24-01-31", "2024/01/31", "2024-01-3", "abcd-ef-gh", "2024-01-311"]
    with RegexRunner(timeout=0.5) as runner:
        result = synthesize(
            model, positives, negatives, SearchConfig(mode="guided", budget=20.0), runner
        )
    assert compare(result.best, parse(r"\d{4}-\d{2}-\d{2}")).equal is True
