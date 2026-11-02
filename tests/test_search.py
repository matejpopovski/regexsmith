import random

import pytest
import torch

from regexsmith.data import collate
from regexsmith.dsl import MAX_TOKENS, TOKEN_ID, VOCAB, from_tokens, parse, preorder, size, to_regex
from regexsmith.executor import RegexRunner
from regexsmith.grammar import CONTEXT_MASK, DecodeState, is_canonical
from regexsmith.search import SearchConfig, SearchResult, _Beam, _TaskSearch, synthesize
from regexsmith.semantics import matches
from regexsmith.tasks import TIERS, sample_task
from regexsmith.train import loss_on


def test_top1_always_returns_a_legal_regex(tiny_model):
    rng = random.Random(0)
    for _ in range(5):
        task = sample_task(rng, 2, 4, 4)
        result = synthesize(tiny_model, task.positives, task.negatives, SearchConfig(mode="top1"))
        assert result.best is not None
        assert is_canonical(result.best)
        assert size(result.best) <= MAX_TOKENS


def test_executing_modes_require_a_runner(tiny_model):
    with pytest.raises(ValueError):
        synthesize(tiny_model, ["a"], ["b"], SearchConfig(mode="guided"))


@pytest.mark.parametrize("mode", ["filter", "guided"])
def test_reported_solutions_really_pass(tiny_model, mode):
    config = SearchConfig(mode=mode, budget=3.0)
    positives, negatives = ["aa", "aaa"], ["b", "ab"]
    with RegexRunner() as runner:
        result = synthesize(tiny_model, positives, negatives, config, runner)
    for candidate in result.solutions:
        assert all(matches(candidate.regex, s) for s in positives)
        assert not any(matches(candidate.regex, s) for s in negatives)


def test_budget_is_respected(tiny_model):
    task = sample_task(random.Random(9), 5)
    config = SearchConfig(mode="guided", budget=0.5)
    with RegexRunner() as runner:
        result = synthesize(tiny_model, task.positives, task.negatives, config, runner)
    assert result.elapsed < 3.0


def _sketch_is_kept(search, tokens):
    ids = tuple(TOKEN_ID[t] for t in tokens)
    return bool(search._viable([_Beam(ids, DecodeState(), 0.0)]))


@pytest.mark.parametrize("tier", list(TIERS))
def test_pruning_never_rejects_a_prefix_of_a_true_solution(tiny_model, tier):
    """The soundness claim, checked end to end through the real executor."""
    rng = random.Random(tier)
    with RegexRunner(timeout=1.0) as runner, torch.inference_mode():
        for _ in range(12):
            task = sample_task(rng, tier)
            search = _TaskSearch(
                tiny_model,
                task.positives,
                task.negatives,
                runner,
                True,
                0.0,
                float("inf"),
                SearchResult(),
            )
            tokens = preorder(parse(task.target))
            for cut in range(1, len(tokens)):
                assert _sketch_is_kept(search, tokens[:cut]), (task.target, tokens[:cut])


def test_pruning_rejects_a_dead_sketch(tiny_model):
    with RegexRunner() as runner, torch.inference_mode():
        search = _TaskSearch(
            tiny_model, ["abc", "abd"], ["ab"], runner, True, 0.0, float("inf"), SearchResult()
        )
        assert _sketch_is_kept(search, ["cat", "lit:a"])
        assert not _sketch_is_kept(search, ["cat", "lit:z"])
        assert not _sketch_is_kept(search, ["alt", "cat", "lit:a", "lit:b"])


def test_a_trained_tiny_model_solves_what_it_saw():
    """Overfit one task end to end: training, decoding and execution have to line up."""
    from regexsmith.model import ModelConfig, RegexTransformer

    torch.manual_seed(0)
    model = RegexTransformer(
        ModelConfig(
            d_model=48, heads=2, char_layers=1, set_layers=1, decoder_layers=2, ff_multiplier=2
        )
    )
    task = sample_task(random.Random(4), 2, 6, 6)
    batch = collate([task] * 8)
    table = torch.from_numpy(CONTEXT_MASK)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    model.train()
    for _ in range(150):
        loss = loss_on(model, batch, table)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    result = synthesize(model, task.positives, task.negatives, SearchConfig(mode="top1"))
    assert to_regex(result.best) == task.target


def test_vocabulary_has_no_duplicates():
    assert len(VOCAB) == len(set(VOCAB))


def test_rebuilding_from_tokens_handles_a_full_program():
    tree = parse(r"(ab|c)+\d?")
    assert from_tokens(preorder(tree)) == tree
