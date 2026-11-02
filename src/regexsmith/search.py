"""Beam search over regex trees, with and without execution.

Three modes, which are the rows of the ablation:

``top1``    Beam search with the grammar mask, returning the most probable finished regex.
            Nothing is executed. This is what the model knows on its own.
``filter``  Candidates are executed only once complete. The beam is widened until time
            runs out or something passes.
``guided``  As ``filter``, and every partial program is executed as well. A sketch is
            dropped if no way of filling its holes could match all positives, or if every
            way of filling them would match a negative. Those two checks come from
            ``overapprox`` and ``underapprox``, so a pruned sketch has no solution inside
            it: pruning spends beam slots better without ever losing a regex that a
            wider beam would have found.

``filter`` and ``guided`` restart with double the width rather than running one large beam.
A narrow pass is cheap and usually enough, and it makes the search anytime: whatever the
budget, it has spent it on the widest pass that fit.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from .data import encode_examples
from .dsl import (
    ARITY,
    BOS_ID,
    MAX_TOKENS,
    VOCAB,
    Node,
    from_tokens,
    overapprox,
    to_regex,
    underapprox,
)
from .executor import BOTH, MUST_ACCEPT, MUST_REJECT, BudgetExceeded, RegexRunner, Verdict
from .grammar import DecodeState
from .model import DecodeCache, HoleInfo, RegexTransformer, legal_logits

Mode = Literal["top1", "filter", "guided"]

# How far past the beam width to look for survivors at one depth, as a multiple of width.
_MAX_OVERDRAW = 4
# Filling holes with .* is what makes sketches slow, and a sketch that times out is kept
# anyway, so waiting long for a verdict buys nothing. The first benchmark run used the
# 250 ms limit of finished regexes here, and one task spent its whole budget on timeouts.
_SKETCH_TIMEOUT = 0.05


@dataclass(slots=True)
class SearchConfig:
    mode: Mode = "guided"
    budget: float = 5.0
    start_width: int = 16
    max_width: int = 4096
    top1_width: int = 32


@dataclass(frozen=True, slots=True)
class Candidate:
    regex: Node
    log_prob: float


@dataclass(slots=True)
class SearchResult:
    solutions: list[Candidate] = field(default_factory=list)
    elapsed: float = 0.0
    time_to_first: float | None = None
    widest_pass: int = 0
    pruned: int = 0
    queries: int = 0
    timeouts: int = 0
    finished: int = 0

    @property
    def best(self) -> Node | None:
        return self.solutions[0].regex if self.solutions else None


@dataclass(slots=True)
class _Beam:
    tokens: tuple[int, ...]
    state: DecodeState
    log_prob: float
    parent: int = 0

    def tree(self) -> Node:
        return from_tokens([VOCAB[t] for t in self.tokens])


class _OutOfTime(Exception):
    pass


class _TaskSearch:
    """Beam-search passes over one task, sharing the encoder output and a verdict cache."""

    def __init__(
        self,
        model: RegexTransformer,
        positives: Sequence[str],
        negatives: Sequence[str],
        runner: RegexRunner | None,
        guided: bool,
        started: float,
        deadline: float,
        result: SearchResult,
    ) -> None:
        self.model = model
        self.device = next(model.parameters()).device
        chars, labels = encode_examples([(positives, negatives)])
        self.memory = model.encode(chars.to(self.device), labels.to(self.device))
        self.positives = positives
        self.negatives = negatives
        self.runner = runner
        self.guided = guided
        self.deadline = deadline
        self.result = result
        self.started = started
        self.verdicts: dict[tuple[str, int], bool] = {}
        self.found: list[Candidate] = []

    def _execute(self, queries: list[tuple[str, int]], timeout: float | None = None) -> None:
        fresh = [q for q in dict.fromkeys(queries) if q not in self.verdicts]
        if not fresh:
            return
        try:
            answers = self.runner.check(
                fresh, self.positives, self.negatives, timeout, self.deadline
            )
        except BudgetExceeded:
            raise _OutOfTime from None
        self.result.queries += len(fresh)
        for query, verdict in zip(fresh, answers, strict=True):
            self.result.timeouts += verdict is Verdict.TIMEOUT
            # A sketch that times out might still be fine, so it stays in the beam and its
            # completions have to prove themselves. A finished regex that cannot decide the
            # examples in time is rejected.
            self.verdicts[query] = verdict is Verdict.PASS or (
                verdict is Verdict.TIMEOUT and query[1] != BOTH
            )

    def _sketch_queries(self, beam: _Beam) -> list[tuple[str, int]]:
        """Bounds on what the sketch can become.

        A token with holes of its own only replaces a hole with more holes, which leaves
        both approximations unchanged, so only leaf tokens need running.
        """
        if ARITY[beam.tokens[-1]] != 0:
            return []
        tree = beam.tree()
        queries = [(to_regex(overapprox(tree)), MUST_ACCEPT)]
        inner = underapprox(tree)
        if inner is not None:
            queries.append((to_regex(inner), MUST_REJECT))
        return queries

    def _viable(self, children: list[_Beam]) -> list[_Beam]:
        queries = [self._sketch_queries(child) for child in children]
        self._execute([q for group in queries for q in group], _SKETCH_TIMEOUT)
        kept = [
            child
            for child, group in zip(children, queries, strict=True)
            if all(self.verdicts[q] for q in group)
        ]
        self.result.pruned += len(children) - len(kept)
        return kept

    def _accept(self, finished: list[_Beam]) -> None:
        self.result.finished += len(finished)
        candidates = [Candidate(b.tree(), b.log_prob) for b in finished]
        if self.runner is None:
            self.found.extend(candidates)
            return
        queries = [(to_regex(c.regex), BOTH) for c in candidates]
        self._execute(queries)
        if time.perf_counter() > self.deadline:
            raise _OutOfTime
        for candidate, query in zip(candidates, queries, strict=True):
            if self.verdicts[query]:
                self.found.append(candidate)
                if self.result.time_to_first is None:
                    self.result.time_to_first = time.perf_counter() - self.started

    def _expand(
        self, beams: list[_Beam], scores: Tensor, pool: int, width: int
    ) -> tuple[list[_Beam], list[_Beam]]:
        """Walk the best expansions in order until ``width`` of them are worth keeping."""
        vocabulary = scores.shape[-1]
        top_scores, top_index = scores.flatten().topk(pool)
        ranked = list(zip(top_scores.cpu().tolist(), top_index.cpu().tolist(), strict=True))
        live: list[_Beam] = []
        finished: list[_Beam] = []
        for start in range(0, pool, width):
            if len(live) >= width:
                break
            children = []
            for score, flat in ranked[start : start + width]:
                parent, token = divmod(flat, vocabulary)
                beam = beams[parent]
                child = _Beam((*beam.tokens, token), beam.state.push(token), score, parent)
                (finished if child.state.done else children).append(child)
            live.extend(self._viable(children) if self.guided else children)
        return live[:width], finished

    def _decoder_input(
        self, cache: DecodeCache, beams: list[_Beam], last_tokens: list[int]
    ) -> tuple[DecodeCache, Tensor, HoleInfo]:
        def column(values: list[int]) -> Tensor:
            return torch.tensor(values, device=self.device).unsqueeze(1)

        return (
            cache,
            column(last_tokens),
            HoleInfo(
                column([b.state.stack[-1] for b in beams]),
                column([len(b.state.stack) for b in beams]),
            ),
        )

    def run(self, width: int) -> None:
        cache = self.model.start_decoding(self.memory)
        beams = [_Beam((), DecodeState(), 0.0)]
        last_tokens = [BOS_ID]
        for _ in range(MAX_TOKENS):
            if time.perf_counter() > self.deadline:
                raise _OutOfTime
            logits, cache = self.model.step(*self._decoder_input(cache, beams, last_tokens))
            legal = torch.from_numpy(np.stack([b.state.legal() for b in beams])).to(self.device)
            totals = torch.tensor([b.log_prob for b in beams], device=self.device)
            scores = legal_logits(logits, legal) + totals.unsqueeze(1)
            pool = min(width * _MAX_OVERDRAW, int(legal.sum()))
            live, finished = self._expand(beams, scores, pool, width)
            self._accept(finished)
            if not live:
                break
            # Scores only fall as a beam grows, so nothing alive can overtake the best
            # solution once it has dropped below it.
            if self.found and max(c.log_prob for c in self.found) >= live[0].log_prob:
                break
            cache = cache.select(torch.tensor([b.parent for b in live], device=self.device))
            last_tokens = [b.tokens[-1] for b in live]
            beams = live


@torch.inference_mode()
def synthesize(
    model: RegexTransformer,
    positives: Sequence[str],
    negatives: Sequence[str],
    config: SearchConfig,
    runner: RegexRunner | None = None,
) -> SearchResult:
    """Search for a regex. ``runner`` is required by the modes that execute candidates."""
    if config.mode != "top1" and runner is None:
        raise ValueError(f"mode {config.mode} executes candidates and needs a RegexRunner")
    started = time.perf_counter()
    result = SearchResult()
    deadline = started + config.budget if config.mode != "top1" else float("inf")
    search = _TaskSearch(
        model,
        positives,
        negatives,
        None if config.mode == "top1" else runner,
        config.mode == "guided",
        started,
        deadline,
        result,
    )
    if config.mode == "top1":
        result.widest_pass = config.top1_width
        search.run(config.top1_width)
    else:
        width = config.start_width
        try:
            while width <= config.max_width and not search.found:
                result.widest_pass = width
                search.run(width)
                width *= 2
        except _OutOfTime:
            pass
    ranked = sorted(search.found, key=lambda c: -c.log_prob)
    result.solutions = ranked[:1] if config.mode == "top1" else ranked
    result.elapsed = time.perf_counter() - started
    return result
