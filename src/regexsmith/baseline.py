"""Bottom-up enumerative synthesis, the opponent the neural search has to beat.

The easy baseline would call ``re`` on every candidate and lose. This one is meant to
be hard to beat, so it adopts the characteristic-vector idea from the literature on
exhaustive regex inference (Valizadeh and Berger, PLDI 2023). A regex is represented by
one bit per substring of the examples, saying whether the regex matches that substring.
Concatenation, alternation and star can all be computed on those vectors alone, which
gives two things:

* No backtracking, so a candidate costs the same whatever its shape.
* Exact observational equivalence: two regexes with one vector behave identically in
  every context over these examples, so only the smallest needs keeping. The search
  space collapses by orders of magnitude.

Candidates are enumerated in order of token count over the same grammar the neural
decoder uses, so the first solution found has minimal size. Vectors are processed in
NumPy batches.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .dsl import ALT, CAT, CLASSES, MAX_TOKENS, QUANTIFIERS, Node, cls, lit

_ATOM, _CAT, _ALT, _QUANT = range(4)
_MAX_POWER = max(high or low for low, high in QUANTIFIERS.values())
_RESERVE_BYTES = 400_000_000
_WORK_PER_CHUNK = 3_000_000


class _Infixes:
    """All substrings of the examples and the ways each one splits in two."""

    def __init__(self, positives: Sequence[str], negatives: Sequence[str]) -> None:
        strings = list(positives) + list(negatives)
        infixes = {s[i:j] for s in strings for i in range(len(s) + 1) for j in range(i, len(s) + 1)}
        infixes.add("")
        self.items = sorted(infixes, key=lambda s: (len(s), s))
        index = {text: i for i, text in enumerate(self.items)}
        self.count = len(self.items)
        self.empty = index[""]
        self.positives = np.array([index[s] for s in positives], dtype=np.int64)
        self.negatives = np.array([index[s] for s in negatives], dtype=np.int64)
        positive_infixes = {
            s[i:j] for s in positives for i in range(len(s) + 1) for j in range(i, len(s) + 1)
        }
        self.positive_infixes = np.array(sorted(index[s] for s in positive_infixes), dtype=np.int64)

        lefts, rights, starts = [], [], []
        for text in self.items:
            starts.append(len(lefts))
            for cut in range(len(text) + 1):
                lefts.append(index[text[:cut]])
                rights.append(index[text[cut:]])
        self.left = np.array(lefts, dtype=np.int64)
        self.right = np.array(rights, dtype=np.int64)
        self.starts = np.array(starts, dtype=np.int64)

        # Star needs proper prefixes only, grouped by length so a row can use shorter rows.
        by_length: dict[int, list[int]] = {}
        for i, text in enumerate(self.items):
            by_length.setdefault(len(text), []).append(i)
        self.star_steps = []
        for length in sorted(by_length):
            if length == 0:
                continue
            wholes = by_length[length]
            step_left, step_right, step_starts = [], [], []
            for whole in wholes:
                text = self.items[whole]
                step_starts.append(len(step_left))
                for cut in range(1, len(text) + 1):
                    step_left.append(index[text[:cut]])
                    step_right.append(index[text[cut:]])
            self.star_steps.append(
                (
                    np.array(wholes, dtype=np.int64),
                    np.array(step_left, dtype=np.int64),
                    np.array(step_right, dtype=np.int64),
                    np.array(step_starts, dtype=np.int64),
                )
            )

    def concat(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        hits = a[:, self.left] & b[:, self.right]
        return np.logical_or.reduceat(hits, self.starts, axis=1)

    def star(self, a: np.ndarray) -> np.ndarray:
        out = np.zeros_like(a)
        out[:, self.empty] = True
        for wholes, left, right, starts in self.star_steps:
            hits = a[:, left] & out[:, right]
            out[:, wholes] = np.logical_or.reduceat(hits, starts, axis=1)
        return out


@dataclass(slots=True)
class EnumerationResult:
    regex: Node | None
    elapsed: float
    candidates: int
    distinct: int
    max_cost_finished: int


class _Solved(Exception):
    def __init__(self, entry: tuple[str, int, int]) -> None:
        self.entry = entry


class _OutOfTime(Exception):
    pass


class Enumerator:
    def __init__(
        self,
        positives: Sequence[str],
        negatives: Sequence[str],
        budget: float,
        max_cost: int = MAX_TOKENS,
    ) -> None:
        self.infixes = _Infixes(positives, negatives)
        self.deadline = time.perf_counter() + budget
        self.start = time.perf_counter()
        self.max_cost = max_cost
        self.candidates = 0

        n = self.infixes.count
        self.chunk = max(8, _WORK_PER_CHUNK // max(len(self.infixes.left), 1))
        self.words = (n + 63) // 64
        self.hash_weights = np.random.default_rng(0).integers(1, 2**63, self.words, dtype=np.uint64)
        # A quantifier may not sit on a quantifier, so a non-quantified regex can do things
        # an equivalent quantified one cannot: (aa)+ is legal and (a{2})+ is not. Plain
        # regexes are therefore only compared with each other. This matters: without it
        # a{2} shadows aa and the enumerator stops being minimal.
        self.seen_plain: set[int] = set()
        self.seen_any: set[int] = set()
        self.capacity = max(_RESERVE_BYTES // (self.words * 8), 1)

        # Global entry tables, plus the packed vectors of each finished cost level.
        self.tokens: list[str] = []
        self.left_ids: list[int] = []
        self.right_ids: list[int] = []
        self.kinds: list[int] = []
        self.levels: dict[int, np.ndarray] = {}
        self.level_ids: dict[int, np.ndarray] = {}
        self.level_kinds: dict[int, np.ndarray] = {}
        self._rows: list[np.ndarray] = []
        self._ids: list[int] = []

        chars = sorted({c for s in positives for c in s})
        self.atoms = [lit(c) for c in chars] + [cls(name) for name in CLASSES]

    def _packed(self, matrix: np.ndarray) -> np.ndarray:
        """Bit-pack each row, padded to whole 64-bit words so rows can be hashed as words."""
        bits = np.packbits(matrix, axis=1)
        return np.pad(bits, ((0, 0), (0, self.words * 8 - bits.shape[1])))

    def _unpacked(self, packed: np.ndarray) -> np.ndarray:
        return np.unpackbits(packed, axis=1, count=self.infixes.count).astype(bool)

    def _emit(
        self,
        matrix: np.ndarray,
        token: str,
        kind: int,
        left: np.ndarray,
        right: np.ndarray | None,
    ) -> None:
        infixes = self.infixes
        self.candidates += len(matrix)
        solved = matrix[:, infixes.positives].all(axis=1)
        if len(infixes.negatives):
            solved &= ~matrix[:, infixes.negatives].any(axis=1)
        if solved.any():
            row = int(np.argmax(solved))
            raise _Solved((token, int(left[row]), -1 if right is None else int(right[row])))

        # A regex matching no substring of any positive cannot help in any context:
        # regex operators are monotone, so deleting it keeps every positive matched
        # and can only reject more negatives.
        useful = np.flatnonzero(matrix[:, infixes.positive_infixes].any(axis=1))
        if not len(useful):
            return
        packed = self._packed(matrix[useful])
        hashes = (packed.view(np.uint64) * self.hash_weights).sum(axis=1, dtype=np.uint64)
        seen = self.seen_any if kind == _QUANT else self.seen_plain
        fresh = []
        for position, value in enumerate(hashes.tolist()):
            if value not in seen:
                seen.add(value)
                self.seen_any.add(value)
                fresh.append(position)
        if not fresh or len(self.tokens) >= self.capacity:
            return
        rows = useful[fresh]
        first = len(self.tokens)
        self.tokens.extend([token] * len(rows))
        self.kinds.extend([kind] * len(rows))
        self.left_ids.extend(left[rows].tolist())
        self.right_ids.extend(right[rows].tolist() if right is not None else [-1] * len(rows))
        self._rows.append(packed[fresh])
        self._ids.extend(range(first, first + len(rows)))

    def _seal_level(self, cost: int) -> None:
        if self._rows:
            self.levels[cost] = np.concatenate(self._rows)
            self.level_ids[cost] = np.array(self._ids, dtype=np.int64)
            self.level_kinds[cost] = np.array([self.kinds[i] for i in self._ids], dtype=np.int8)
        self._rows = []
        self._ids = []

    def _tick(self) -> None:
        if time.perf_counter() > self.deadline:
            raise _OutOfTime

    def _enumerate_atoms(self) -> None:
        infixes = self.infixes
        for node in self.atoms:
            matrix = np.zeros((1, infixes.count), dtype=bool)
            if node.tok.startswith("lit:"):
                chars = {node.tok[4:]}
            else:
                chars = set(CLASSES[node.tok[4:]].chars)
            for i, text in enumerate(infixes.items):
                if len(text) > 1:
                    break
                if text in chars:
                    matrix[0, i] = True
            self._emit(matrix, node.tok, _ATOM, np.array([-1]), None)
        self._seal_level(1)

    def _quantified(self, cost: int) -> None:
        """Every quantifier over every entry of the cost below."""
        if cost - 1 not in self.levels:
            return
        keep = np.flatnonzero(self.level_kinds[cost - 1] != _QUANT)
        rows = self.levels[cost - 1][keep]
        ids = self.level_ids[cost - 1][keep]
        infixes = self.infixes
        step = max(self.chunk // 16, 1)
        for start in range(0, len(rows), step):
            self._tick()
            chunk_ids = ids[start : start + step]
            base = self._unpacked(rows[start : start + step])
            star = infixes.star(base)
            powers = [None, base]
            for _ in range(2, _MAX_POWER + 1):
                powers.append(infixes.concat(base, powers[-1]))
            plus = infixes.concat(base, star)
            optional = base.copy()
            optional[:, infixes.empty] = True
            for q, (low, high) in QUANTIFIERS.items():
                if q == "*":
                    matrix = star
                elif q == "+":
                    matrix = plus
                elif q == "?":
                    matrix = optional
                else:
                    matrix = powers[low]
                    for power in range(low + 1, (high or low) + 1):
                        matrix = matrix | powers[power]
                self._emit(matrix, f"q:{q}", _QUANT, chunk_ids, None)

    def _binary(self, cost: int, kind: int) -> None:
        token = CAT if kind == _CAT else ALT
        for left_cost in range(1, cost - 1):
            right_cost = cost - 1 - left_cost
            if left_cost not in self.levels or right_cost not in self.levels:
                continue
            keep = np.flatnonzero(self.level_kinds[left_cost] != kind)
            left_rows = self.levels[left_cost][keep]
            left_ids = self.level_ids[left_cost][keep]
            right_rows = self.levels[right_cost]
            right_ids = self.level_ids[right_cost]
            for l_start in range(0, len(left_rows), 64):
                l_ids = left_ids[l_start : l_start + 64]
                left = self._unpacked(left_rows[l_start : l_start + 64])
                span = max(self.chunk // len(l_ids), 1)
                for r_start in range(0, len(right_rows), span):
                    self._tick()
                    r_ids = right_ids[r_start : r_start + span]
                    right = self._unpacked(right_rows[r_start : r_start + span])
                    pair_left = np.repeat(np.arange(len(l_ids)), len(r_ids))
                    pair_right = np.tile(np.arange(len(r_ids)), len(l_ids))
                    a, b = left[pair_left], right[pair_right]
                    matrix = self.infixes.concat(a, b) if kind == _CAT else a | b
                    self._emit(matrix, token, kind, l_ids[pair_left], r_ids[pair_right])

    def _build(self, token: str, left: int, right: int) -> Node:
        if left == -1:
            return Node(token)
        kids = [self._entry(left)]
        if right != -1:
            kids.append(self._entry(right))
        return Node(token, tuple(kids))

    def _entry(self, index: int) -> Node:
        return self._build(self.tokens[index], self.left_ids[index], self.right_ids[index])

    def run(self) -> EnumerationResult:
        finished = 0
        answer = None
        try:
            self._enumerate_atoms()
            finished = 1
            for cost in range(2, self.max_cost + 1):
                self._quantified(cost)
                self._binary(cost, _CAT)
                self._binary(cost, _ALT)
                self._seal_level(cost)
                finished = cost
        except _Solved as solved:
            answer = self._build(*solved.entry)
        except _OutOfTime:
            pass
        return EnumerationResult(
            regex=answer,
            elapsed=time.perf_counter() - self.start,
            candidates=self.candidates,
            distinct=len(self.tokens),
            max_cost_finished=finished,
        )


def enumerate_regex(
    positives: Sequence[str], negatives: Sequence[str], budget: float
) -> EnumerationResult:
    return Enumerator(positives, negatives, budget).run()
