"""Language-level reasoning with Brzozowski derivatives.

This is the ground truth the evaluation stands on. Matching here cannot backtrack,
and two regexes can be compared as languages rather than on a sample of strings:
``equivalent`` decides whether the synthesized regex is the target, and when it is
not, returns a shortest-ish string that tells them apart.

Terms are interned as small integers, so derivatives memoize cheaply and two equal
terms are one state. Alternation is kept associative, commutative and idempotent,
which is what makes the set of derivatives of a term finite.
"""

from __future__ import annotations

from dataclasses import dataclass

from .dsl import ALPHABET, CAT, CLASSES, QUANTIFIERS, Node

_CHAR_BIT = {char: 1 << i for i, char in enumerate(ALPHABET)}


def _mask(chars: str) -> int:
    bits = 0
    for char in chars:
        bits |= _CHAR_BIT[char]
    return bits


_CLASS_MASK = {name: _mask(c.chars) for name, c in CLASSES.items()}
_EMPTY, _EPS = 0, 1
_MAX_UNBOUNDED_RUNS = 4


@dataclass(frozen=True, slots=True)
class _Term:
    kind: str
    args: tuple
    nullable: bool


class Algebra:
    """An intern table plus derivative memo. Terms from one instance can be compared."""

    def __init__(self) -> None:
        self._terms: list[_Term] = []
        self._index: dict[tuple, int] = {}
        self._deriv: dict[tuple[int, int], int] = {}
        self._intern("empty", (), False)
        self._intern("eps", (), True)

    def _intern(self, kind: str, args: tuple, nullable: bool) -> int:
        key = (kind, args)
        term_id = self._index.get(key)
        if term_id is None:
            term_id = len(self._terms)
            self._terms.append(_Term(kind, args, nullable))
            self._index[key] = term_id
        return term_id

    def nullable(self, term: int) -> bool:
        return self._terms[term].nullable

    def chars(self, mask: int) -> int:
        return self._intern("chars", (mask,), False) if mask else _EMPTY

    def seq(self, left: int, right: int) -> int:
        if left == _EMPTY or right == _EMPTY:
            return _EMPTY
        if left == _EPS:
            return right
        if right == _EPS:
            return left
        term = self._terms[left]
        if term.kind == "seq":
            right = self.seq(term.args[1], right)
            left = term.args[0]
        return self._intern(
            "seq", (left, right), self._terms[left].nullable and self._terms[right].nullable
        )

    def either(self, left: int, right: int) -> int:
        members: set[int] = set()
        char_bits = 0
        for operand in (left, right):
            term = self._terms[operand]
            parts = term.args if term.kind == "alt" else (operand,)
            for part in parts:
                part_term = self._terms[part]
                if part_term.kind == "chars":
                    char_bits |= part_term.args[0]
                elif part != _EMPTY:
                    members.add(part)
        if char_bits:
            members.add(self.chars(char_bits))
        if not members:
            return _EMPTY
        if len(members) == 1:
            return next(iter(members))
        args = tuple(sorted(members))
        return self._intern("alt", args, any(self._terms[m].nullable for m in args))

    def star(self, body: int) -> int:
        term = self._terms[body]
        if body in (_EMPTY, _EPS):
            return _EPS
        if term.kind == "star":
            return body
        return self._intern("star", (body,), True)

    def compile(self, node: Node) -> int:
        tok = node.tok
        if tok.startswith("lit:"):
            return self.chars(_CHAR_BIT[tok[4:]])
        if tok.startswith("cls:"):
            return self.chars(_CLASS_MASK[tok[4:]])
        if tok == CAT:
            return self.seq(self.compile(node.kids[0]), self.compile(node.kids[1]))
        if tok == "alt":
            return self.either(self.compile(node.kids[0]), self.compile(node.kids[1]))
        low, high = QUANTIFIERS[tok[2:]]
        body = self.compile(node.kids[0])
        if high is None:
            tail = self.star(body)
        else:
            tail = _EPS
            optional = self.either(body, _EPS)
            for _ in range(high - low):
                tail = self.seq(optional, tail)
        for _ in range(low):
            tail = self.seq(body, tail)
        return tail

    def derive(self, term: int, bit: int) -> int:
        key = (term, bit)
        cached = self._deriv.get(key)
        if cached is not None:
            return cached
        data = self._terms[term]
        if data.kind == "chars":
            result = _EPS if data.args[0] & bit else _EMPTY
        elif data.kind == "seq":
            left, right = data.args
            result = self.seq(self.derive(left, bit), right)
            if self._terms[left].nullable:
                result = self.either(result, self.derive(right, bit))
        elif data.kind == "alt":
            result = _EMPTY
            for member in data.args:
                result = self.either(result, self.derive(member, bit))
        elif data.kind == "star":
            result = self.seq(self.derive(data.args[0], bit), term)
        else:
            result = _EMPTY
        self._deriv[key] = result
        return result

    def matches(self, term: int, text: str) -> bool:
        for char in text:
            bit = _CHAR_BIT.get(char)
            if bit is None:
                return False
            term = self.derive(term, bit)
            if term == _EMPTY:
                return False
        return self._terms[term].nullable

    def _masks(self, term: int, seen: set[int], out: set[int]) -> None:
        if term in seen:
            return
        seen.add(term)
        data = self._terms[term]
        if data.kind == "chars":
            out.add(data.args[0])
        elif data.kind != "eps" and data.kind != "empty":
            for arg in data.args:
                self._masks(arg, seen, out)

    def _minterms(self, a: int, b: int) -> list[tuple[int, str]]:
        """One representative per group of characters that no class tells apart."""
        masks: set[int] = set()
        seen: set[int] = set()
        self._masks(a, seen, masks)
        self._masks(b, seen, masks)
        signatures: dict[tuple[bool, ...], str] = {}
        ordered = sorted(masks)
        for char, bit in _CHAR_BIT.items():
            signatures.setdefault(tuple(bool(m & bit) for m in ordered), char)
        return [(_CHAR_BIT[char], char) for char in signatures.values()]

    def compare(self, a: int, b: int, max_states: int = 50_000) -> Comparison:
        """Hopcroft-Karp equivalence over lazily built DFAs.

        Merging states with union-find keeps this near-linear in the product automaton
        that is actually reached, and the breadth-first order means a counterexample
        is short, which matters because it ends up in the failure report.
        """
        parent: dict[int, int] = {}

        def find(x: int) -> int:
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        letters = self._minterms(a, b)
        frontier = [(a, b, "")]
        steps = 0
        while frontier:
            next_frontier = []
            for x, y, prefix in frontier:
                root_x, root_y = find(x), find(y)
                if root_x == root_y:
                    continue
                if self.nullable(x) != self.nullable(y):
                    return Comparison(False, prefix)
                parent[root_x] = root_y
                steps += 1
                if steps > max_states:
                    return Comparison(None, None)
                for bit, char in letters:
                    next_frontier.append((self.derive(x, bit), self.derive(y, bit), prefix + char))
            frontier = next_frontier
        return Comparison(True, None)


@dataclass(frozen=True, slots=True)
class Comparison:
    """``equal`` is None if the state limit was hit before an answer."""

    equal: bool | None
    witness: str | None


def compare(a: Node, b: Node, max_states: int = 50_000) -> Comparison:
    algebra = Algebra()
    return algebra.compare(algebra.compile(a), algebra.compile(b), max_states)


def matches(node: Node, text: str) -> bool:
    algebra = Algebra()
    return algebra.matches(algebra.compile(node), text)


def _is_repeat(tok: str) -> bool:
    if not tok.startswith("q:"):
        return False
    high = QUANTIFIERS[tok[2:]][1]
    return high is None or high > 1


def _count_unbounded(node: Node) -> int:
    own = node.tok in ("q:*", "q:+")
    return own + sum(_count_unbounded(kid) for kid in node.kids)


def may_backtrack_badly(node: Node) -> bool:
    """True if a backtracking engine might not finish on a 40-character string.

    The danger is a body that can match one string in several ways sitting inside any
    repetition: ``(a+b?)+``, ``(\\d|[0-9a-f])+``, or with a counted loop, ``((a|b|a){8}){8}``,
    which is finite but astronomically large. A long chain of unbounded runs such as
    ``.*a.*b.*c.*d`` is polynomial, but of a high enough degree to stall as well.

    Each of those shapes hung the training data pipeline at some point. The test
    over-approximates on purpose: flagged targets go through the derivative matcher,
    which is slower and always terminates.
    """
    return _count_unbounded(node) >= _MAX_UNBOUNDED_RUNS or _ambiguous_loop(node, False)


def _ambiguous_loop(node: Node, under_loop: bool) -> bool:
    if under_loop:
        tok = node.tok
        variable = tok.startswith("q:") and QUANTIFIERS[tok[2:]][0] != QUANTIFIERS[tok[2:]][1]
        if variable or tok == "alt":
            return True
    loop = under_loop or _is_repeat(node.tok)
    return any(_ambiguous_loop(kid, loop) for kid in node.kids)
