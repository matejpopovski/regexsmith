"""Random synthesis tasks, ordered by difficulty.

A task's tier is the token count of its target, the same quantity the enumerator's
cost measures, so the tiers line up with how far enumeration can reach. The sampler is
biased toward format-like regexes (runs of a class, separators, short words) rather
than uniform trees, which are mostly noise such as ``(q|.)?Z*``.

Negatives are the part that needs care. Uniformly random strings are rejected by
almost any regex, so they say little. Most negatives here are near-misses: a positive
with one edit, or a string from a slightly different regex.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass

from .dsl import (
    ALPHABET,
    ALT,
    CAT,
    CLASSES,
    QUANTIFIERS,
    Node,
    UnsupportedRegex,
    alt,
    cat,
    cls,
    lit,
    normalize,
    parse,
    quant,
    size,
    spine,
    to_regex,
    word,
)
from .field import FIELD_TARGETS
from .grammar import is_canonical
from .semantics import Algebra, may_backtrack_badly

TIERS: dict[int, tuple[int, int]] = {1: (1, 4), 2: (5, 9), 3: (10, 15), 4: (16, 23), 5: (24, 36)}
MAX_STRING = 40
HELDOUT_BUCKETS = 32

_CLASS_NAMES = list(CLASSES)
_CLASS_WEIGHTS = [8, 8, 4, 3, 3, 2, 2, 3, 1]
_SEPARATORS = "-_.:/@#,=+"
_QUANT_NAMES = list(QUANTIFIERS)
_QUANT_WEIGHTS = [5, 10, 6, 3, 3, 3, 1, 1, 1, 1, 2, 3, 1, 1, 2, 1]
_FIELD_REGEXES = frozenset(to_regex(parse(text)) for _, text in FIELD_TARGETS)


@dataclass(frozen=True, slots=True)
class Task:
    target: str
    tier: int
    positives: tuple[str, ...]
    negatives: tuple[str, ...]
    hidden_positives: tuple[str, ...] = ()
    hidden_negatives: tuple[str, ...] = ()
    name: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, line: str) -> Task:
        raw = json.loads(line)
        for key in ("positives", "negatives", "hidden_positives", "hidden_negatives"):
            raw[key] = tuple(raw[key])
        return cls(**raw)


def is_heldout(regex: str) -> bool:
    """Assign a target to the test split by content, so no test regex is ever trained on."""
    digest = hashlib.blake2b(regex.encode(), digest_size=4).digest()
    return int.from_bytes(digest) % HELDOUT_BUCKETS == 0


def _nullable(node: Node) -> bool:
    if node.tok == CAT:
        return all(_nullable(kid) for kid in node.kids)
    if node.tok == ALT:
        return any(_nullable(kid) for kid in node.kids)
    if node.tok.startswith("q:"):
        return QUANTIFIERS[node.tok[2:]][0] == 0 or _nullable(node.kids[0])
    return False


def _atom(rng: random.Random, palette: str) -> Node:
    roll = rng.random()
    if roll < 0.5:
        return cls(rng.choices(_CLASS_NAMES, _CLASS_WEIGHTS)[0])
    return lit(rng.choice(_SEPARATORS if roll < 0.72 else palette))


def _quantify(rng: random.Random, body: Node) -> Node:
    if body.tok.startswith("q:"):
        return body
    return quant(rng.choices(_QUANT_NAMES, _QUANT_WEIGHTS)[0], body)


def _split(rng: random.Random, total: int, parts: int) -> list[int]:
    cuts = sorted(rng.sample(range(1, total), parts - 1))
    return [b - a for a, b in zip([0, *cuts], [*cuts, total], strict=True)]


def _expr(rng: random.Random, budget: int, depth: int, palette: str) -> Node:
    if budget <= 1:
        return _atom(rng, palette)
    if budget == 2:
        return _quantify(rng, _atom(rng, palette))
    choice = rng.choices(
        ("seq", "word", "alt", "group"),
        (0.58, 0.1, 0.14, 0.18) if depth < 3 else (0.85, 0.15, 0, 0),
    )[0]
    if choice == "word":
        length = min((budget + 1) // 2, 6)
        return word("".join(rng.choice(palette) for _ in range(length)))
    if choice == "group":
        return _quantify(rng, _expr(rng, budget - 1, depth + 1, palette))
    parts = rng.randint(2, min(5, (budget + 1) // 2)) if choice == "seq" else rng.randint(2, 3)
    free = budget - (parts - 1)
    if free < parts:
        parts = 2
        free = budget - 1
    kids = [_expr(rng, b, depth + 1, palette) for b in _split(rng, free, parts)]
    return cat(*kids) if choice == "seq" else alt(*kids)


def sample_target(rng: random.Random, tier: int, heldout: bool = False) -> Node:
    """Sample a canonical target of the tier from the train or the test split."""
    low, high = TIERS[tier]
    while True:
        palette = "".join(rng.sample(ALPHABET[:52], rng.randint(2, 6)))
        node = normalize(_expr(rng, rng.randint(low, high), 0, palette))
        if not low <= size(node) <= high or not is_canonical(node):
            continue
        if any(n.tok.startswith("q:") and _nullable(n.kids[0]) for n in _walk(node)):
            continue
        regex = to_regex(node)
        if regex in _FIELD_REGEXES or is_heldout(regex) != heldout or not _round_trips(node, regex):
            continue
        return node


def _round_trips(node: Node, regex: str) -> bool:
    """Tasks store regex text, so reject trees whose rendering reads back as another tree.

    ``[a-z0-9]`` is one class token, so the two-token alternation that renders the same way
    would give the same text two program encodings.
    """
    try:
        return parse(regex) == node
    except UnsupportedRegex:
        return False


def _walk(node: Node) -> Iterator[Node]:
    yield node
    for kid in node.kids:
        yield from _walk(kid)


def _class_char(rng: random.Random, name: str) -> str:
    return rng.choice(rng.choice(CLASSES[name].groups))


def _sample_into(rng: random.Random, node: Node, out: list[str]) -> None:
    tok = node.tok
    if tok.startswith("lit:"):
        out.append(tok[4:])
    elif tok.startswith("cls:"):
        out.append(_class_char(rng, tok[4:]))
    elif tok == CAT:
        _sample_into(rng, node.kids[0], out)
        _sample_into(rng, node.kids[1], out)
    elif tok == ALT:
        _sample_into(rng, rng.choice(spine(node, ALT)), out)
    else:
        low, high = QUANTIFIERS[tok[2:]]
        if high is None:
            single_char = not node.kids[0].kids
            high = low + (rng.choice((1, 2, 3, 4, 6, 8)) if single_char else rng.choice((1, 2, 3)))
        for _ in range(rng.randint(low, high)):
            _sample_into(rng, node.kids[0], out)


def sample_string(rng: random.Random, node: Node) -> str:
    out: list[str] = []
    _sample_into(rng, node, out)
    return "".join(out)


def _acceptor(target: Node) -> Callable[[str], bool]:
    if may_backtrack_badly(target):
        algebra = Algebra()
        term = algebra.compile(target)
        return lambda text: algebra.matches(term, text)
    fullmatch = re.compile(to_regex(target)).fullmatch
    return lambda text: fullmatch(text) is not None


def _edit_string(rng: random.Random, text: str) -> str:
    if not text:
        return rng.choice(ALPHABET)
    i = rng.randrange(len(text))
    edit = rng.randrange(7)
    swap_in = _class_char(rng, "any")
    if edit == 0:
        return text[:i] + swap_in + text[i + 1 :]
    if edit == 1:
        return text[:i] + text[i + 1 :]
    if edit == 2:
        return text[:i] + swap_in + text[i:]
    if edit == 3:
        return text[:i] + text[i] + text[i:]
    if edit == 4 and len(text) > 1:
        j = min(i, len(text) - 2)
        return text[:j] + text[j + 1] + text[j] + text[j + 2 :]
    if edit == 5:
        return text[:i] if rng.random() < 0.5 else text[i:]
    return text + rng.choice(text) if rng.random() < 0.5 else rng.choice(text) + text


def _mutate_tree(rng: random.Random, node: Node) -> Node:
    """Replace one random subtree with a close relative of itself."""
    nodes = list(_walk(node))
    chosen = rng.choice(nodes)

    def replacement(n: Node) -> Node:
        if n.tok.startswith("cls:"):
            return cls(rng.choice(_CLASS_NAMES))
        if n.tok.startswith("lit:"):
            return lit(rng.choice(ALPHABET))
        if n.tok.startswith("q:"):
            return rng.choice((n.kids[0], quant(rng.choice(_QUANT_NAMES), n.kids[0])))
        return rng.choice(n.kids)

    def rebuild(n: Node) -> Node:
        if n is chosen:
            return replacement(n)
        return Node(n.tok, tuple(rebuild(kid) for kid in n.kids))

    return rebuild(node)


def _negative(rng: random.Random, target: Node, positives: list[str]) -> str:
    roll = rng.random()
    if roll < 0.68 and positives:
        text = _edit_string(rng, rng.choice(positives))
        return _edit_string(rng, text) if rng.random() < 0.25 else text
    if roll < 0.92:
        return sample_string(rng, _mutate_tree(rng, target))
    if positives:
        pool = "".join(positives) or ALPHABET
        return "".join(rng.choice(pool) for _ in range(rng.randint(0, 12)))
    return ""


def _collect(sample: Callable[[], str], keep: Callable[[str], bool], want: int) -> list[str]:
    found: dict[str, None] = {}
    for _ in range(want * 6):
        text = sample()
        if len(text) <= MAX_STRING and text not in found and keep(text):
            found[text] = None
            if len(found) == want:
                break
    return list(found)


def make_task(
    target: Node,
    rng: random.Random,
    n_pos: int = 10,
    n_neg: int = 10,
    n_hidden: int = 0,
    tier: int = 0,
    name: str = "",
) -> Task | None:
    """Sample examples for a target, or None if it cannot support a fair task."""
    accepts = _acceptor(target)
    positives = _collect(lambda: sample_string(rng, target), accepts, n_pos + n_hidden)
    visible = positives[:n_pos]
    if not any(visible):
        return None
    negatives = _collect(
        lambda: _negative(rng, target, positives),
        lambda text: not accepts(text),
        n_neg + n_hidden,
    )
    if len(negatives) < min(n_neg, 4):
        return None
    return Task(
        target=to_regex(target),
        tier=tier,
        positives=tuple(visible),
        negatives=tuple(negatives[:n_neg]),
        hidden_positives=tuple(positives[len(visible) :]),
        hidden_negatives=tuple(negatives[n_neg:]),
        name=name,
    )


def sample_task(
    rng: random.Random,
    tier: int,
    n_pos: int = 10,
    n_neg: int = 10,
    n_hidden: int = 0,
    heldout: bool = False,
) -> Task:
    while True:
        target = sample_target(rng, tier, heldout)
        task = make_task(target, rng, n_pos, n_neg, n_hidden, tier)
        if task is not None:
            return task
