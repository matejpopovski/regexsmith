"""The regex language every other module shares.

Regexes are trees, and programs are the pre-order walk of a tree. That choice
carries the design: a prefix of a pre-order walk is a tree with holes, so every
partial program the decoder produces has a well-defined meaning that can be
bounded and executed (see ``overapprox`` and ``underapprox``). A character-level
decoder has no such property; ``([a-z]+|`` is not a regex of anything.

Size is measured in tokens, which is also the enumerator's cost metric, so the
difficulty tiers mean the same thing to both synthesizers.
"""

from __future__ import annotations

import string
from dataclasses import dataclass

# Printable ASCII. Letters come first; the task sampler relies on that ordering.
ALPHABET = (
    string.ascii_lowercase + string.ascii_uppercase + string.digits + string.punctuation + " "
)
MAX_TOKENS = 48


@dataclass(frozen=True, slots=True)
class CharClass:
    regex: str
    set_body: str
    groups: tuple[str, ...]

    @property
    def chars(self) -> str:
        return "".join(self.groups)


_LOWER = string.ascii_lowercase
_UPPER = string.ascii_uppercase
_DIGIT = string.digits
_PUNCT = string.punctuation + " "

# ``groups`` keeps a class's visibly different parts apart so the task sampler can
# pick a group first. Sampling flat from \w would almost never show an underscore,
# which would make \w and [a-zA-Z0-9] indistinguishable from examples.
CLASSES: dict[str, CharClass] = {
    "digit": CharClass(r"\d", "0-9", (_DIGIT,)),
    "lower": CharClass("[a-z]", "a-z", (_LOWER,)),
    "upper": CharClass("[A-Z]", "A-Z", (_UPPER,)),
    "alpha": CharClass("[a-zA-Z]", "a-zA-Z", (_LOWER, _UPPER)),
    "lowernum": CharClass("[a-z0-9]", "a-z0-9", (_LOWER, _DIGIT)),
    "alnum": CharClass("[a-zA-Z0-9]", "a-zA-Z0-9", (_LOWER, _UPPER, _DIGIT)),
    "word": CharClass(r"\w", "a-zA-Z0-9_", (_LOWER, _UPPER, _DIGIT, "_")),
    "hex": CharClass("[0-9a-f]", "0-9a-f", (_DIGIT, "abcdef")),
    "any": CharClass(".", "", (_LOWER, _UPPER, _DIGIT, _PUNCT)),
}

# Lower bound and upper bound (None for unbounded) for each quantifier token.
QUANTIFIERS: dict[str, tuple[int, int | None]] = {
    "*": (0, None),
    "+": (1, None),
    "?": (0, 1),
    "{2}": (2, 2),
    "{3}": (3, 3),
    "{4}": (4, 4),
    "{5}": (5, 5),
    "{6}": (6, 6),
    "{8}": (8, 8),
    "{12}": (12, 12),
    "{1,2}": (1, 2),
    "{1,3}": (1, 3),
    "{1,4}": (1, 4),
    "{2,3}": (2, 3),
    "{2,4}": (2, 4),
    "{3,5}": (3, 5),
}

CAT = "cat"
ALT = "alt"
HOLE = "?hole"


def _build_vocab() -> list[str]:
    vocab = ["<pad>", "<bos>", CAT, ALT]
    vocab += [f"q:{q}" for q in QUANTIFIERS]
    vocab += [f"cls:{name}" for name in CLASSES]
    vocab += [f"lit:{c}" for c in ALPHABET]
    return vocab


VOCAB = _build_vocab()
TOKEN_ID = {tok: i for i, tok in enumerate(VOCAB)}
PAD_ID = TOKEN_ID["<pad>"]
BOS_ID = TOKEN_ID["<bos>"]


def arity(tok: str) -> int:
    if tok in (CAT, ALT):
        return 2
    return 1 if tok.startswith("q:") else 0


ARITY = [arity(tok) for tok in VOCAB]


@dataclass(frozen=True, slots=True)
class Node:
    tok: str
    kids: tuple[Node, ...] = ()


def lit(char: str) -> Node:
    return Node(f"lit:{char}")


def word(text: str) -> Node:
    return cat(*[lit(c) for c in text])


def cls(name: str) -> Node:
    return Node(f"cls:{name}")


def quant(q: str, body: Node) -> Node:
    return Node(f"q:{q}", (body,))


def cat(*parts: Node) -> Node:
    return _right_nest(CAT, parts)


def alt(*parts: Node) -> Node:
    return _right_nest(ALT, parts)


def _right_nest(tok: str, parts: tuple[Node, ...] | list[Node]) -> Node:
    node = parts[-1]
    for part in reversed(parts[:-1]):
        node = Node(tok, (part, node))
    return node


def spine(node: Node, tok: str) -> list[Node]:
    """The operands of a right-nested chain of ``tok``."""
    parts: list[Node] = []
    while node.tok == tok:
        parts.append(node.kids[0])
        node = node.kids[1]
    parts.append(node)
    return parts


def size(node: Node) -> int:
    return 1 + sum(size(kid) for kid in node.kids)


def preorder(node: Node) -> list[str]:
    out: list[str] = []
    stack = [node]
    while stack:
        current = stack.pop()
        out.append(current.tok)
        stack.extend(reversed(current.kids))
    return out


def from_tokens(tokens: list[str] | tuple[str, ...]) -> Node:
    """Rebuild a tree from a token prefix. Missing subtrees become holes."""
    position = 0

    def build() -> Node:
        nonlocal position
        if position >= len(tokens):
            return Node(HOLE)
        tok = tokens[position]
        position += 1
        return Node(tok, tuple(build() for _ in range(arity(tok))))

    node = build()
    if position != len(tokens):
        raise ValueError("tokens continue past the end of the tree")
    return node


def has_hole(node: Node) -> bool:
    return node.tok == HOLE or any(has_hole(kid) for kid in node.kids)


# Rendering ------------------------------------------------------------------------

_ESCAPE = set(".+*?()[]{}|^$\\")
_SET_ESCAPE = set("-[]^\\")
_PREC_ALT, _PREC_CAT, _PREC_ATOM = 0, 1, 2


def _escape(char: str) -> str:
    return "\\" + char if char in _ESCAPE else char


def _set_member(node: Node) -> str | None:
    if node.tok.startswith("lit:"):
        char = node.tok[4:]
        return "\\" + char if char in _SET_ESCAPE else char
    if node.tok.startswith("cls:") and node.tok != "cls:any":
        return CLASSES[node.tok[4:]].set_body
    return None


def to_regex(node: Node, _prec: int = _PREC_ALT) -> str:
    tok = node.tok
    if tok == HOLE:
        return "□"
    if tok.startswith("lit:"):
        return _escape(tok[4:])
    if tok.startswith("cls:"):
        return CLASSES[tok[4:]].regex
    if tok.startswith("q:"):
        return to_regex(node.kids[0], _PREC_ATOM) + tok[2:]
    if tok == CAT:
        text = "".join(to_regex(part, _PREC_CAT) for part in spine(node, CAT))
        return f"({text})" if _prec >= _PREC_ATOM else text
    branches = spine(node, ALT)
    members = [_set_member(branch) for branch in branches]
    if all(member is not None for member in members):
        return "[" + "".join(members) + "]"
    text = "|".join(to_regex(branch, _PREC_ALT) for branch in branches)
    return f"({text})" if _prec >= _PREC_CAT else text


# Parsing --------------------------------------------------------------------------


class UnsupportedRegex(ValueError):
    """The text uses syntax outside this language."""


_SET_BODIES = sorted(
    ((c.set_body, name) for name, c in CLASSES.items() if c.set_body),
    key=lambda pair: -len(pair[0]),
)


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.i = 0

    def peek(self) -> str:
        return self.text[self.i] if self.i < len(self.text) else ""

    def eat(self, char: str) -> bool:
        if self.peek() == char:
            self.i += 1
            return True
        return False

    def parse(self) -> Node:
        node = self.alternation()
        if self.i != len(self.text):
            raise UnsupportedRegex(f"unexpected {self.peek()!r} at offset {self.i}")
        return node

    def alternation(self) -> Node:
        branches = [self.sequence()]
        while self.eat("|"):
            branches.append(self.sequence())
        return alt(*branches)

    def sequence(self) -> Node:
        parts: list[Node] = []
        while self.peek() and self.peek() not in "|)":
            parts.append(self.term())
        if not parts:
            raise UnsupportedRegex(f"empty branch at offset {self.i}")
        return cat(*parts)

    def term(self) -> Node:
        node = self.atom()
        q = self.quantifier()
        if q is None:
            return node
        if node.tok.startswith("q:"):
            raise UnsupportedRegex("a quantifier directly on a quantifier is outside the grammar")
        return quant(q, node)

    def quantifier(self) -> str | None:
        char = self.peek()
        if char in ("*", "+", "?") and char:
            self.i += 1
            return char
        if char == "{":
            end = self.text.find("}", self.i)
            q = self.text[self.i : end + 1] if end >= 0 else ""
            if q not in QUANTIFIERS:
                raise UnsupportedRegex(f"quantifier {q or '{'!r} is not in the grammar")
            self.i = end + 1
            return q
        return None

    def atom(self) -> Node:
        if self.eat("("):
            if self.text.startswith("?:", self.i):
                self.i += 2
            elif self.peek() == "?":
                raise UnsupportedRegex("lookaround and named groups are outside the grammar")
            node = self.alternation()
            if not self.eat(")"):
                raise UnsupportedRegex("unbalanced parenthesis")
            return node
        if self.eat("["):
            return self.char_set()
        if self.eat("."):
            return cls("any")
        if self.eat("\\"):
            char = self.peek()
            self.i += 1
            if char == "d":
                return cls("digit")
            if char == "w":
                return cls("word")
            if char.isalnum():
                raise UnsupportedRegex(f"escape \\{char} is outside the grammar")
            return self.literal(char)
        char = self.peek()
        if char in ("^", "$"):
            raise UnsupportedRegex("anchors are implicit: solutions are checked with fullmatch")
        if char in ("*", "+", "?", "{"):
            raise UnsupportedRegex(f"nothing to repeat at offset {self.i}")
        self.i += 1
        return self.literal(char)

    def literal(self, char: str) -> Node:
        if len(char) != 1 or char not in ALPHABET:
            raise UnsupportedRegex(f"{char!r} is outside the alphabet")
        return lit(char)

    def char_set(self) -> Node:
        members: list[Node] = []
        if self.peek() == "^":
            raise UnsupportedRegex("negated sets are outside the grammar")
        while not self.eat("]"):
            if not self.peek():
                raise UnsupportedRegex("unbalanced bracket")
            for body, name in _SET_BODIES:
                if self.text.startswith(body, self.i):
                    members.append(cls(name))
                    self.i += len(body)
                    break
            else:
                escaped = self.eat("\\")
                low = self.literal(self.peek())
                self.i += 1
                if escaped or self.peek() != "-" or self.text[self.i + 1 : self.i + 2] in ("]", ""):
                    members.append(low)
                    continue
                high = self.literal(self.text[self.i + 1])
                self.i += 2
                first, last = ord(low.tok[4:]), ord(high.tok[4:])
                if first > last:
                    raise UnsupportedRegex(f"bad range ending at offset {self.i}")
                members.extend(lit(chr(code)) for code in range(first, last + 1))
        return alt(*members)


def parse(text: str) -> Node:
    """Parse the subset of regex syntax that ``to_regex`` can emit."""
    return normalize(_Parser(text).parse())


def normalize(node: Node) -> Node:
    """Canonical form: right-nested chains and no duplicate alternation branches."""
    kids = tuple(normalize(kid) for kid in node.kids)
    if node.tok == CAT:
        parts = [leaf for kid in kids for leaf in spine(kid, CAT)]
        return cat(*parts)
    if node.tok == ALT:
        branches = dict.fromkeys(leaf for kid in kids for leaf in spine(kid, ALT))
        return alt(*branches)
    return Node(node.tok, kids)


# Approximating sketches -----------------------------------------------------------

_SIGMA_STAR = quant("*", cls("any"))


def overapprox(node: Node) -> Node:
    """A regex whose language contains every completion of the sketch.

    Holes become ``.*``. Concatenation, alternation and every quantifier are monotone
    in their operands, so the result is a superset of any way of filling the holes.
    """
    if node.tok == HOLE:
        return _SIGMA_STAR
    if not node.kids:
        return node
    kids = tuple(overapprox(kid) for kid in node.kids)
    # (.*)* and .*|a add nothing to the language and everything to the backtracking.
    absorbs = node.tok == ALT and _SIGMA_STAR in kids
    absorbs |= node.tok == CAT and kids == (_SIGMA_STAR, _SIGMA_STAR)
    absorbs |= node.tok.startswith("q:") and kids[0] == _SIGMA_STAR
    return _SIGMA_STAR if absorbs else Node(node.tok, kids)


def underapprox(node: Node) -> Node | None:
    """A regex whose language is inside every completion, or None for the empty language.

    Holes become the empty language and the constructors are simplified around it, since
    the DSL has no token for the empty set. A quantifier over the empty language is
    treated as empty even when it could still match the empty string, which only makes
    the approximation smaller and so stays sound.
    """
    if node.tok == HOLE:
        return None
    if not node.kids:
        return node
    if node.tok == CAT:
        left, right = underapprox(node.kids[0]), underapprox(node.kids[1])
        return None if left is None or right is None else Node(CAT, (left, right))
    if node.tok == ALT:
        left, right = underapprox(node.kids[0]), underapprox(node.kids[1])
        if left is None or right is None:
            return left or right
        return Node(ALT, (left, right))
    body = underapprox(node.kids[0])
    return None if body is None else Node(node.tok, (body,))
