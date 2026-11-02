"""Which token may come next.

Decoding is a stack machine over open holes. Each hole carries a context that rules out
trees with a redundant equivalent: chains must nest to the right, so ``(ab)c`` and
``a(bc)`` are one program, and a quantifier cannot sit directly on a quantifier, which
removes ``(a*)*`` and its relatives. The same rules bound the enumerator, so both
synthesizers search exactly one space.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dsl import ALT, ARITY, CAT, MAX_TOKENS, TOKEN_ID, VOCAB, Node, preorder

CTX_ANY, CTX_NOT_CAT, CTX_NOT_ALT, CTX_NOT_QUANT, CTX_DONE = range(5)
NUM_CONTEXTS = 5

_CAT_ID = TOKEN_ID[CAT]
_ALT_ID = TOKEN_ID[ALT]
_QUANT_IDS = [i for i, tok in enumerate(VOCAB) if tok.startswith("q:")]
_SPECIAL_IDS = [TOKEN_ID["<pad>"], TOKEN_ID["<bos>"]]


def _context_table() -> np.ndarray:
    table = np.ones((NUM_CONTEXTS, len(VOCAB)), dtype=bool)
    table[:, _SPECIAL_IDS] = False
    table[CTX_NOT_CAT, _CAT_ID] = False
    table[CTX_NOT_ALT, _ALT_ID] = False
    table[CTX_NOT_QUANT, _QUANT_IDS] = False
    table[CTX_DONE] = False
    return table


CONTEXT_MASK = _context_table()
ARITY_ARRAY = np.array(ARITY)


def child_contexts(token_id: int) -> tuple[int, ...]:
    """Contexts of the holes a token opens, leftmost first."""
    if token_id == _CAT_ID:
        return (CTX_NOT_CAT, CTX_ANY)
    if token_id == _ALT_ID:
        return (CTX_NOT_ALT, CTX_ANY)
    return (CTX_NOT_QUANT,) * ARITY[token_id]


@dataclass(slots=True)
class DecodeState:
    """Open holes, top of the stack being the leftmost."""

    stack: list[int] = field(default_factory=lambda: [CTX_ANY])
    length: int = 0

    @property
    def done(self) -> bool:
        return not self.stack

    def legal(self, max_tokens: int = MAX_TOKENS) -> np.ndarray:
        if not self.stack:
            return CONTEXT_MASK[CTX_DONE]
        # Every open hole needs at least one more token, so an operator is legal only if
        # the holes it leaves behind can still be closed within the length limit.
        room = max_tokens - self.length - 1
        holes_after = len(self.stack) - 1 + ARITY_ARRAY
        return CONTEXT_MASK[self.stack[-1]] & (holes_after <= room)

    def push(self, token_id: int) -> DecodeState:
        stack = self.stack[:-1]
        stack.extend(reversed(child_contexts(token_id)))
        return DecodeState(stack, self.length + 1)


def contexts_for(tokens: list[int]) -> tuple[list[int], list[int]]:
    """For each position in a token sequence, the hole context and open-hole count."""
    state = DecodeState()
    contexts, holes = [], []
    for token in tokens:
        contexts.append(state.stack[-1])
        holes.append(len(state.stack))
        state = state.push(token)
    return contexts, holes


def is_canonical(node: Node, max_tokens: int = MAX_TOKENS) -> bool:
    """True if the grammar can produce this tree."""
    state = DecodeState()
    for tok in preorder(node):
        token_id = TOKEN_ID[tok]
        if state.done or not state.legal(max_tokens)[token_id]:
            return False
        state = state.push(token_id)
    return state.done
