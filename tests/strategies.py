"""Random regex trees for property tests, covering the whole grammar and alphabet."""

from hypothesis import strategies as st

from regexsmith.dsl import (
    ALPHABET,
    CLASSES,
    QUANTIFIERS,
    Node,
    alt,
    cat,
    cls,
    lit,
    normalize,
    quant,
)
from regexsmith.grammar import is_canonical

_leaves = st.one_of(
    st.sampled_from(sorted(ALPHABET)).map(lit),
    st.sampled_from(sorted(CLASSES)).map(cls),
)


def _extend(children):
    quantified = st.builds(
        lambda q, body: body if body.tok.startswith("q:") else quant(q, body),
        st.sampled_from(sorted(QUANTIFIERS)),
        children,
    )
    return st.one_of(
        st.builds(cat, children, children),
        st.builds(alt, children, children),
        quantified,
    )


def _usable(node: Node) -> bool:
    return sum(1 for _ in _nodes(node)) <= 16 and is_canonical(node)


def _nodes(node: Node):
    yield node
    for kid in node.kids:
        yield from _nodes(kid)


trees = st.recursive(_leaves, _extend, max_leaves=6).map(normalize).filter(_usable)
texts = st.text(alphabet="abAB01-_. ", max_size=7)
