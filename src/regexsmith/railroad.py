"""Railroad diagrams, for the terminal and as SVG.

A regex that passes every example can still be wrong, and a diagram is the fastest way
for a person to see what it really accepts. Both renderers share one simplified layout
tree, which merges literal runs into words, so ``c o m`` draws as one box.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

from .dsl import ALT, CAT, CLASSES, QUANTIFIERS, Node, spine


@dataclass(frozen=True, slots=True)
class Box:
    text: str
    terminal: bool


@dataclass(frozen=True, slots=True)
class Seq:
    parts: tuple


@dataclass(frozen=True, slots=True)
class Stack:
    branches: tuple


@dataclass(frozen=True, slots=True)
class Loop:
    body: object
    label: str
    skippable: bool


@dataclass(frozen=True, slots=True)
class Skip:
    body: object


def layout(node: Node) -> Box | Seq | Stack | Loop | Skip:
    tok = node.tok
    if tok.startswith("lit:"):
        return Box(tok[4:], True)
    if tok.startswith("cls:"):
        return Box(CLASSES[tok[4:]].regex, False)
    if tok == CAT:
        parts: list = []
        for part in spine(node, CAT):
            item = layout(part)
            if isinstance(item, Box) and item.terminal and parts:
                previous = parts[-1]
                if isinstance(previous, Box) and previous.terminal:
                    parts[-1] = Box(previous.text + item.text, True)
                    continue
            parts.append(item)
        return parts[0] if len(parts) == 1 else Seq(tuple(parts))
    if tok == ALT:
        return Stack(tuple(layout(branch) for branch in spine(node, ALT)))
    q = tok[2:]
    body = layout(node.kids[0])
    if q == "?":
        return Skip(body)
    low, high = QUANTIFIERS[q]
    if q in ("*", "+"):
        return Loop(body, "", q == "*")
    label = f"{low}" if low == high else f"{low}-{high}"
    return Loop(body, f"{label}x", False)


# Terminal -------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Art:
    lines: tuple[str, ...]
    rail: int

    @property
    def width(self) -> int:
        return len(self.lines[0])


def _pad(art: _Art, width: int) -> _Art:
    extra = width - art.width
    return _Art(
        tuple(
            line + ("─" * extra if row == art.rail else " " * extra)
            for row, line in enumerate(art.lines)
        ),
        art.rail,
    )


def _text_box(box: Box) -> _Art:
    label = f'"{box.text}"' if box.terminal else box.text
    return _Art((f"─{label}─",), 0)


def _text_seq(items: list[_Art]) -> _Art:
    above = max(art.rail for art in items)
    below = max(len(art.lines) - art.rail for art in items)
    rows = [""] * (above + below)
    for art in items:
        offset = above - art.rail
        for row in range(len(rows)):
            inside = offset <= row < offset + len(art.lines)
            if inside:
                rows[row] += art.lines[row - offset]
            else:
                rows[row] += " " * art.width
    return _Art(tuple(rows), above)


def _text_stack(items: list[_Art]) -> _Art:
    width = max(art.width for art in items)
    rows: list[str] = []
    rail = items[0].rail
    last = len(items) - 1
    for index, art in enumerate(items):
        padded = _pad(art, width)
        for row, line in enumerate(padded.lines):
            if row == padded.rail:
                if index == 0:
                    left, right = "┬", "┬"
                elif index == last:
                    left, right = "└", "┘"
                else:
                    left, right = "├", "┤"
            elif (
                (index == 0 and row > padded.rail)
                or (index == last and row < padded.rail)
                or 0 < index < last
            ):
                left, right = "│", "│"
            else:
                left, right = " ", " "
            rows.append(left + line + right)
    return _Art(tuple(rows), rail)


def _text_skip(body: _Art) -> _Art:
    width = body.width
    top = "┬" + "─" * width + "┬"
    rows = [top]
    for row, line in enumerate(body.lines):
        if row < body.rail:
            rows.append("│" + line + "│")
        elif row == body.rail:
            rows.append("└" + line + "┘")
        else:
            rows.append(" " + line + " ")
    return _Art(tuple(rows), 0)


def _text_loop(body: _Art, label: str) -> _Art:
    width = body.width
    tag = f" {label} " if label else ""
    fill = max(width - len(tag) - 1, 0)
    arrow = "─" * (fill // 2) + "<" + tag + "─" * (fill - fill // 2)
    if len(arrow) < width:
        arrow += "─" * (width - len(arrow))
    grown = _pad(body, len(arrow))
    rows = []
    for row, line in enumerate(grown.lines):
        if row == grown.rail:
            rows.append("┬" + line + "┬")
        elif row > grown.rail:
            rows.append("│" + line + "│")
        else:
            rows.append(" " + line + " ")
    rows.append("└" + arrow + "┘")
    return _Art(tuple(rows), grown.rail)


def _render_text(item) -> _Art:
    if isinstance(item, Box):
        return _text_box(item)
    if isinstance(item, Seq):
        return _text_seq([_render_text(part) for part in item.parts])
    if isinstance(item, Stack):
        return _text_stack([_render_text(branch) for branch in item.branches])
    if isinstance(item, Skip):
        return _text_skip(_render_text(item.body))
    looped = _text_loop(_render_text(item.body), item.label)
    return _text_skip(looped) if item.skippable else looped


def to_text(node: Node) -> str:
    art = _render_text(layout(node))
    rows = [
        ("●─" if row == art.rail else "  ") + line + ("─●" if row == art.rail else "")
        for row, line in enumerate(art.lines)
    ]
    return "\n".join(line.rstrip() for line in rows)


# SVG ------------------------------------------------------------------------------

_CHAR_WIDTH = 8.4
_BOX_HEIGHT = 26
_GAP = 16
_ROW_GAP = 12
_RADIUS = 9


@dataclass(frozen=True, slots=True)
class _Size:
    width: float
    up: float
    down: float


def _measure(item) -> _Size:
    if isinstance(item, Box):
        extra = 2 if item.terminal else 0
        return _Size((len(item.text) + extra) * _CHAR_WIDTH + 20, _BOX_HEIGHT / 2, _BOX_HEIGHT / 2)
    if isinstance(item, Seq):
        sizes = [_measure(part) for part in item.parts]
        return _Size(
            sum(s.width for s in sizes) + _GAP * (len(sizes) - 1),
            max(s.up for s in sizes),
            max(s.down for s in sizes),
        )
    if isinstance(item, Stack):
        sizes = [_measure(branch) for branch in item.branches]
        height = sum(s.up + s.down for s in sizes) + _ROW_GAP * (len(sizes) - 1)
        return _Size(max(s.width for s in sizes) + 2 * _GAP, sizes[0].up, height - sizes[0].up)
    if isinstance(item, Skip):
        size = _measure(item.body)
        return _Size(size.width + 2 * _GAP, 0.0, size.up + size.down + _ROW_GAP)
    size = _measure(item.body)
    loop = _Size(size.width + 2 * _GAP, size.up, size.down + _ROW_GAP + (14 if item.label else 6))
    if item.skippable:
        return _Size(loop.width + 2 * _GAP, 0.0, loop.up + loop.down + _ROW_GAP)
    return loop


def _curve_down(x: float, y0: float, y1: float, direction: int) -> str:
    """A rail leaving y0 and arriving at y1 with rounded corners, travelling right."""
    sign = 1 if y1 > y0 else -1
    r = min(_RADIUS, abs(y1 - y0) / 2)
    return (
        f'<path class="rail" d="M{x:.1f} {y0:.1f} q{r * direction:.1f} 0 '
        f"{r * direction:.1f} {r * sign:.1f} V{y1 - r * sign:.1f} "
        f'q0 {r * sign:.1f} {r * direction:.1f} {r * sign:.1f}"/>'
    )


def _line(x0: float, y: float, x1: float) -> str:
    return f'<path class="rail" d="M{x0:.1f} {y:.1f} H{x1:.1f}"/>'


def _draw(item, x: float, y: float, out: list[str]) -> None:
    size = _measure(item)
    if isinstance(item, Box):
        kind = "literal" if item.terminal else "class"
        rx = 12 if item.terminal else 2
        text = f'"{item.text}"' if item.terminal else item.text
        top = y - _BOX_HEIGHT / 2
        out.append(
            f'<rect class="{kind}" x="{x:.1f}" y="{top:.1f}" width="{size.width:.1f}" '
            f'height="{_BOX_HEIGHT}" rx="{rx}"/>'
        )
        out.append(
            f'<text class="{kind}-text" x="{x + size.width / 2:.1f}" y="{y + 4.5:.1f}" '
            f'text-anchor="middle" xml:space="preserve">{escape(text)}</text>'
        )
    elif isinstance(item, Seq):
        cursor = x
        for index, part in enumerate(item.parts):
            if index:
                out.append(_line(cursor, y, cursor + _GAP))
                cursor += _GAP
            _draw(part, cursor, y, out)
            cursor += _measure(part).width
    elif isinstance(item, Stack):
        offset = 0.0
        sizes = [_measure(branch) for branch in item.branches]
        for index, (branch, branch_size) in enumerate(zip(item.branches, sizes, strict=True)):
            row_y = y + offset
            if index == 0:
                out.append(_line(x, y, x + _GAP))
                out.append(_line(x + _GAP + branch_size.width, y, x + size.width))
            else:
                out.append(_curve_down(x, y, row_y, 1))
                out.append(_curve_down(x + size.width, y, row_y, -1))
                out.append(_line(x + _GAP + branch_size.width, row_y, x + size.width - 2 * _RADIUS))
                out.append(_line(x + _RADIUS * 2, row_y, x + _GAP))
            _draw(branch, x + _GAP, row_y, out)
            if index + 1 < len(sizes):
                offset += branch_size.down + _ROW_GAP + sizes[index + 1].up
    elif isinstance(item, Skip):
        _draw_skip(item.body, x, y, size, out)
    elif item.skippable:
        _draw_skip(Loop(item.body, item.label, False), x, y, size, out)
    else:
        body = _measure(item.body)
        out.append(_line(x, y, x + _GAP))
        out.append(_line(x + _GAP + body.width, y, x + size.width))
        _draw(item.body, x + _GAP, y, out)
        bottom = y + body.down + _ROW_GAP
        right = x + size.width - _GAP / 2
        left = x + _GAP / 2
        out.append(
            f'<path class="rail" d="M{right:.1f} {y:.1f} V{bottom - 6:.1f} '
            f'q0 6 -6 6 H{left + 6:.1f} q-6 0 -6 -6 V{y:.1f}"/>'
        )
        middle = (left + right) / 2
        out.append(f'<path class="arrow" d="M{middle + 4:.1f} {bottom - 4:.1f} l-8 4 l8 4 z"/>')
        if item.label:
            out.append(
                f'<text class="count" x="{middle:.1f}" y="{bottom + 15:.1f}" '
                f'text-anchor="middle">{escape(item.label)}</text>'
            )


def _draw_skip(body_item, x: float, y: float, size: _Size, out: list[str]) -> None:
    body = _measure(body_item)
    row_y = y + _ROW_GAP + body.up
    out.append(_line(x, y, x + size.width))
    out.append(_curve_down(x, y, row_y, 1))
    out.append(_curve_down(x + size.width, y, row_y, -1))
    _draw(body_item, x + _GAP, row_y, out)
    out.append(_line(x + _RADIUS * 2, row_y, x + _GAP))
    out.append(_line(x + _GAP + body.width, row_y, x + size.width - _RADIUS * 2))


_STYLE = """
.rail{fill:none;stroke:#7ee787;stroke-width:1.6}
.arrow{fill:#7ee787}
.literal{fill:#10261a;stroke:#7ee787;stroke-width:1.4}
.class{fill:#132036;stroke:#79c0ff;stroke-width:1.4}
text{font:14px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.literal-text{fill:#aff5b4}.class-text{fill:#a5d6ff}
.count{fill:#8b949e;font-size:11px}.caption{fill:#8b949e;font-size:12px}
"""


def to_svg(node: Node, caption: str = "") -> str:
    item = layout(node)
    size = _measure(item)
    margin = 18
    stops = 10
    caption_height = 24 if caption else 0
    width = size.width + 2 * (margin + stops + _GAP)
    width = max(width, len(caption) * 7.4 + 2 * margin)
    height = size.up + size.down + 2 * margin + caption_height
    y = margin + caption_height + size.up
    x = margin + stops + _GAP
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'width="{width:.0f}" height="{height:.0f}" role="img">',
        f"<style>{_STYLE}</style>",
        f'<rect width="{width:.0f}" height="{height:.0f}" rx="8" fill="#0d1117"/>',
    ]
    if caption:
        parts.append(
            f'<text class="caption" x="{margin}" y="{margin + 6}" xml:space="preserve">'
            f"{escape(caption)}</text>"
        )
    parts.append(f'<circle cx="{margin + stops / 2}" cy="{y:.1f}" r="4" fill="#7ee787"/>')
    parts.append(_line(margin + stops / 2, y, x))
    _draw(item, x, y, parts)
    end = x + size.width
    parts.append(_line(end, y, end + _GAP + stops / 2))
    parts.append(f'<circle cx="{end + _GAP + stops / 2:.1f}" cy="{y:.1f}" r="4" fill="#7ee787"/>')
    parts.append("</svg>")
    return "\n".join(parts)
