"""Turn results/records.jsonl into the tables, charts and failure list the README quotes.

uv run --group report python scripts/make_report.py
"""

from __future__ import annotations

import json
import random
import sys
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from regexsmith.evaluate import summarize

DEFAULT_RESULTS = Path(__file__).resolve().parents[1] / "results"
SURFACE = "#0d1117"
TEXT = "#e6edf3"
MUTED = "#8b949e"
GRID = "#21262d"
# Fixed slots from a palette checked for colour-vision separation on this surface.
METHODS = {
    "guided": ("execution-guided beam", "#3987e5", "o"),
    "enumerative": ("enumerator", "#d95926", "s"),
    "filter": ("filter finished only", "#199e70", "^"),
    "top1": ("model alone, no execution", "#c98500", "D"),
}
TIER_LABELS = {1: "1\n1-4", 2: "2\n5-9", 3: "3\n10-15", 4: "4\n16-23", 5: "5\n24-36"}
GRADES = (
    ("consistent", "Consistent with visible examples"),
    ("generalizes", "Also right on hidden strings"),
    ("exact", "Exact language match"),
)


def style(axis) -> None:
    axis.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(MUTED)
    axis.tick_params(colors=MUTED, length=0)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)


def solve_rate_chart(summary: dict, path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True, facecolor=SURFACE)
    tiers = sorted(TIER_LABELS)
    for axis, (grade, title) in zip(axes, GRADES, strict=True):
        style(axis)
        for method, (label, color, marker) in METHODS.items():
            rows = [summary[method][str(t)][grade] for t in tiers]
            rates = [r["rate"] * 100 for r in rows]
            low = [(r["rate"] - r["ci95"][0]) * 100 for r in rows]
            high = [(r["ci95"][1] - r["rate"]) * 100 for r in rows]
            axis.errorbar(
                tiers,
                rates,
                yerr=[low, high],
                color=color,
                marker=marker,
                markersize=6,
                linewidth=2,
                capsize=0,
                elinewidth=1,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                label=label,
            )
        axis.set_title(title, color=TEXT, fontsize=11, loc="left")
        axis.set_xticks(tiers, [TIER_LABELS[t] for t in tiers])
        axis.set_xlabel("tier, target size in tokens", color=MUTED, fontsize=9)
        axis.set_ylim(-3, 103)
    axes[0].set_ylabel("tasks solved, %", color=MUTED, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    legend = figure.legend(
        handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=10, labelcolor=TEXT
    )
    legend.set_in_layout(True)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)


def cactus_chart(records: list[dict], budget: float, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 4), facecolor=SURFACE)
    style(axis)
    endpoints = []
    for method in ("guided", "enumerative", "filter"):
        label, color, _ = METHODS[method]
        rows = [r for r in records if r["method"] == method and r["tier"] > 0]
        times = sorted(r["time"] for r in rows if r["exact"] and r["time"] is not None)
        total = len(rows)
        xs = [0.003, *times, budget]
        ys = [0, *[(i + 1) / total * 100 for i in range(len(times))], len(times) / total * 100]
        axis.step(xs, ys, where="post", color=color, linewidth=2, label=label)
        endpoints.append([ys[-1], label])
    # Direct labels at the right edge, nudged apart when curves finish close together.
    endpoints.sort()
    for lower, upper in pairwise(endpoints):
        upper[0] = max(upper[0], lower[0] + 4)
    for height, label in endpoints:
        axis.annotate(
            label,
            (budget, height),
            xytext=(6, 0),
            textcoords="offset points",
            color=TEXT,
            fontsize=9,
            va="center",
            annotation_clip=False,
        )
    axis.set_xscale("log")
    axis.set_xlim(0.003, budget)
    axis.set_ylim(0, None)
    axis.set_xlabel("seconds per task", color=MUTED, fontsize=9)
    axis.set_ylabel("tasks solved exactly, %", color=MUTED, fontsize=9)
    axis.set_title(
        f"Exact solves as the budget grows (tiers 1-5, {budget:g} s cap)",
        color=TEXT,
        fontsize=11,
        loc="left",
    )
    figure.tight_layout()
    figure.subplots_adjust(right=0.74)
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)


def percent(cell: dict) -> str:
    return f"{cell['rate'] * 100:.1f}"


def table(summary: dict, grade: str, methods: list[str]) -> str:
    names = {**{m: v[0] for m, v in METHODS.items()}, "enumerative-10x": "enumerator, 10x budget"}
    tiers = sorted({t for m in methods for t in summary[m]}, key=lambda t: (t == "0", int(t)))
    head = ["method", *[("field" if t == "0" else f"tier {t}") for t in tiers]]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for method in methods:
        cells = [percent(summary[method][t][grade]) for t in tiers]
        lines.append("| " + " | ".join([names[method], *cells]) + " |")
    return "\n".join(lines)


def failure_report(records: list[dict]) -> str:
    """A reproducible sample of what guided search got wrong, by kind of failure."""
    rng = random.Random(0)
    enumerated = {
        (r["task"], r["method"]): r for r in records if r["method"] in ("guided", "enumerative")
    }
    guided = [r for r in records if r["method"] == "guided"]
    groups = {
        "No regex found inside the budget": [r for r in guided if r["regex"] is None],
        "Passes the visible examples, wrong on hidden strings": [
            r for r in guided if r["consistent"] and not r["generalizes"]
        ],
        "Right on every string tested, but not the target language": [
            r for r in guided if r["generalizes"] and not r["exact"]
        ],
    }
    sections = ["# Failures of the guided search\n"]
    sections.append("Generated by `scripts/make_report.py`. Samples are seeded.\n")
    for title, rows in groups.items():
        sections.append(f"## {title} ({len(rows)} tasks)\n")
        sections.append("| tier | target | answer | enumerator | distinguishing string |")
        sections.append("|---|---|---|---|---|")
        chosen = sorted(rng.sample(rows, min(10, len(rows))), key=lambda r: r["tier"])
        for row in chosen:
            rival = enumerated[(row["task"], "enumerative")]["regex"]
            cells = [
                str(row["tier"]) if row["tier"] else "field",
                f"`{row['target']}`",
                f"`{row['regex']}`" if row["regex"] else "none",
                f"`{rival}`" if rival else "none",
                f"`{row['witness']!r}`" if row["witness"] is not None else "",
            ]
            sections.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
        sections.append("")
    return "\n".join(sections)


def paired(records: list[dict], first: str, second: str, grade: str) -> dict:
    """Per tier: tasks only ``first`` solved, only ``second`` solved, and both.

    Rates on the same tasks overlap more than their intervals suggest, so the honest
    question is how many tasks each method solved that the other did not.
    """
    by_task = {(r["task"], r["method"]): r for r in records}
    report: dict = {}
    for tier in sorted({r["tier"] for r in records}):
        tasks = sorted({r["task"] for r in records if r["tier"] == tier})
        a = [by_task[(t, first)][grade] for t in tasks]
        b = [by_task[(t, second)][grade] for t in tasks]
        report[str(tier)] = {
            f"only_{first}": sum(x and not y for x, y in zip(a, b, strict=True)),
            f"only_{second}": sum(y and not x for x, y in zip(a, b, strict=True)),
            "both": sum(x and y for x, y in zip(a, b, strict=True)),
            "tasks": len(tasks),
        }
    return report


def search_effort(records: list[dict]) -> dict:
    report: dict = {}
    for method in ("filter", "guided"):
        rows = [r for r in records if r["method"] == method]
        widths = sorted(r["width"] for r in rows)
        report[method] = {
            "median_widest_pass": widths[len(widths) // 2],
            "mean_regex_runs": round(sum(r["queries"] for r in rows) / len(rows), 1),
            "mean_sketches_pruned": round(sum(r["pruned"] for r in rows) / len(rows), 1),
            "regex_timeouts": sum(r["timeouts"] for r in rows),
            "tasks_with_a_timeout": sum(r["timeouts"] > 0 for r in rows),
        }
    rows = [r for r in records if r["method"] == "enumerative"]
    report["enumerative"] = {
        "median_candidates_per_second": sorted(r["candidates"] / r["elapsed"] for r in rows)[
            len(rows) // 2
        ],
        "median_max_cost_finished": sorted(r["max_cost"] for r in rows)[len(rows) // 2],
    }
    return report


def main(results: Path = DEFAULT_RESULTS) -> None:
    records = [json.loads(line) for line in (results / "records.jsonl").read_text().splitlines()]
    meta = json.loads((results / "summary.json").read_text())["meta"]
    summary = summarize(records)
    methods = [
        m for m in ("top1", "filter", "guided", "enumerative", "enumerative-10x") if m in summary
    ]
    solve_rate_chart(summary, results / "solve_rates.svg")
    cactus_chart(records, meta["budget_seconds"], results / "budget_curve.svg")
    tables = [
        f"## {name}\n\n{table(summary, grade, methods)}\n"
        for grade, name in (
            ("exact", "Exact language match, %"),
            ("generalizes", "Generalizes, %"),
            ("consistent", "Consistent, %"),
        )
    ]
    (results / "tables.md").write_text("\n".join(tables))
    (results / "failures.md").write_text(failure_report(records))
    comparison = {
        "guided_vs_filter_exact": paired(records, "guided", "filter", "exact"),
        "guided_vs_filter_consistent": paired(records, "guided", "filter", "consistent"),
        "guided_vs_enumerative_exact": paired(records, "guided", "enumerative", "exact"),
        "guided_vs_enumerative_generalizes": paired(
            records, "guided", "enumerative", "generalizes"
        ),
        "search_effort": search_effort(records),
    }
    (results / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULTS)
