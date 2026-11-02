"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / "checkpoints" / "regexsmith.pt"

_GREEN = "\033[32m"
_RED = "\033[31m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _paint(text: str, style: str, enabled: bool) -> str:
    return f"{style}{text}{_RESET}" if enabled else text


def _read_examples(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    positives = list(args.pos or [])
    negatives = list(args.neg or [])
    if args.file:
        for line in Path(args.file).read_text().splitlines():
            if line.startswith("+ "):
                positives.append(line[2:])
            elif line.startswith("- "):
                negatives.append(line[2:])
    if not positives:
        raise SystemExit("at least one positive example is needed (--pos, or '+ text' in --file)")
    return positives, negatives


def _command_solve(args: argparse.Namespace) -> int:
    from .baseline import enumerate_regex
    from .dsl import ALPHABET, to_regex
    from .executor import RegexRunner
    from .railroad import to_text
    from .search import SearchConfig, synthesize
    from .tasks import MAX_STRING
    from .train import load_model

    positives, negatives = _read_examples(args)
    color = sys.stdout.isatty() and not args.no_color
    for text in positives + negatives:
        if len(text) > MAX_STRING or set(text) - set(ALPHABET):
            raise SystemExit(f"{text!r}: examples must be printable ASCII, {MAX_STRING} chars max")

    print(_paint(f"{len(positives)} must match, {len(negatives)} must not", _DIM, color))
    model = load_model(args.checkpoint, args.device)
    config = SearchConfig(mode=args.mode, budget=args.budget)
    with RegexRunner() as runner:
        result = synthesize(model, positives, negatives, config, runner)
    print(
        _paint(
            f"beam width {result.widest_pass}, {result.pruned} sketches pruned, "
            f"{result.queries} regex runs, {result.timeouts} timeouts, {result.elapsed:.2f} s",
            _DIM,
            color,
        )
    )
    if result.best is None:
        print(_paint("no regex passes every example within the budget", _RED, color))
        if not args.fallback:
            return 1
        outcome = enumerate_regex(positives, negatives, args.budget)
        if outcome.regex is None:
            print(_paint("the enumerator found nothing either", _RED, color))
            return 1
        print(_paint("found by the enumerative fallback:", _DIM, color))
        answer = outcome.regex
    else:
        answer = result.best
    print(_paint(to_regex(answer), _GREEN + _BOLD, color))
    for alternative in result.solutions[1:4]:
        print(_paint(f"  also passes: {to_regex(alternative.regex)}", _DIM, color))
    print()
    print(to_text(answer))
    if args.svg:
        from .railroad import to_svg

        Path(args.svg).write_text(to_svg(answer, to_regex(answer)))
    return 0


def _command_draw(args: argparse.Namespace) -> int:
    from .dsl import UnsupportedRegex, parse, to_regex
    from .railroad import to_svg, to_text

    try:
        node = parse(args.regex)
    except UnsupportedRegex as error:
        raise SystemExit(f"cannot draw this regex: {error}") from error
    print(to_text(node))
    if args.svg:
        Path(args.svg).write_text(to_svg(node, to_regex(node)))
    return 0


def _command_train(args: argparse.Namespace) -> int:
    from .model import ModelConfig
    from .train import TrainConfig, export_checkpoint, train

    config = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        max_minutes=args.max_minutes,
        workers=args.workers,
        seed=args.seed,
    )
    train(config, ModelConfig(), Path(args.run_dir), resume=not args.fresh)
    if args.export:
        export_checkpoint(Path(args.run_dir) / "latest.pt", Path(args.export))
    return 0


def _command_benchmark(args: argparse.Namespace) -> int:
    from .evaluate import build_benchmark, save_tasks

    tasks = build_benchmark(args.per_tier, args.seed)
    save_tasks(tasks, Path(args.out))
    print(f"wrote {len(tasks)} tasks to {args.out}")
    return 0


def _command_evaluate(args: argparse.Namespace) -> int:
    import torch

    from .evaluate import load_tasks, run_benchmark

    torch.set_num_threads(args.threads)
    tasks = load_tasks(Path(args.tasks))
    if args.limit:
        tasks = tasks[:: max(len(tasks) // args.limit, 1)]
    run_benchmark(
        tasks,
        args.methods,
        args.budget,
        Path(args.checkpoint),
        args.device,
        args.jobs,
        Path(args.out),
    )
    return 0


def _command_merge(args: argparse.Namespace) -> int:
    from .evaluate import merge_runs

    merge_runs([Path(run) for run in args.runs], Path(args.out))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="regexsmith", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    solve = commands.add_parser("solve", help="write a regex from examples")
    solve.add_argument("-p", "--pos", action="append", help="a string that must match")
    solve.add_argument("-n", "--neg", action="append", help="a string that must not match")
    solve.add_argument("-f", "--file", help="file of lines starting with '+ ' or '- '")
    solve.add_argument("--budget", type=float, default=5.0, help="seconds to search")
    solve.add_argument("--mode", choices=("filter", "guided"), default="guided")
    solve.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    solve.add_argument("--device", default="cpu")
    solve.add_argument("--fallback", action="store_true", help="try the enumerator on failure")
    solve.add_argument("--svg", help="also write a railroad diagram to this file")
    solve.add_argument("--no-color", action="store_true")
    solve.set_defaults(run=_command_solve)

    draw = commands.add_parser("draw", help="draw a railroad diagram of a regex")
    draw.add_argument("regex")
    draw.add_argument("--svg", help="also write an SVG to this file")
    draw.set_defaults(run=_command_draw)

    train = commands.add_parser("train", help="train a model")
    train.add_argument("--run-dir", default="runs/main")
    train.add_argument("--steps", type=int, default=14_000)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--max-minutes", type=float, default=42.0)
    train.add_argument("--workers", type=int, default=5)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint")
    train.add_argument("--export", help="write the deployable checkpoint here when finished")
    train.set_defaults(run=_command_train)

    benchmark = commands.add_parser("benchmark", help="generate the held-out task file")
    benchmark.add_argument("--out", default="data/benchmark.jsonl")
    benchmark.add_argument("--per-tier", type=int, default=150)
    benchmark.add_argument("--seed", type=int, default=2026)
    benchmark.set_defaults(run=_command_benchmark)

    evaluate = commands.add_parser("evaluate", help="run methods over the task file")
    evaluate.add_argument("--tasks", default="data/benchmark.jsonl")
    evaluate.add_argument(
        "--methods",
        nargs="+",
        default=["top1", "filter", "guided", "enumerative", "enumerative-10x"],
    )
    evaluate.add_argument("--budget", type=float, default=5.0)
    evaluate.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    evaluate.add_argument("--device", default="cpu", help="scripts/bench_decode.py says why")
    evaluate.add_argument("--threads", type=int, default=1, help="CPU threads for the model")
    evaluate.add_argument("--jobs", type=int, default=4, help="processes for the enumerator")
    evaluate.add_argument("--out", default="results")
    evaluate.add_argument("--limit", type=int, help="evenly subsample to about this many tasks")
    evaluate.set_defaults(run=_command_evaluate)

    merge = commands.add_parser("merge", help="combine evaluation directories run in parallel")
    merge.add_argument("runs", nargs="+")
    merge.add_argument("--out", default="results")
    merge.set_defaults(run=_command_merge)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
