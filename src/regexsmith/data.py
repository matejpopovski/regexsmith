"""Turning tasks into tensors, and an endless stream of training batches."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .dsl import BOS_ID, PAD_ID, TOKEN_ID, parse, preorder
from .grammar import contexts_for
from .model import CHAR_CLS, CHAR_ID, CHAR_PAD, LABEL_NEGATIVE, LABEL_POSITIVE, HoleInfo
from .tasks import MAX_STRING, TIERS, Task, sample_task

# MPS compiles a graph per tensor shape, so shapes are rounded up to keep their number small.
_LENGTH_STEP = 8
_TOKEN_STEP = 8
_POOL_BATCHES = 8


def _round_up(value: int, step: int) -> int:
    return (value + step - 1) // step * step


def encode_examples(
    tasks: Sequence[tuple[Sequence[str], Sequence[str]]], shuffle: random.Random | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack ``(positives, negatives)`` pairs into char ids and per-string labels."""
    rows: list[list[tuple[str, int]]] = []
    for positives, negatives in tasks:
        row = [(s, LABEL_POSITIVE) for s in positives] + [(s, LABEL_NEGATIVE) for s in negatives]
        if shuffle is not None:
            shuffle.shuffle(row)
        rows.append(row)
    strings = max(len(row) for row in rows)
    longest = max((len(s) for row in rows for s, _ in row), default=0)
    if longest > MAX_STRING:
        raise ValueError(f"examples are limited to {MAX_STRING} characters")
    length = min(_round_up(longest + 1, _LENGTH_STEP), MAX_STRING + 1)
    chars = np.full((len(rows), strings, length), CHAR_PAD, dtype=np.int64)
    chars[:, :, 0] = CHAR_CLS
    labels = np.zeros((len(rows), strings), dtype=np.int64)
    for i, row in enumerate(rows):
        for j, (text, label) in enumerate(row):
            chars[i, j, 1 : len(text) + 1] = [CHAR_ID[c] for c in text]
            labels[i, j] = label
    return torch.from_numpy(chars), torch.from_numpy(labels)


@dataclass
class Batch:
    chars: torch.Tensor
    labels: torch.Tensor
    inputs: torch.Tensor
    targets: torch.Tensor
    contexts: torch.Tensor
    open_holes: torch.Tensor

    @property
    def holes(self) -> HoleInfo:
        return HoleInfo(self.contexts, self.open_holes)

    def to(self, device: torch.device) -> Batch:
        return Batch(*(t.to(device) for t in vars(self).values()))


def collate(tasks: Sequence[Task], shuffle: random.Random | None = None) -> Batch:
    chars, labels = encode_examples([(t.positives, t.negatives) for t in tasks], shuffle)
    programs = [[TOKEN_ID[tok] for tok in preorder(parse(t.target))] for t in tasks]
    length = _round_up(max(len(p) for p in programs), _TOKEN_STEP)
    inputs = np.full((len(tasks), length), PAD_ID, dtype=np.int64)
    targets = np.full((len(tasks), length), PAD_ID, dtype=np.int64)
    contexts = np.zeros((len(tasks), length), dtype=np.int64)
    holes = np.zeros((len(tasks), length), dtype=np.int64)
    for i, program in enumerate(programs):
        size = len(program)
        inputs[i, 0] = BOS_ID
        inputs[i, 1:size] = program[:-1]
        targets[i, :size] = program
        contexts[i, :size], holes[i, :size] = contexts_for(program)
    return Batch(
        chars,
        labels,
        torch.from_numpy(inputs),
        torch.from_numpy(targets),
        torch.from_numpy(contexts),
        torch.from_numpy(holes),
    )


def tier_weights(progress: float) -> list[float]:
    """The difficulty curriculum: shift mass from easy to hard over the first 60% of training."""
    ramp = min(progress / 0.6, 1.0)
    start = np.array([0.40, 0.35, 0.20, 0.05, 0.00])
    end = np.array([0.08, 0.17, 0.25, 0.25, 0.25])
    return ((1 - ramp) * start + ramp * end).tolist()


class TaskStream(IterableDataset):
    """Generates fresh tasks forever, so the model never sees a training task twice on purpose.

    Workers derive their progress through the curriculum from their own batch count,
    which keeps them stateless and makes resuming a run a matter of passing ``start_step``.
    """

    def __init__(self, batch_size: int, total_steps: int, start_step: int = 0, seed: int = 0):
        self.batch_size = batch_size
        self.total_steps = total_steps
        self.start_step = start_step
        self.seed = seed

    def __iter__(self) -> Iterator[Batch]:
        info = get_worker_info()
        workers, worker_id = (info.num_workers, info.id) if info else (1, 0)
        rng = random.Random(self.seed * 1_000_003 + self.start_step + worker_id)
        step = self.start_step
        tiers = list(TIERS)
        while True:
            weights = tier_weights(step / self.total_steps)
            n_pos = rng.randint(4, 10)
            n_neg = rng.randint(4, 10)
            pool = [
                sample_task(rng, rng.choices(tiers, weights)[0], n_pos, n_neg)
                for _ in range(self.batch_size * _POOL_BATCHES)
            ]
            # Grouping tasks of similar length is the difference between padding every
            # batch to the longest string in 256 tasks and padding each to its own.
            pool.sort(key=lambda t: max(len(s) for s in t.positives + t.negatives))
            starts = list(range(0, len(pool), self.batch_size))
            rng.shuffle(starts)
            for start in starts:
                yield collate(pool[start : start + self.batch_size], rng)
                step += workers
