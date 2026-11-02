"""Time one incremental decoder step at several beam widths, on each device.

Writes results/decode_speed.json. This is why evaluation decodes on the CPU even though
the model trains on MPS.

    uv run python scripts/bench_decode.py
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch

from regexsmith.cli import DEFAULT_CHECKPOINT
from regexsmith.data import encode_examples
from regexsmith.dsl import BOS_ID
from regexsmith.model import HoleInfo
from regexsmith.tasks import sample_task
from regexsmith.train import load_model

WIDTHS = (1, 16, 64, 256, 1024)
DEPTH = 24
REPEATS = 5


@torch.inference_mode()
def steps_per_second(model, device: str, width: int) -> float:
    task = sample_task(random.Random(0), 3)
    chars, labels = encode_examples([(task.positives, task.negatives)])
    memory = model.encode(chars.to(device), labels.to(device))
    tokens = torch.full((width, 1), BOS_ID, device=device)
    holes = HoleInfo(torch.zeros_like(tokens), torch.ones_like(tokens))
    parents = torch.arange(width, device=device).flip(0)
    best = float("inf")
    for _ in range(REPEATS):
        cache = model.start_decoding(memory)
        started = time.perf_counter()
        for _ in range(DEPTH):
            logits, cache = model.step(cache, tokens, holes)
            logits.argmax(dim=-1).cpu()
            cache = cache.select(parents)
        best = min(best, time.perf_counter() - started)
    return DEPTH / best


def main() -> None:
    torch.set_num_threads(1)
    report = {"depth": DEPTH, "cpu_threads": 1, "steps_per_second": {}}
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for device in devices:
        model = load_model(DEFAULT_CHECKPOINT, device)
        report["steps_per_second"][device] = {
            str(width): round(steps_per_second(model, device, width), 1) for width in WIDTHS
        }
    out = Path(__file__).resolve().parents[1] / "results" / "decode_speed.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
