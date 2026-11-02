"""Training loop. Resumable, because the GPU is shared and runs get interrupted."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import Batch, TaskStream, collate, tier_weights
from .grammar import CONTEXT_MASK
from .model import ModelConfig, RegexTransformer, count_parameters, default_device, legal_logits
from .tasks import TIERS, sample_task


@dataclass(frozen=True, slots=True)
class TrainConfig:
    steps: int = 14_000
    batch_size: int = 128
    learning_rate: float = 4e-4
    warmup_steps: int = 500
    max_minutes: float = 42.0
    workers: int = 5
    seed: int = 0
    log_every: int = 100
    eval_every: int = 1000
    checkpoint_every: int = 500


def learning_rate(config: TrainConfig, step: int, progress: float) -> float:
    warmup = min(step / config.warmup_steps, 1.0)
    return config.learning_rate * warmup * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def loss_on(model: RegexTransformer, batch: Batch, legal_table: torch.Tensor) -> torch.Tensor:
    logits = model(batch.chars, batch.labels, batch.inputs, batch.holes)
    log_probs = legal_logits(logits, legal_table[batch.contexts])
    return F.nll_loss(log_probs.flatten(0, 1), batch.targets.flatten(), ignore_index=0)


@torch.inference_mode()
def validate(
    model: RegexTransformer, batches: list[Batch], legal_table: torch.Tensor
) -> dict[str, float]:
    """Teacher-forced accuracy per tier on a fixed set of tasks from the training split."""
    model.eval()
    report = {}
    for tier, batch in enumerate(batches, start=1):
        logits = model(batch.chars, batch.labels, batch.inputs, batch.holes)
        predicted = legal_logits(logits, legal_table[batch.contexts]).argmax(dim=-1)
        real = batch.targets != 0
        right = (predicted == batch.targets) | ~real
        report[f"tier{tier}_tokens"] = float((right.sum() - (~real).sum()) / real.sum())
        report[f"tier{tier}_programs"] = float(right.all(dim=1).float().mean())
    model.train()
    return report


def save_checkpoint(path: Path, model, optimizer, step, elapsed, config, model_config) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "elapsed": elapsed,
            "train_config": asdict(config),
            "model_config": model_config.to_dict(),
        },
        temporary,
    )
    temporary.replace(path)


def export_checkpoint(run_checkpoint: Path, out: Path) -> None:
    """Strip training state and store weights as float16.

    Half precision takes a checkpoint from about 24 MB to 12 MB. Evaluation loads the
    exported file, so published results describe the exact weights in the repository.
    """
    state = torch.load(run_checkpoint, map_location="cpu", weights_only=True)
    weights = {name: tensor.half() for name, tensor in state["model"].items()}
    torch.save({"model": weights, "model_config": state["model_config"]}, out)


def load_model(path: Path | str, device: torch.device | str = "cpu") -> RegexTransformer:
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = RegexTransformer(ModelConfig(**state["model_config"]))
    model.load_state_dict({name: tensor.float() for name, tensor in state["model"].items()})
    return model.to(device).eval()


def train(
    config: TrainConfig,
    model_config: ModelConfig,
    run_dir: Path,
    device: torch.device | None = None,
    resume: bool = True,
) -> RegexTransformer:
    device = device or default_device()
    torch.manual_seed(config.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "latest.pt"
    model = RegexTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=0.01, betas=(0.9, 0.98)
    )
    step, elapsed = 0, 0.0
    if resume and checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step, elapsed = state["step"], state["elapsed"]
        print(f"resumed at step {step}, {elapsed / 60:.1f} min in", flush=True)
    print(f"{count_parameters(model):,} parameters on {device}", flush=True)

    legal_table = torch.from_numpy(CONTEXT_MASK).to(device)
    rng = random.Random(config.seed + 12345)
    validation = [
        collate([sample_task(rng, tier) for _ in range(128)]).to(device) for tier in TIERS
    ]
    stream = TaskStream(config.batch_size, config.steps, step, config.seed)
    loader = DataLoader(
        stream,
        batch_size=None,
        num_workers=config.workers,
        prefetch_factor=4 if config.workers else None,
        persistent_workers=config.workers > 0,
        timeout=300 if config.workers else 0,
    )

    log_path = run_dir / "metrics.jsonl"
    time_limit = config.max_minutes * 60
    started = time.perf_counter() - elapsed
    window_loss, window_steps, window_start = 0.0, 0, time.perf_counter()
    model.train()
    for batch in loader:
        batch = batch.to(device)
        elapsed = time.perf_counter() - started
        progress = max(step / config.steps, elapsed / time_limit)
        if progress >= 1.0:
            break
        rate = learning_rate(config, step + 1, progress)
        for group in optimizer.param_groups:
            group["lr"] = rate
        loss = loss_on(model, batch, legal_table)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1
        window_loss += float(loss.detach())
        window_steps += 1

        if step % config.log_every == 0:
            record = {
                "step": step,
                "minutes": round((time.perf_counter() - started) / 60, 2),
                "loss": round(window_loss / window_steps, 4),
                "lr": rate,
                "steps_per_second": round(window_steps / (time.perf_counter() - window_start), 2),
                "tier_weights": [round(w, 3) for w in tier_weights(step / config.steps)],
            }
            if step % config.eval_every == 0:
                record.update(
                    {k: round(v, 4) for k, v in validate(model, validation, legal_table).items()}
                )
            print(json.dumps(record), flush=True)
            with log_path.open("a") as log:
                log.write(json.dumps(record) + "\n")
            window_loss, window_steps, window_start = 0.0, 0, time.perf_counter()
        if step % config.checkpoint_every == 0:
            save_checkpoint(
                checkpoint,
                model,
                optimizer,
                step,
                time.perf_counter() - started,
                config,
                model_config,
            )
    save_checkpoint(
        checkpoint, model, optimizer, step, time.perf_counter() - started, config, model_config
    )
    return model
