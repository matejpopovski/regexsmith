import ast
import inspect
import random

import torch

from regexsmith import model as model_module
from regexsmith.data import collate, encode_examples, tier_weights
from regexsmith.dsl import BOS_ID, VOCAB
from regexsmith.grammar import CONTEXT_MASK
from regexsmith.model import HoleInfo, count_parameters
from regexsmith.tasks import sample_task
from regexsmith.train import TrainConfig, export_checkpoint, load_model, loss_on, save_checkpoint

from .conftest import TINY


def _batch(seed=0, tasks=4):
    rng = random.Random(seed)
    return collate([sample_task(rng, 1 + i % 3, 4, 4) for i in range(tasks)])


def test_no_library_transformer_is_used():
    tree = ast.parse(inspect.getsource(model_module))
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not names & {"Transformer", "TransformerDecoder", "MultiheadAttention"}


def test_output_shape(tiny_model):
    batch = _batch()
    logits = tiny_model(batch.chars, batch.labels, batch.inputs, batch.holes)
    assert logits.shape == (*batch.inputs.shape, len(VOCAB))


def test_decoder_cannot_see_the_future(tiny_model):
    batch = _batch()
    changed = batch.inputs.clone()
    changed[:, 3:] = (changed[:, 3:] + 7) % len(VOCAB)
    with torch.no_grad():
        first = tiny_model(batch.chars, batch.labels, batch.inputs, batch.holes)
        second = tiny_model(batch.chars, batch.labels, changed, batch.holes)
    assert torch.allclose(first[:, :3], second[:, :3], atol=1e-5)
    assert not torch.allclose(first[:, 3:], second[:, 3:], atol=1e-5)


def test_incremental_decoding_matches_full_decoding(tiny_model):
    batch = _batch()
    with torch.no_grad():
        memory = tiny_model.encode(batch.chars, batch.labels)
        full = tiny_model.decode(memory, batch.inputs, batch.holes)
        cache = tiny_model.start_decoding(memory)
        for i in range(6):
            holes = HoleInfo(batch.contexts[:, i : i + 1], batch.open_holes[:, i : i + 1])
            logits, cache = tiny_model.step(cache, batch.inputs[:, i : i + 1], holes)
            assert torch.allclose(logits, full[:, i], atol=1e-4)


def test_beams_over_a_shared_memory_match_separate_decoding(tiny_model):
    chars, labels = encode_examples([(["ab1"], ["x"])])
    tokens = torch.tensor([[BOS_ID], [5], [9]])
    contexts = torch.tensor([[0], [1], [3]])
    holes = HoleInfo(contexts, torch.tensor([[1], [2], [3]]))
    with torch.no_grad():
        memory = tiny_model.encode(chars, labels)
        shared, _ = tiny_model.step(tiny_model.start_decoding(memory), tokens, holes)
        for i in range(3):
            one = HoleInfo(contexts[i : i + 1], holes.open_holes[i : i + 1])
            alone, _ = tiny_model.step(tiny_model.start_decoding(memory), tokens[i : i + 1], one)
            assert torch.allclose(alone[0], shared[i], atol=1e-4)


def test_selecting_beams_keeps_them_aligned(tiny_model):
    batch = _batch()
    parents = torch.tensor([2, 2, 0, 3])
    with torch.no_grad():
        memory = tiny_model.encode(batch.chars, batch.labels)
        holes = HoleInfo(batch.contexts[:, :1], batch.open_holes[:, :1])
        cache = tiny_model.start_decoding(memory)
        _, cache = tiny_model.step(cache, batch.inputs[:, :1], holes)
    selected = cache.select(parents)
    assert selected.self_keys[0].shape[0] == 4
    assert torch.equal(selected.self_keys[0][0], cache.self_keys[0][2])


def test_example_order_does_not_matter(tiny_model):
    positives, negatives = ["ab1", "cd22", "e"], ["x", "!!", ""]
    tokens = torch.tensor([[BOS_ID, 5, 9]])
    holes = HoleInfo(torch.zeros_like(tokens), torch.ones_like(tokens))
    reordered = [(positives[::-1], negatives[1:] + negatives[:1])]
    with torch.no_grad():
        first = tiny_model(*encode_examples([(positives, negatives)]), tokens, holes)
        second = tiny_model(*encode_examples(reordered), tokens, holes)
    assert torch.allclose(first, second, atol=1e-4)


def test_labels_reach_the_decoder(tiny_model):
    tokens = torch.tensor([[BOS_ID]])
    holes = HoleInfo(torch.zeros_like(tokens), torch.ones_like(tokens))
    with torch.no_grad():
        first = tiny_model(*encode_examples([(["ab"], ["cd"])]), tokens, holes)
        second = tiny_model(*encode_examples([(["cd"], ["ab"])]), tokens, holes)
    assert not torch.allclose(first, second, atol=1e-4)


def test_padding_does_not_change_the_output(tiny_model):
    short = [(["ab"], ["c"])]
    padded = [(["ab"], ["c"]), (["x" * 30], ["y", "z"])]
    tokens = torch.tensor([[BOS_ID, 5]])
    holes = HoleInfo(torch.zeros_like(tokens), torch.ones_like(tokens))
    with torch.no_grad():
        alone = tiny_model(*encode_examples(short), tokens, holes)
        chars, labels = encode_examples(padded)
        wide = HoleInfo(holes.contexts.expand(2, -1), holes.open_holes.expand(2, -1))
        together = tiny_model(chars, labels, tokens.expand(2, -1), wide)
    assert torch.allclose(alone[0], together[0], atol=1e-4)


def test_loss_falls_on_a_fixed_batch(tiny_model):
    batch = _batch(tasks=8)
    table = torch.from_numpy(CONTEXT_MASK)
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=3e-3)
    tiny_model.train()
    first = None
    for _ in range(60):
        loss = loss_on(tiny_model, batch, table)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        first = first if first is not None else loss.item()
    assert loss.item() < first * 0.6


def test_curriculum_moves_toward_hard_tiers():
    early, late = tier_weights(0.0), tier_weights(1.0)
    assert abs(sum(early) - 1) < 1e-9
    assert abs(sum(late) - 1) < 1e-9
    assert early[0] > late[0]
    assert early[4] < late[4]


def test_export_preserves_predictions(tiny_model, tmp_path):
    optimizer = torch.optim.AdamW(tiny_model.parameters())
    run = tmp_path / "latest.pt"
    save_checkpoint(run, tiny_model, optimizer, 10, 1.0, TrainConfig(), TINY)
    out = tmp_path / "small.pt"
    export_checkpoint(run, out)
    assert out.stat().st_size < run.stat().st_size
    loaded = load_model(out)
    assert count_parameters(loaded) == count_parameters(tiny_model)
    batch = _batch()
    with torch.no_grad():
        before = tiny_model(batch.chars, batch.labels, batch.inputs, batch.holes).softmax(-1)
        after = loaded(batch.chars, batch.labels, batch.inputs, batch.holes).softmax(-1)
    assert (before - after).abs().max() < 1e-2
