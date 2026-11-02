"""A transformer written out by hand.

The encoder respects the shape of the input. Each example string gets its own
self-attention over characters; then one summary token per string attends across the
set of examples. There is no positional information between strings, so reordering the
examples reorders the memory and changes nothing else, a property the tests pin down.
Positions inside a string are embedded from both ends, because regex structure aligns
with both ends of a string and a variable-length run in the middle breaks alignment
from either side alone.

The decoder emits pre-order tokens and cross-attends to every character plus the
string summaries. For beam search it runs incrementally, caching keys and values.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .dsl import ALPHABET, MAX_TOKENS, PAD_ID, VOCAB
from .grammar import NUM_CONTEXTS
from .tasks import MAX_STRING

CHAR_PAD, CHAR_CLS = 0, 1
CHAR_ID = {char: i + 2 for i, char in enumerate(ALPHABET)}
NUM_CHARS = len(ALPHABET) + 2
LABEL_POSITIVE, LABEL_NEGATIVE = 1, 2
_MEMORY_STEP = 32
_MAX_HOLE_FEATURE = 15


def _round_up(value: int, step: int) -> int:
    return (value + step - 1) // step * step


@dataclass(frozen=True, slots=True)
class ModelConfig:
    d_model: int = 256
    heads: int = 8
    char_layers: int = 2
    set_layers: int = 1
    decoder_layers: int = 4
    ff_multiplier: int = 3

    def to_dict(self) -> dict:
        return asdict(self)


class Attention(nn.Module):
    """Multi-head scaled dot-product attention."""

    def __init__(self, d_model: int, heads: int) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads = heads
        self.head_dim = d_model // heads
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)

    def _split(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

    def keys_values(self, context: Tensor) -> tuple[Tensor, Tensor]:
        return self._split(self.key(context)), self._split(self.value(context))

    def attend(self, x: Tensor, keys: Tensor, values: Tensor, mask: Tensor | None) -> Tensor:
        """``mask`` is True where attention is allowed; it broadcasts to (B, H, Q, K)."""
        query = self._split(self.query(x))
        scores = query @ keys.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(~mask, -math.inf)
        mixed = torch.softmax(scores, dim=-1) @ values
        batch, _, length, _ = mixed.shape
        return self.out(mixed.transpose(1, 2).reshape(batch, length, -1))


class FeedForward(nn.Sequential):
    def __init__(self, d_model: int, multiplier: int) -> None:
        super().__init__(
            nn.Linear(d_model, d_model * multiplier),
            nn.GELU(),
            nn.Linear(d_model * multiplier, d_model),
        )


class EncoderLayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(config.d_model)
        self.attention = Attention(config.d_model, config.heads)
        self.norm_ff = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config.d_model, config.ff_multiplier)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        norm = self.norm_attention(x)
        keys, values = self.attention.keys_values(norm)
        x = x + self.attention.attend(norm, keys, values, mask)
        return x + self.ff(self.norm_ff(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(config.d_model)
        self.self_attention = Attention(config.d_model, config.heads)
        self.norm_cross = nn.LayerNorm(config.d_model)
        self.cross_attention = Attention(config.d_model, config.heads)
        self.norm_ff = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config.d_model, config.ff_multiplier)

    def forward(
        self,
        x: Tensor,
        self_keys: Tensor,
        self_values: Tensor,
        self_mask: Tensor | None,
        norm: Tensor,
        memory_keys: Tensor,
        memory_values: Tensor,
        memory_mask: Tensor,
    ) -> Tensor:
        x = x + self.self_attention.attend(norm, self_keys, self_values, self_mask)
        query = self.norm_cross(x)
        if memory_keys.shape[0] == 1 and x.shape[1] == 1:
            # All beams share one memory. Treating beams as positions of a single sequence
            # avoids broadcasting the memory across the batch inside matmul, which was
            # over half of the search's run time.
            cross = self.cross_attention.attend(
                query.transpose(0, 1), memory_keys, memory_values, memory_mask
            ).transpose(0, 1)
        else:
            cross = self.cross_attention.attend(query, memory_keys, memory_values, memory_mask)
        x = x + cross
        return x + self.ff(self.norm_ff(x))


@dataclass
class HoleInfo:
    """For each decoder position: the context of the hole and how many are open."""

    contexts: Tensor
    open_holes: Tensor


@dataclass
class Memory:
    """Encoder output as attention keys and values shared by every decoder layer.

    One projection instead of one per layer saves work in every training step and leaves
    beam search a single pair of tensors to hold on to.
    """

    keys: Tensor
    values: Tensor
    mask: Tensor


@dataclass
class DecodeCache:
    """Encoder memory plus the self-attention keys and values accumulated so far."""

    memory_keys: Tensor
    memory_values: Tensor
    memory_mask: Tensor
    self_keys: list[Tensor | None]
    self_values: list[Tensor | None]
    length: int = 0

    def select(self, parents: Tensor) -> DecodeCache:
        return DecodeCache(
            self.memory_keys,
            self.memory_values,
            self.memory_mask,
            [k.index_select(0, parents) for k in self.self_keys],
            [v.index_select(0, parents) for v in self.self_values],
            self.length,
        )


class RegexTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.char_embedding = nn.Embedding(NUM_CHARS, d, padding_idx=CHAR_PAD)
        self.label_embedding = nn.Embedding(3, d)
        self.position_from_start = nn.Embedding(MAX_STRING + 2, d)
        self.position_from_end = nn.Embedding(MAX_STRING + 2, d)
        self.char_layers = nn.ModuleList(EncoderLayer(config) for _ in range(config.char_layers))
        self.set_layers = nn.ModuleList(EncoderLayer(config) for _ in range(config.set_layers))
        self.string_context = nn.Linear(d, d)
        self.memory_norm = nn.LayerNorm(d)
        self.memory_key = nn.Linear(d, d, bias=False)
        self.memory_value = nn.Linear(d, d, bias=False)

        self.token_embedding = nn.Embedding(len(VOCAB), d, padding_idx=PAD_ID)
        self.token_position = nn.Embedding(MAX_TOKENS + 2, d)
        self.hole_context = nn.Embedding(NUM_CONTEXTS, d)
        self.open_holes = nn.Embedding(_MAX_HOLE_FEATURE + 1, d)
        self.decoder_layers = nn.ModuleList(
            DecoderLayer(config) for _ in range(config.decoder_layers)
        )
        self.output_norm = nn.LayerNorm(d)
        self.output = nn.Linear(d, len(VOCAB))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])

    def encode(self, chars: Tensor, labels: Tensor) -> Memory:
        """``chars`` is (batch, strings, length) with a CLS in column 0; label 0 pads a string."""
        batch, strings, length = chars.shape
        flat = chars.flatten(0, 1)
        char_mask = flat != CHAR_PAD
        starts = torch.arange(length, device=chars.device)
        from_end = (char_mask.sum(dim=1, keepdim=True) - starts).clamp(min=0)
        x = (
            self.char_embedding(flat)
            + self.position_from_start(starts)
            + self.position_from_end(from_end)
            + self.label_embedding(labels.flatten())[:, None]
        )
        attention_mask = char_mask[:, None, None, :]
        for layer in self.char_layers:
            x = layer(x, attention_mask)

        d_model = x.shape[-1]
        present = labels > 0
        summaries = x[:, 0].view(batch, strings, d_model)
        for layer in self.set_layers:
            summaries = layer(summaries, present[:, None, None, :])

        context = self.string_context(summaries).view(batch * strings, 1, d_model)
        states = torch.cat([(x + context).view(batch, strings * length, d_model), summaries], 1)
        # CLS states are excluded: the refined summaries stand in for them.
        char_visible = chars != CHAR_PAD
        char_visible[..., 0] = False
        mask = torch.cat([char_visible.flatten(1, 2), present], dim=1)

        # Most of the grid is padding. Moving real states to the front and cutting the rest
        # saves a large factor in cross-attention; order is irrelevant to attention, and
        # every state already carries its position.
        order = mask.long().argsort(dim=1, descending=True)
        keep = min(_round_up(int(mask.sum(dim=1).max()), _MEMORY_STEP), mask.shape[1])
        order = order[:, :keep]
        states = states.gather(1, order[..., None].expand(-1, -1, d_model))
        memory_states = self.memory_norm(states)
        return Memory(
            self.memory_key(memory_states).view(batch, keep, self.config.heads, -1).transpose(1, 2),
            self.memory_value(memory_states)
            .view(batch, keep, self.config.heads, -1)
            .transpose(1, 2),
            mask.gather(1, order),
        )

    def _embed_tokens(self, tokens: Tensor, holes: HoleInfo, offset: int) -> Tensor:
        """Token, position, and what the grammar says about the hole being filled.

        A flat token sequence hides the tree. Telling the decoder which kind of hole is
        next and how many remain gives it the structure directly, at almost no cost.
        """
        positions = torch.arange(offset, offset + tokens.shape[1], device=tokens.device)
        return (
            self.token_embedding(tokens)
            + self.token_position(positions)
            + self.hole_context(holes.contexts)
            + self.open_holes(holes.open_holes.clamp(max=_MAX_HOLE_FEATURE))
        )

    def decode(self, memory: Memory, tokens: Tensor, holes: HoleInfo) -> Tensor:
        """Teacher-forced logits for ``tokens``, which start with BOS."""
        x = self._embed_tokens(tokens, holes, 0)
        length = tokens.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=tokens.device).tril()
        memory_mask = memory.mask[:, None, None, :]
        for layer in self.decoder_layers:
            norm = layer.norm_self(x)
            keys, values = layer.self_attention.keys_values(norm)
            x = layer(x, keys, values, causal, norm, memory.keys, memory.values, memory_mask)
        return self.output(self.output_norm(x))

    def forward(self, chars: Tensor, labels: Tensor, tokens: Tensor, holes: HoleInfo) -> Tensor:
        return self.decode(self.encode(chars, labels), tokens, holes)

    def start_decoding(self, memory: Memory) -> DecodeCache:
        layers = len(self.decoder_layers)
        return DecodeCache(
            memory.keys,
            memory.values,
            memory.mask[:, None, None, :],
            [None] * layers,
            [None] * layers,
        )

    def step(
        self, cache: DecodeCache, tokens: Tensor, holes: HoleInfo
    ) -> tuple[Tensor, DecodeCache]:
        """Logits for the next position given the last token of each beam, all (beams, 1)."""
        x = self._embed_tokens(tokens, holes, cache.length)
        new_keys, new_values = [], []
        for i, layer in enumerate(self.decoder_layers):
            norm = layer.norm_self(x)
            keys, values = layer.self_attention.keys_values(norm)
            if cache.self_keys[i] is not None:
                keys = torch.cat([cache.self_keys[i], keys], dim=2)
                values = torch.cat([cache.self_values[i], values], dim=2)
            new_keys.append(keys)
            new_values.append(values)
            x = layer(
                x,
                keys,
                values,
                None,
                norm,
                cache.memory_keys,
                cache.memory_values,
                cache.memory_mask,
            )
        logits = self.output(self.output_norm(x))[:, 0]
        updated = DecodeCache(
            cache.memory_keys,
            cache.memory_values,
            cache.memory_mask,
            new_keys,
            new_values,
            cache.length + 1,
        )
        return logits, updated


def default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def legal_logits(logits: Tensor, legal: Tensor) -> Tensor:
    """Log-probabilities restricted to legal tokens, used in training and decoding alike."""
    return F.log_softmax(logits.masked_fill(~legal, -math.inf), dim=-1)
