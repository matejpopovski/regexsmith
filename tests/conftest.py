import pytest
import torch

from regexsmith.model import ModelConfig, RegexTransformer

TINY = ModelConfig(
    d_model=32, heads=2, char_layers=1, set_layers=1, decoder_layers=1, ff_multiplier=2
)


@pytest.fixture
def tiny_model() -> RegexTransformer:
    torch.manual_seed(0)
    return RegexTransformer(TINY).eval()
