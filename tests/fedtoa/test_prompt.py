import pathlib
import sys

import torch
from torch import nn

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from methods.fedtoa.prompt import (  # noqa: E402
    ModalityAdaptiveStructuralPrompt,
    PromptedAttentionAdapter,
    prompt_lipschitz_regularization,
)


class _DummyTokenModule(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def test_prompt_parameter_shapes():
    prompt = ModalityAdaptiveStructuralPrompt(embed_dim=16, prompt_len=10)

    assert prompt.prompt.shape == (10, 16)

    expanded = prompt.expanded_prompt(batch_size=4)
    assert expanded.shape == (4, 10, 16)


def test_adapter_forward_preserves_non_prompt_shape():
    batch_size, seq_len, embed_dim, prompt_len = 2, 5, 8, 3
    x = torch.randn(batch_size, seq_len, embed_dim)

    prompt = ModalityAdaptiveStructuralPrompt(embed_dim=embed_dim, prompt_len=prompt_len)
    adapter = PromptedAttentionAdapter(_DummyTokenModule(embed_dim), prompt)

    out = adapter(x)
    assert out.shape == x.shape


def test_prompt_lipschitz_regularization_returns_scalar():
    prompt = ModalityAdaptiveStructuralPrompt(embed_dim=12, prompt_len=4)

    reg = prompt_lipschitz_regularization(prompt)
    assert isinstance(reg, torch.Tensor)
    assert reg.ndim == 0
    assert reg.item() > 0


def test_prompt_adapter_minimal_trainable_parameters_by_default():
    embed_dim, prompt_len = 6, 2
    prompt = ModalityAdaptiveStructuralPrompt(embed_dim=embed_dim, prompt_len=prompt_len)
    adapter = PromptedAttentionAdapter(_DummyTokenModule(embed_dim), prompt, freeze_base=True)

    trainable = {name for name, p in adapter.named_parameters() if p.requires_grad}
    assert trainable == {"prompt.prompt"}
