"""Prompt modules for FedToA student-side prompt-only adaptation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import List, Union

import torch
from torch import nn


class ModalityAdaptiveStructuralPrompt(nn.Module):
    """Learnable prompt table used for MASP adaptation.

    Args:
        embed_dim: Token embedding dimension ``D``.
        prompt_len: Number of prompt tokens ``P``.
        init_std: Normal initialization std for prompt parameters.

    Attributes:
        prompt: Learnable prompt tensor with shape ``[P, D]``.
    """

    def __init__(self, embed_dim: int, prompt_len: int, init_std: float = 0.02) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        if prompt_len <= 0:
            raise ValueError("prompt_len must be positive.")

        self.embed_dim = embed_dim
        self.prompt_len = prompt_len
        self.prompt = nn.Parameter(torch.empty(prompt_len, embed_dim))
        nn.init.normal_(self.prompt, mean=0.0, std=init_std)

    def expanded_prompt(self, batch_size: int) -> torch.Tensor:
        """Expand prompt table for a batch.

        Args:
            batch_size: Batch size ``B``.

        Returns:
            Prompt tokens with shape ``[B, P, D]``.
        """

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        return self.prompt.unsqueeze(0).expand(batch_size, -1, -1)


class PromptedAttentionAdapter(nn.Module):
    """Adapter that prepends prompts and preserves downstream tensor shape.

    This adapter is intentionally lightweight so it can wrap an existing token
    processor (e.g., attention block) without invasive backbone changes.

    Args:
        base_module: Wrapped token processor mapping ``[B, T, D] -> [B, T, D]``.
        prompt: Prompt module providing tokens of shape ``[B, P, D]``.
        freeze_base: If ``True``, freeze all parameters in ``base_module``.
    """

    def __init__(
        self,
        base_module: nn.Module,
        prompt: ModalityAdaptiveStructuralPrompt,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.base_module = base_module
        self.prompt = prompt

        if freeze_base:
            for param in self.base_module.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Run wrapped module with prompt injection.

        Args:
            x: Input token features with shape ``[B, T, D]``.
            *args: Additional args forwarded to ``base_module``.
            **kwargs: Additional kwargs forwarded to ``base_module``.

        Returns:
            Tensor with shape ``[B, T, D]`` (same sequence length as input).
        """

        if x.ndim != 3:
            raise ValueError("x must have shape [B, T, D].")
        b, _, d = x.shape
        if d != self.prompt.embed_dim:
            raise ValueError("Input embedding dim must match prompt embed_dim.")

        p = self.prompt.expanded_prompt(batch_size=b).to(dtype=x.dtype, device=x.device)
        prompted = torch.cat([p, x], dim=1)
        out = self.base_module(prompted, *args, **kwargs)
        if out.ndim != 3 or out.shape[0] != b or out.shape[2] != d:
            raise ValueError("base_module must return [B, T+P, D].")
        return out[:, self.prompt.prompt_len :, :]


def _iter_prompt_tensors(
    prompt_params: Union[ModalityAdaptiveStructuralPrompt, torch.Tensor, Iterable[torch.Tensor]],
) -> List[torch.Tensor]:
    if isinstance(prompt_params, ModalityAdaptiveStructuralPrompt):
        return [prompt_params.prompt]
    if isinstance(prompt_params, torch.Tensor):
        return [prompt_params]
    if isinstance(prompt_params, Iterable):
        tensors = [p for p in prompt_params if isinstance(p, torch.Tensor)]
        if len(tensors) == 0:
            raise ValueError("Iterable prompt_params must contain at least one tensor.")
        return tensors
    raise TypeError("Unsupported prompt_params type.")


def prompt_lipschitz_regularization(
    prompt_params: Union[ModalityAdaptiveStructuralPrompt, torch.Tensor, Iterable[torch.Tensor]],
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute spectral/Lipschitz regularization over prompt parameters.

    Args:
        prompt_params: Prompt tensor(s), each with shape ``[P, D]`` (or any shape
            flattenable to 2D).
        eps: Small constant for numerical stabilization.

    Returns:
        Scalar tensor equal to the sum of squared spectral norms.
    """

    if eps <= 0:
        raise ValueError("eps must be positive.")

    tensors = _iter_prompt_tensors(prompt_params)
    penalties: List[torch.Tensor] = []
    for param in tensors:
        if param.numel() == 0:
            continue
        mat = param.reshape(param.shape[0], -1)
        spectral = torch.linalg.norm(mat, ord=2)
        penalties.append(spectral.pow(2) + eps)

    if len(penalties) == 0:
        ref = tensors[0]
        return ref.new_tensor(0.0)

    return torch.stack(penalties).sum()
