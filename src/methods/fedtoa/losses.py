"""FedToA losses operating on class-level topology and spectral summaries."""

from __future__ import annotations

import torch


def masked_topology_loss(
    local_topology: torch.Tensor,
    global_topology: torch.Tensor,
    edge_mask: torch.Tensor,
    class_support_mask: torch.Tensor,
    reduction: str = "mean",
    normalize: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Topology alignment loss with explicit edge and class support masking.

    Args:
        local_topology: Student local topology, shape ``[C, C]``.
        global_topology: Global topology blueprint values, shape ``[C, C]``.
        edge_mask: Server-approved edge mask, shape ``[C, C]``.
        class_support_mask: Student class support mask, shape ``[C]``.
        reduction: ``"mean"`` or ``"sum"`` over valid edges.
        normalize: If ``True``, divide edge-wise residuals by a symmetric
            magnitude scale ``0.5 * (|local| + |global|) + eps`` before squaring.
            This keeps topology loss numerically comparable to task losses.
        eps: Stabilizer for normalization denominator.

    Returns:
        Scalar topology loss. Returns zero if no valid edge exists.
    """

    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be 'mean' or 'sum'.")

    support = class_support_mask.to(dtype=torch.bool)
    support_edges = support.unsqueeze(0) & support.unsqueeze(1)
    valid_edges = edge_mask.to(dtype=torch.bool) & support_edges

    if normalize:
        denom = 0.5 * (local_topology.abs() + global_topology.abs()) + float(eps)
        diff_sq = ((local_topology - global_topology) / denom).pow(2)
    else:
        diff_sq = (local_topology - global_topology).pow(2)
    selected = diff_sq[valid_edges]
    if selected.numel() == 0:
        return diff_sq.new_tensor(0.0)

    if reduction == "sum":
        return selected.sum()
    return selected.mean()


def spectral_consistency_loss(
    local_spectral: torch.Tensor,
    global_spectral: torch.Tensor,
) -> torch.Tensor:
    """MSE loss between local and global spectral signatures.

    Args:
        local_spectral: Local signature, shape ``[K]``.
        global_spectral: Global signature, shape ``[K]``.

    Returns:
        Scalar spectral consistency loss.
    """

    if local_spectral.shape != global_spectral.shape:
        raise ValueError("local_spectral and global_spectral must match shape.")

    return torch.mean((local_spectral - global_spectral).pow(2))


def fedtoa_total_loss(
    task_loss: torch.Tensor,
    topo_loss: torch.Tensor,
    spec_loss: torch.Tensor,
    lip_loss: torch.Tensor,
    beta: float,
    gamma: float,
    eta: float,
) -> torch.Tensor:
    """Compose FedToA objective.

    ``L_total = L_task + beta * L_topo + gamma * L_spec + eta * L_lip``.

    Args:
        task_loss: Task-specific objective scalar.
        topo_loss: Topology alignment loss scalar.
        spec_loss: Spectral consistency loss scalar.
        lip_loss: Prompt Lipschitz regularization scalar.
        beta: Weight for topology term.
        gamma: Weight for spectral term.
        eta: Weight for Lipschitz term.

    Returns:
        Scalar total FedToA loss.
    """

    return task_loss + beta * topo_loss + gamma * spec_loss + eta * lip_loss
