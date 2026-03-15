"""Server-side pure operations for aggregating FedToA topology payloads."""

from __future__ import annotations

import torch
from typing import Optional

from .payloads import GlobalTopologyBlueprint


def aggregate_topologies_mean(topologies: torch.Tensor) -> torch.Tensor:
    """Compute edge-wise mean topology over teachers.

    Args:
        topologies: Tensor of teacher topology matrices, shape ``[T, C, C]``.

    Returns:
        Mean topology tensor with shape ``[C, C]``.
    """

    if topologies.ndim != 3:
        raise ValueError("topologies must have shape [T, C, C].")
    return topologies.mean(dim=0)


def aggregate_topologies_var(topologies: torch.Tensor) -> torch.Tensor:
    """Compute edge-wise population variance over teacher topologies.

    Args:
        topologies: Tensor of teacher topology matrices, shape ``[T, C, C]``.

    Returns:
        Variance tensor with shape ``[C, C]``.
    """

    if topologies.ndim != 3:
        raise ValueError("topologies must have shape [T, C, C].")
    return topologies.var(dim=0, unbiased=False)


def build_confidence_mask(
    topo_mean: torch.Tensor,
    topo_var: torch.Tensor,
    topk_edges: int,
    var_threshold: Optional[float] = None,
) -> torch.Tensor:
    """Build confidence mask from mean/variance with sparsification.

    Strategy:
      1) remove diagonal edges
      2) optionally keep only edges with variance <= var_threshold
      3) among remaining edges, keep top-k by mean value globally
      4) enforce symmetry by OR-ing with transpose

    Args:
        topo_mean: Mean topology matrix, shape ``[C, C]``.
        topo_var: Topology variance matrix, shape ``[C, C]``.
        topk_edges: Number of undirected edges to retain globally.
        var_threshold: Optional variance threshold.

    Returns:
        Boolean confidence mask with shape ``[C, C]``.
    """

    if topo_mean.shape != topo_var.shape:
        raise ValueError("topo_mean and topo_var must have the same shape.")

    c = topo_mean.shape[0]
    device = topo_mean.device
    mask = torch.ones((c, c), device=device, dtype=torch.bool)
    mask.fill_diagonal_(False)

    if var_threshold is not None:
        mask = mask & (topo_var <= var_threshold)

    # Consider only upper-triangular candidate edges for undirected sparsity.
    upper = torch.triu(mask, diagonal=1)
    candidate_idx = torch.nonzero(upper, as_tuple=False)
    if candidate_idx.numel() == 0 or topk_edges <= 0:
        return torch.zeros_like(mask)

    candidate_scores = topo_mean[candidate_idx[:, 0], candidate_idx[:, 1]]
    k_keep = min(topk_edges, candidate_scores.numel())
    topk = torch.topk(candidate_scores, k=k_keep, largest=True)

    selected = candidate_idx[topk.indices]
    conf = torch.zeros_like(mask)
    conf[selected[:, 0], selected[:, 1]] = True
    conf = conf | conf.transpose(0, 1)
    return conf


def build_global_blueprint(
    topo_mean: torch.Tensor,
    confidence_mask: torch.Tensor,
    spectral_list: torch.Tensor,
    class_masks: torch.Tensor,
) -> GlobalTopologyBlueprint:
    """Build FedToA global blueprint from server aggregates.

    Args:
        topo_mean: Aggregated topology mean, shape ``[C, C]``.
        confidence_mask: Confidence edge mask, shape ``[C, C]``.
        spectral_list: Teacher spectral signatures, shape ``[T, K]``.
        class_masks: Teacher class support masks, shape ``[T, C]``.

    Returns:
        ``GlobalTopologyBlueprint`` with masked global topology, global spectral
        mean, and active class mask.
    """

    if spectral_list.ndim != 2:
        raise ValueError("spectral_list must have shape [T, K].")
    if class_masks.ndim != 2:
        raise ValueError("class_masks must have shape [T, C].")

    topo_mask_bool = confidence_mask.to(dtype=torch.bool)
    topology_mean = topo_mean * topo_mask_bool.to(dtype=topo_mean.dtype)
    spectral_global = spectral_list.mean(dim=0)
    active_classes = class_masks.to(dtype=torch.bool).any(dim=0)

    return GlobalTopologyBlueprint(
        topology_mean=topology_mean,
        topology_mask=topo_mask_bool,
        spectral_global=spectral_global,
        active_classes=active_classes,
    )
