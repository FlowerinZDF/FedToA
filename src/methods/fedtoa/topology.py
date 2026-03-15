"""Pure topology and spectral utilities for FedToA.

All operations are class-level and support-mask aware.
"""

from __future__ import annotations

import torch
from typing import Tuple
import torch.nn.functional as F


def compute_class_prototypes(
    feats: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute class-level prototypes from features and integer labels.

    Args:
        feats: Feature tensor with shape ``[N, D]``.
        labels: Class labels with shape ``[N]`` in ``[0, num_classes)``.
        num_classes: Number of global classes ``C``.
        normalize: If ``True``, L2-normalize valid class prototypes.

    Returns:
        Tuple ``(prototypes, support_mask)`` where:
          - ``prototypes`` has shape ``[C, D]``.
          - ``support_mask`` has shape ``[C]`` and dtype ``torch.bool``.
    """

    if feats.ndim != 2:
        raise ValueError("feats must be rank-2 [N, D].")
    if labels.ndim != 1 or labels.shape[0] != feats.shape[0]:
        raise ValueError("labels must be rank-1 [N] matching feats.")

    device = feats.device
    dtype = feats.dtype
    n, d = feats.shape
    if n == 0:
        return torch.zeros(num_classes, d, device=device, dtype=dtype), torch.zeros(
            num_classes, device=device, dtype=torch.bool
        )

    prototypes = torch.zeros(num_classes, d, device=device, dtype=dtype)
    counts = torch.zeros(num_classes, device=device, dtype=dtype)

    prototypes.index_add_(0, labels, feats)
    ones = torch.ones_like(labels, dtype=dtype)
    counts.index_add_(0, labels, ones)

    support_mask = counts > 0
    safe_counts = counts.clamp_min(1.0).unsqueeze(1)
    prototypes = prototypes / safe_counts

    if normalize:
        normalized = F.normalize(prototypes[support_mask], dim=-1)
        prototypes = prototypes.clone()
        prototypes[support_mask] = normalized

    return prototypes, support_mask


def fuse_joint_prototypes(
    proto_img: torch.Tensor,
    proto_txt: torch.Tensor,
    support_mask_img: torch.Tensor,
    support_mask_txt: torch.Tensor,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse image/text class prototypes into a joint class prototype table.

    Args:
        proto_img: Image prototypes, shape ``[C, D]``.
        proto_txt: Text prototypes, shape ``[C, D]``.
        support_mask_img: Image support mask, shape ``[C]``.
        support_mask_txt: Text support mask, shape ``[C]``.
        normalize: If ``True``, L2-normalize supported joint prototypes.

    Returns:
        Tuple ``(joint_proto, joint_support_mask)`` where:
          - ``joint_proto`` has shape ``[C, D]``.
          - ``joint_support_mask`` has shape ``[C]`` (union support).
    """

    if proto_img.shape != proto_txt.shape:
        raise ValueError("proto_img and proto_txt must have same shape [C, D].")

    img_mask = support_mask_img.to(dtype=torch.bool)
    txt_mask = support_mask_txt.to(dtype=torch.bool)
    support = img_mask | txt_mask

    img_w = img_mask.to(dtype=proto_img.dtype).unsqueeze(1)
    txt_w = txt_mask.to(dtype=proto_img.dtype).unsqueeze(1)
    denom = (img_w + txt_w).clamp_min(1.0)
    joint = (proto_img * img_w + proto_txt * txt_w) / denom

    if normalize and torch.any(support):
        joint = joint.clone()
        joint[support] = F.normalize(joint[support], dim=-1)

    return joint, support


def build_topology_matrix(
    prototypes: torch.Tensor,
    support_mask: torch.Tensor,
    tau: float,
    zero_diag: bool = True,
) -> torch.Tensor:
    """Build class-level topology matrix from class prototypes.

    ``M[i, j] = exp(cos_sim(z_i, z_j) / tau)`` for supported classes, 0 otherwise.

    Args:
        prototypes: Class prototype matrix, shape ``[C, D]``.
        support_mask: Supported classes mask, shape ``[C]``.
        tau: Temperature scalar (>0).
        zero_diag: If ``True``, set diagonal entries to zero.

    Returns:
        Topology matrix with shape ``[C, C]``.
    """

    if tau <= 0:
        raise ValueError("tau must be positive.")

    support = support_mask.to(dtype=torch.bool)
    proto = F.normalize(prototypes, dim=-1)
    sim = proto @ proto.transpose(0, 1)
    topo = torch.exp(sim / tau)

    edge_support = support.unsqueeze(0) & support.unsqueeze(1)
    topo = topo * edge_support.to(dtype=topo.dtype)

    if zero_diag:
        topo = topo.clone()
        topo.fill_diagonal_(0.0)

    return topo


def build_normalized_laplacian(topology: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Build numerically stable symmetric normalized graph Laplacian.

    Args:
        topology: Topology/adjacency matrix, shape ``[C, C]``.
        eps: Diagonal damping and degree stabilization constant.

    Returns:
        Normalized Laplacian tensor with shape ``[C, C]``:
        ``L = I - D^{-1/2} (A + eps I) D^{-1/2}``.
    """

    if topology.ndim != 2 or topology.shape[0] != topology.shape[1]:
        raise ValueError("topology must be square [C, C].")

    c = topology.shape[0]
    eye = torch.eye(c, device=topology.device, dtype=topology.dtype)

    sym_topology = 0.5 * (topology + topology.transpose(0, 1))
    damped = sym_topology + eps * eye

    degree = damped.sum(dim=1)
    inv_sqrt_degree = torch.rsqrt(degree.clamp_min(eps))
    norm = inv_sqrt_degree.unsqueeze(1) * damped * inv_sqrt_degree.unsqueeze(0)

    laplacian = eye - norm
    laplacian = 0.5 * (laplacian + laplacian.transpose(0, 1))
    return laplacian


def spectral_signature(laplacian: torch.Tensor, k: int) -> torch.Tensor:
    """Extract the ``k`` smallest non-trivial eigenvalues of Laplacian.

    Args:
        laplacian: Normalized Laplacian matrix, shape ``[C, C]``.
        k: Number of non-trivial eigenvalues to keep.

    Returns:
        Spectral signature tensor with shape ``[k]``.
        If ``C - 1 < k``, values are right-padded by repeating the last available
        non-trivial eigenvalue (or zeros when not available).
    """

    if k <= 0:
        raise ValueError("k must be positive.")

    eigvals = torch.linalg.eigvalsh(laplacian)
    eigvals = eigvals.clamp_min(0.0)

    non_trivial = eigvals[1:] if eigvals.numel() > 1 else eigvals.new_zeros(0)
    if non_trivial.numel() >= k:
        return non_trivial[:k]

    if non_trivial.numel() == 0:
        return eigvals.new_zeros(k)

    pad_value = non_trivial[-1]
    pad = pad_value.expand(k - non_trivial.numel())
    return torch.cat([non_trivial, pad], dim=0)
