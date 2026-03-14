"""FedToA pure method modules."""

from .losses import fedtoa_total_loss, masked_topology_loss, spectral_consistency_loss
from .payloads import FedToAConfig, GlobalTopologyBlueprint, TeacherTopologyPayload
from .prompt import (
    ModalityAdaptiveStructuralPrompt,
    PromptedAttentionAdapter,
    prompt_lipschitz_regularization,
)
from .server_ops import (
    aggregate_topologies_mean,
    aggregate_topologies_var,
    build_confidence_mask,
    build_global_blueprint,
)
from .topology import (
    build_normalized_laplacian,
    build_topology_matrix,
    compute_class_prototypes,
    fuse_joint_prototypes,
    spectral_signature,
)

__all__ = [
    "TeacherTopologyPayload",
    "GlobalTopologyBlueprint",
    "FedToAConfig",
    "compute_class_prototypes",
    "fuse_joint_prototypes",
    "build_topology_matrix",
    "build_normalized_laplacian",
    "spectral_signature",
    "aggregate_topologies_mean",
    "aggregate_topologies_var",
    "build_confidence_mask",
    "build_global_blueprint",
    "masked_topology_loss",
    "spectral_consistency_loss",
    "fedtoa_total_loss",
    "ModalityAdaptiveStructuralPrompt",
    "PromptedAttentionAdapter",
    "prompt_lipschitz_regularization",
]
