# FedToA API contracts

## TeacherTopologyPayload
- client_id: int
- class_ids: Tensor[C_local]
- topology: Tensor[C_global, C_global]
- spectral: Tensor[K]
- support_mask: Tensor[C_global]
- num_samples: int

## GlobalTopologyBlueprint
- topology_mean: Tensor[C_global, C_global]
- topology_mask: Tensor[C_global, C_global]
- spectral_global: Tensor[K]
- active_classes: Tensor[C_global]
