# FedToA API Contracts

## TeacherTopologyPayload

- client_id: int
- class_ids: Tensor[C_local]
- topology: Tensor[C_global, C_global]
- spectral: Tensor[K]
- support_mask: Tensor[C_global]
- num_samples: int

Meaning:
- class_ids: classes present on this teacher client
- topology: class-level topology aligned to global class index space
- spectral: teacher spectral signature
- support_mask: indicates which global classes are present locally
- num_samples: total local samples used for extraction

## GlobalTopologyBlueprint

- topology_mean: Tensor[C_global, C_global]
- topology_mask: Tensor[C_global, C_global]
- spectral_global: Tensor[K]
- active_classes: Tensor[C_global]

Meaning:
- topology_mean: aggregated global topology
- topology_mask: confidence-filtered sparse edge mask
- spectral_global: aggregated global spectral signature
- active_classes: classes sufficiently supported across teachers

## FedToAConfig

- tau: float
- eig_k: int
- topk_edges: int
- beta_topo: float
- gamma_spec: float
- eta_lip: float
- prompt_len: int
- diagonal_eps: float

## Required Functions in topology.py

- compute_class_prototypes(feats, labels, num_classes, normalize=True)
- fuse_joint_prototypes(proto_img, proto_txt, support_mask_img, support_mask_txt, normalize=True)
- build_topology_matrix(prototypes, support_mask, tau, zero_diag=True)
- build_normalized_laplacian(topology, eps=1e-4)
- spectral_signature(laplacian, k)

## Required Functions in server_ops.py

- aggregate_topologies_mean(topologies)
- aggregate_topologies_var(topologies)
- build_confidence_mask(topo_mean, topo_var, topk_edges, var_threshold=None)
- build_global_blueprint(topo_mean, confidence_mask, spectral_list, class_masks)

## Required Functions in losses.py

- masked_topology_loss(local_topology, global_topology, edge_mask, class_support_mask, reduction="mean")
- spectral_consistency_loss(local_spectral, global_spectral)
- fedtoa_total_loss(task_loss, topo_loss, spec_loss, lip_loss, beta, gamma, eta)

## Required Components in prompt.py

- ModalityAdaptiveStructuralPrompt
- PromptedAttentionAdapter
- prompt_lipschitz_regularization

## Required Methods in fedtoaclient.py

- extract_teacher_topology()
- set_global_blueprint(blueprint)
- local_train_student(epochs)

## Required Responsibilities in fedtoaserver.py

- collect teacher payloads
- aggregate topology blueprint
- broadcast blueprint to students
- run student local update
- keep existing FedCola algorithms intact
