# FedToA Specification

## Goal

Implement FedToA on top of the FedCola repository with minimal invasive changes.

FedToA is a federated multimodal alignment method for missing-modality clients.
It does NOT rely on external public data for bridging modalities.
Instead, it transfers class-level multimodal structure via topology and spectral summaries.

## Non-goals

- Do not rewrite the federated training framework.
- Do not remove or replace existing FedCola / CreamFL implementations.
- Do not redesign dataloaders unless necessary.
- Do not implement missing-modality generation.
- Do not introduce external public-data distillation into FedToA.
- Do not full-finetune the entire student backbone by default.

## Method Summary

### Teacher clients
Teacher clients are full-modality clients.
They:
1. use frozen multimodal encoders
2. compute class prototypes for image and text modalities
3. fuse them into joint class prototypes
4. construct a class-level topology matrix M_k
5. construct a normalized Laplacian L_k
6. extract the K smallest non-trivial eigenvalues as spectral signature Lambda_k
7. upload topology and spectral summaries to the server

### Server
The server:
1. collects teacher topology matrices
2. computes edge-wise mean and variance
3. filters unstable or low-confidence edges
4. builds a sparse confidence-filtered global topology blueprint M_hat_global
5. averages teacher spectral signatures into Lambda_global
6. broadcasts the global blueprint to student clients

### Student clients
Student clients are missing-modality clients.
They:
1. keep the backbone frozen
2. train modality-adaptive structural prompts (MASP) only
3. build local topology from local class prototypes
4. align local topology to the global blueprint
5. align local spectral summary to Lambda_global
6. apply prompt Lipschitz / spectral regularization
7. optimize:
   L_total = L_task + beta * L_topo + gamma * L_spec + eta * L_lip

## Mathematical Definitions

### Class prototypes
For each class c, compute modality-specific class prototypes:
- z_v^c for image
- z_l^c for text

### Joint prototype
Fuse teacher class prototypes:
z_joint^c = normalize(z_v^c + z_l^c)

### Topology matrix
For classes i and j:
M_k[i, j] = exp(sim(z_i, z_j) / tau)

where sim() is cosine similarity or normalized dot-product.

### Normalized Laplacian
L_k = I - D^(-1/2) M_k D^(-1/2)

where D is the degree matrix of M_k.

### Spectral signature
Keep the K smallest non-trivial eigenvalues of L_k:
Lambda_k = eigvals(L_k)[1 : K+1]

### Server aggregation
Given teacher topologies:
- compute topology_mean
- compute topology_var
- build confidence mask S
- build sparse global blueprint:
  M_hat_global = S ⊙ topology_mean

### Student losses

#### Topology alignment
Align student local topology to global blueprint only on:
- locally supported classes
- server-approved edges

#### Spectral consistency
Match local spectral signature to Lambda_global.

#### Prompt regularization
Apply spectral norm / Lipschitz regularization on prompt parameters.

#### Total objective
L_total = L_task + beta * L_topo + gamma * L_spec + eta * L_lip

## Implementation Constraints

- Topology must be class-level, not sample-level.
- Student backbone is frozen by default.
- Prompt parameters are trainable by default.
- Use support masks so absent classes are excluded from topology alignment.
- Add epsilon / diagonal damping for numerical stability in eigendecomposition.
- Keep all FedToA components configurable.

## Required Config Switches

- use_masp
- use_topo
- use_spec
- use_lip

## Required Hyperparameters

- tau
- eig_k
- topk_edges
- beta_topo
- gamma_spec
- eta_lip
- prompt_len
- diagonal_eps

## Default Experimental Setup

Use these as the default implementation targets:

- datasets: Flickr30k, MS-COCO
- total clients: 32
- teacher clients: 8 multimodal
- student clients: 12 image-only + 12 text-only
- Dirichlet alpha: 0.5
- sparse heterogeneity alpha: 0.1
- participation ratio r: 0.25
- sparse participation ratio r: 0.125
- rounds: 50
- local epochs: 5
- batch size: 64
- optimizer: AdamW
- learning rate: 2e-4
- weight decay: 0.01
- prompt length: 10
- eig_k: 5
- diagonal_eps: 1e-4

## Ablation Targets

Support component-wise ablation with the following staged variants:

A: no MASP, no topo, no spec, no lip
B: MASP only
C: MASP + topo
D: MASP + topo + spec
E: MASP + topo + spec + lip

## File-level Design Intent

The preferred implementation layout is:

- src/methods/fedtoa/payloads.py
- src/methods/fedtoa/topology.py
- src/methods/fedtoa/server_ops.py
- src/methods/fedtoa/losses.py
- src/methods/fedtoa/prompt.py
- src/client/fedtoaclient.py
- src/server/fedtoaserver.py
- scripts/fedtoa/*
- tests/test_fedtoa_*
