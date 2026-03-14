# FedToA spec

## Goal
Implement FedToA on top of FedCola with minimal invasive changes.

## Non-goals
- Do not rewrite the federated training framework.
- Do not remove existing CreamFL code.
- Do not redesign dataloaders unless necessary.

## Method summary
### Teacher
- frozen multimodal encoders
- compute class prototypes
- build topology matrix M_k
- compute spectral signature Lambda_k

### Server
- aggregate topology matrices from teacher clients
- compute edge-wise mean and variance
- build sparse confidence-filtered global blueprint

### Student
- frozen backbone
- train modality-adaptive structural prompts only
- optimize L_task + beta * L_topo + gamma * L_spec + eta * L_lip
