from typing import List, Optional, Tuple
import warnings

import torch
import torch.nn as nn

from .fedavgclient import FedavgClient
from src.methods.fedtoa.losses import (
    fedtoa_total_loss,
    masked_topology_loss,
    spectral_consistency_loss,
)
from src.methods.fedtoa.payloads import GlobalTopologyBlueprint, TeacherTopologyPayload
from src.methods.fedtoa.prompt import prompt_lipschitz_regularization
from src.methods.fedtoa.topology import (
    build_normalized_laplacian,
    build_topology_matrix,
    compute_class_prototypes,
    fuse_joint_prototypes,
    spectral_signature,
)


class FedtoaClient(FedavgClient):
    """FedToA client with student-side local adaptation support.

    This class reuses ``FedavgClient`` plumbing while adding local methods for:
    - teacher-side topology extraction
    - receiving global topology blueprint
    - student prompt-only local adaptation
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.global_blueprint: Optional[GlobalTopologyBlueprint] = None
        self._fedtoa_warned_missing_task_target = False

    def _num_classes(self) -> int:
        if self.global_blueprint is not None:
            return int(self.global_blueprint.topology_mean.shape[0])
        if hasattr(self.args, "num_classes") and self.args.num_classes is not None:
            return int(self.args.num_classes)
        if hasattr(self.args, "fedtoa_group_count") and self.args.fedtoa_group_count is not None:
            return int(self.args.fedtoa_group_count)
        return 128

    @staticmethod
    def _as_group_ids(candidate, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        if not torch.is_tensor(candidate):
            candidate = torch.as_tensor(candidate)
        if candidate.ndim != 1 or candidate.shape[0] != batch_size:
            return None
        if not (torch.is_floating_point(candidate) or candidate.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)):
            return None
        return candidate.to(device=device, dtype=torch.long)

    def _resolve_topology_groups(self, batch, batch_size: int, device: torch.device):
        if not isinstance(batch, (tuple, list)):
            return None

        # Retrieval dataloaders expose semantic group/image ids as later fields.
        for idx in range(2, len(batch)):
            group_ids = self._as_group_ids(batch[idx], batch_size=batch_size, device=device)
            if group_ids is not None:
                return group_ids

        # Supervised classification fallback: use labels as topology groups.
        if len(batch) >= 2:
            return self._as_group_ids(batch[1], batch_size=batch_size, device=device)
        return None

    def _map_groups_to_table(self, group_ids: torch.Tensor, num_groups: int) -> torch.Tensor:
        group_ids = group_ids.abs().to(dtype=torch.long)
        if group_ids.numel() == 0:
            return group_ids
        if int(group_ids.max()) < num_groups:
            return group_ids
        return torch.remainder(group_ids, num_groups)

    def _prompt_name_tokens(self) -> Tuple[str, ...]:
        tokens = getattr(self.args, "fedtoa_prompt_param_names", None)
        if tokens is None:
            return ("prompt",)
        if isinstance(tokens, str):
            return (tokens,)
        return tuple(tokens)

    def _configure_trainable_params(self) -> List[torch.nn.Parameter]:
        freeze_backbone = bool(getattr(self.args, "freeze_backbone", True))
        prompt_only = bool(getattr(self.args, "fedtoa_prompt_only", True))
        prompt_tokens = self._prompt_name_tokens()

        if freeze_backbone or prompt_only:
            for param in self.model.parameters():
                param.requires_grad = False

        prompt_params: List[torch.nn.Parameter] = []
        for name, param in self.model.named_parameters():
            if any(token in name for token in prompt_tokens):
                param.requires_grad = True
                prompt_params.append(param)

        if len(prompt_params) == 0:
            if not hasattr(self.model, "_fedtoa_prompt_fallback"):
                self.model.register_parameter(
                    "_fedtoa_prompt_fallback",
                    nn.Parameter(torch.zeros(1, device=self.device))
                )
            fallback = getattr(self.model, "_fedtoa_prompt_fallback")
            fallback.requires_grad = True
            prompt_params.append(fallback)

        if not prompt_only:
            for _, param in self.model.named_parameters():
                if not (freeze_backbone and not param.requires_grad):
                    param.requires_grad = True

        return prompt_params

    def _student_forward(self, batch):
        targets: Optional[torch.Tensor] = None

        if isinstance(batch, (tuple, list)):
            if self.modality == "img":
                inputs = batch[0]
                if len(batch) >= 2:
                    candidate = self._as_group_ids(batch[1], batch_size=inputs.shape[0], device=self.device)
                    targets = candidate
            elif self.modality == "txt":
                # Retrieval batches are typically (image, text, ...), while classification
                # text-only batches are commonly (text, labels).
                use_retrieval_layout = len(batch) >= 2 and torch.is_tensor(batch[1]) and batch[1].ndim > 1
                if use_retrieval_layout:
                    inputs = batch[1]
                else:
                    inputs = batch[0]
                    if len(batch) >= 2:
                        candidate = self._as_group_ids(batch[1], batch_size=inputs.shape[0], device=self.device)
                        targets = candidate
            else:
                raise ValueError("FedToA student local adaptation currently supports img or txt modality only.")
        else:
            inputs = batch

        if self.modality == "img":
            inputs = inputs.to(self.device)
            if targets is not None:
                targets = targets.to(self.device)
            logits = self.model([inputs, None])[0]
            feats = self.model([inputs, None], feat_out=True)[0]
            return logits, feats, targets

        if self.modality == "txt":
            inputs = inputs.to(self.device)
            if targets is not None:
                targets = targets.to(self.device)
            logits = self.model([None, inputs])[1]
            feats = self.model([None, inputs], feat_out=True)[1]
            return logits, feats, targets

        raise ValueError("FedToA student local adaptation currently supports img or txt modality only.")

    @torch.no_grad()
    def extract_teacher_topology(self) -> TeacherTopologyPayload:
        """Extract class-level topology/spectral teacher summary from local data."""

        self.model.eval()
        self.model.to(self.device)

        feat_img, feat_txt, all_groups = [], [], []
        fallback_group_offset = 0

        for batch in self.train_loader:
            if self.modality == "img":
                x, y = batch
                x = x.to(self.device)
                f = self.model([x, None], feat_out=True)[0]
                feat_img.append(f)
                groups = self._resolve_topology_groups(batch, batch_size=x.shape[0], device=self.device)
                if groups is None:
                    groups = torch.arange(fallback_group_offset, fallback_group_offset + x.shape[0], device=self.device)
                    fallback_group_offset += x.shape[0]
                all_groups.append(groups)
            elif self.modality == "txt":
                x, y = batch
                x = x.to(self.device)
                f = self.model([None, x], feat_out=True)[1]
                feat_txt.append(f)
                groups = self._resolve_topology_groups(batch, batch_size=x.shape[0], device=self.device)
                if groups is None:
                    groups = torch.arange(fallback_group_offset, fallback_group_offset + x.shape[0], device=self.device)
                    fallback_group_offset += x.shape[0]
                all_groups.append(groups)
            elif self.modality == "img+txt":
                x_img, x_txt, y, _, _ = batch
                x_img = x_img.to(self.device)
                x_txt = x_txt.to(self.device)
                out = self.model([x_img, x_txt], feat_out=True)
                feat_img.append(out[0])
                feat_txt.append(out[1])
                groups = self._resolve_topology_groups(batch, batch_size=x_img.shape[0], device=self.device)
                if groups is None:
                    groups = torch.arange(fallback_group_offset, fallback_group_offset + x_img.shape[0], device=self.device)
                    fallback_group_offset += x_img.shape[0]
                all_groups.append(groups)
            else:
                raise ValueError(f"Unsupported modality: {self.modality}")

        groups = torch.cat(all_groups, dim=0)
        num_classes = self._num_classes()
        mapped_groups = self._map_groups_to_table(groups, num_classes)

        if self.modality == "img":
            proto, support = compute_class_prototypes(torch.cat(feat_img, dim=0), mapped_groups, num_classes)
        elif self.modality == "txt":
            proto, support = compute_class_prototypes(torch.cat(feat_txt, dim=0), mapped_groups, num_classes)
        else:
            proto_img, support_img = compute_class_prototypes(torch.cat(feat_img, dim=0), mapped_groups, num_classes)
            proto_txt, support_txt = compute_class_prototypes(torch.cat(feat_txt, dim=0), mapped_groups, num_classes)
            proto, support = fuse_joint_prototypes(
                proto_img=proto_img,
                proto_txt=proto_txt,
                support_mask_img=support_img,
                support_mask_txt=support_txt,
            )

        tau = float(getattr(self.args, "tau", 0.2))
        diagonal_eps = float(getattr(self.args, "diagonal_eps", 1e-4))
        eig_k = int(getattr(self.args, "eig_k", 4))

        topo = build_topology_matrix(proto, support, tau=tau, zero_diag=True)
        lap = build_normalized_laplacian(topo, eps=diagonal_eps)
        spec = spectral_signature(lap, k=eig_k)

        payload = TeacherTopologyPayload(
            client_id=int(self.id),
            class_ids=torch.nonzero(support, as_tuple=False).flatten(),
            topology=topo.detach().cpu(),
            spectral=spec.detach().cpu(),
            support_mask=support.detach().cpu(),
            num_samples=int(groups.shape[0]),
        )
        self.model.to("cpu")
        return payload

    def set_global_blueprint(self, blueprint: GlobalTopologyBlueprint) -> None:
        """Set global topology blueprint broadcast by server."""

        self.global_blueprint = blueprint

    def local_train_student(self, epochs: int):
        """Run student local adaptation with prompt-only training by default."""

        if self.global_blueprint is None:
            raise ValueError("Global blueprint must be set before local_train_student.")

        self.model.train()
        self.model.to(self.device)

        prompt_params = self._configure_trainable_params()
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = self.optim(trainable_params, **self._refine_optim_args(self.args))

        criterion = self.criterion()
        beta = float(getattr(self.args, "beta_topo", 1.0))
        gamma = float(getattr(self.args, "gamma_spec", 1.0))
        eta = float(getattr(self.args, "eta_lip", 1.0))
        tau = float(getattr(self.args, "tau", 0.2))
        eig_k = int(getattr(self.args, "eig_k", self.global_blueprint.spectral_global.shape[0]))
        diagonal_eps = float(getattr(self.args, "diagonal_eps", 1e-4))

        use_topo = bool(getattr(self.args, "use_topo", True))
        use_spec = bool(getattr(self.args, "use_spec", True))
        use_lip = bool(getattr(self.args, "use_lip", True))

        last_metrics = {}
        num_classes = self._num_classes()
        fallback_group_offset = 0

        for _ in range(epochs):
            for batch in self.train_loader:
                optimizer.zero_grad()
                logits, feats, targets = self._student_forward(batch)

                if targets is None:
                    task_loss = logits.new_tensor(0.0)
                    if not self._fedtoa_warned_missing_task_target:
                        warnings.warn(
                            "FedToA student task targets unavailable in current batch; using zero task loss for smoke execution.",
                            RuntimeWarning,
                        )
                        self._fedtoa_warned_missing_task_target = True
                else:
                    task_loss = criterion(logits.to(targets.device), targets)

                topology_groups = self._resolve_topology_groups(batch, batch_size=feats.shape[0], device=feats.device)
                if topology_groups is None:
                    if targets is not None:
                        topology_groups = targets.to(dtype=torch.long)
                    else:
                        topology_groups = torch.arange(
                            fallback_group_offset,
                            fallback_group_offset + feats.shape[0],
                            device=feats.device,
                            dtype=torch.long,
                        )
                        fallback_group_offset += feats.shape[0]
                topology_groups = self._map_groups_to_table(topology_groups, num_classes)

                local_proto, local_support = compute_class_prototypes(
                    feats,
                    topology_groups,
                    num_classes=num_classes,
                    normalize=True,
                )
                local_topology = build_topology_matrix(local_proto, local_support, tau=tau, zero_diag=True)
                local_laplacian = build_normalized_laplacian(local_topology, eps=diagonal_eps)
                local_spectral = spectral_signature(local_laplacian, k=eig_k)

                active_classes = self.global_blueprint.active_classes.to(self.device).to(dtype=torch.bool)
                support_mask = local_support & active_classes

                topo_loss = task_loss.new_tensor(0.0)
                if use_topo:
                    topo_loss = masked_topology_loss(
                        local_topology=local_topology,
                        global_topology=self.global_blueprint.topology_mean.to(self.device),
                        edge_mask=self.global_blueprint.topology_mask.to(self.device),
                        class_support_mask=support_mask,
                    )

                spec_loss = task_loss.new_tensor(0.0)
                if use_spec:
                    spec_loss = spectral_consistency_loss(
                        local_spectral=local_spectral,
                        global_spectral=self.global_blueprint.spectral_global.to(self.device),
                    )

                lip_loss = task_loss.new_tensor(0.0)
                if use_lip:
                    lip_loss = prompt_lipschitz_regularization(prompt_params)

                total_loss = fedtoa_total_loss(
                    task_loss=task_loss,
                    topo_loss=topo_loss,
                    spec_loss=spec_loss,
                    lip_loss=lip_loss,
                    beta=beta,
                    gamma=gamma,
                    eta=eta,
                )

                total_loss.backward()
                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.args.max_grad_norm)
                optimizer.step()

                last_metrics = {
                    "task_loss": float(task_loss.detach().cpu()),
                    "topo_loss": float(topo_loss.detach().cpu()),
                    "spec_loss": float(spec_loss.detach().cpu()),
                    "lip_loss": float(lip_loss.detach().cpu()),
                    "total_loss": float(total_loss.detach().cpu()),
                }

        self.model.to("cpu")
        return last_metrics

    def update(self):
        return self.local_train_student(getattr(self.args, "E", 1))
