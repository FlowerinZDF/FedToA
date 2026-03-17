import logging
from typing import Dict, List, Optional, Tuple
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


logger = logging.getLogger(__name__)


_FEDTOA_DEFAULT_PROMPT_TOKENS: Tuple[str, ...] = (
    "prompt",
    "masp",
    "adapter",
    "context",
    "ctx",
    "prefix",
    "softprompt",
    "cls_token",
    "reg_token",
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
        self._fedtoa_warned_nonfinite_fallback_task_loss = False
        self._fedtoa_upload_base_state: Optional[Dict[str, torch.Tensor]] = None

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
            return None, "none"

        # Retrieval dataloaders expose semantic group/image ids as later fields.
        for idx in range(2, len(batch)):
            group_ids = self._as_group_ids(batch[idx], batch_size=batch_size, device=device)
            if group_ids is not None:
                return group_ids, f"batch[{idx}]"

        # Supervised classification fallback: use labels as topology groups.
        if len(batch) >= 2:
            labels = self._as_group_ids(batch[1], batch_size=batch_size, device=device)
            if labels is not None:
                return labels, "batch[1]"
        return None, "none"

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
            return _FEDTOA_DEFAULT_PROMPT_TOKENS
        if isinstance(tokens, str):
            configured = (tokens,)
        else:
            configured = tuple(tokens)

        merged: List[str] = []
        for token in configured + _FEDTOA_DEFAULT_PROMPT_TOKENS:
            t = str(token).strip()
            if t:
                merged.append(t)
        return tuple(dict.fromkeys(merged))

    @staticmethod
    def _normalize_name_tokens(tokens: Tuple[str, ...]) -> Tuple[str, ...]:
        normalized = []
        for token in tokens:
            t = str(token).strip()
            if not t:
                continue
            normalized.append(t.lower())
            normalized.append(t.replace("_", "").lower())
        return tuple(dict.fromkeys(normalized))

    @staticmethod
    def _param_matches_prompt_tokens(name: str, normalized_tokens: Tuple[str, ...]) -> bool:
        lower_name = name.lower()
        compact_name = lower_name.replace("_", "")
        return any(token in lower_name or token in compact_name for token in normalized_tokens)

    def _configure_trainable_params(self) -> List[torch.nn.Parameter]:
        freeze_backbone = bool(getattr(self.args, "freeze_backbone", True))
        prompt_only = bool(getattr(self.args, "fedtoa_prompt_only", True))
        prompt_tokens = self._prompt_name_tokens()
        normalized_prompt_tokens = self._normalize_name_tokens(prompt_tokens)

        if freeze_backbone or prompt_only:
            for param in self.model.parameters():
                param.requires_grad = False

        active_modality_index = 0 if self.modality == "img" else 1 if self.modality == "txt" else None

        prompt_candidates = []
        for name, param in self.model.named_parameters():
            if self._param_matches_prompt_tokens(name, normalized_prompt_tokens):
                prompt_candidates.append(name)

        comm_round = int(getattr(self.args, "fedtoa_comm_round", 0))
        if getattr(self, "_fedtoa_prompt_candidate_log_round", None) != comm_round:
            self._fedtoa_prompt_candidate_log_round = comm_round
            logger.info(
                "[FEDTOA][PROMPT_CANDIDATES] client=%s round=%s candidate_count=%s candidates=%s",
                self.id,
                comm_round,
                len(prompt_candidates),
                prompt_candidates,
            )

        def _is_active_branch_param(name: str) -> bool:
            if active_modality_index is None:
                return True
            return (
                name.startswith("norm.")
                or name.startswith(f"embeddings.{active_modality_index}.")
                or name.startswith(f"blockses.{active_modality_index}.")
                or name.startswith(f"heads.{active_modality_index}.")
            )

        named_params = list(self.model.named_parameters())
        name_to_param = {name: param for name, param in named_params}

        prompt_params: List[torch.nn.Parameter] = []
        matched_prompt_names: List[str] = []
        selection_reasons: Dict[str, str] = {}
        inactive_prompt_matches: List[str] = []
        for name, param in named_params:
            if not self._param_matches_prompt_tokens(name, normalized_prompt_tokens):
                continue
            if not _is_active_branch_param(name):
                inactive_prompt_matches.append(name)
                continue
            param.requires_grad = True
            prompt_params.append(param)
            matched_prompt_names.append(name)
            selection_reasons[name] = "prompt_like"

        lightweight_fallback_names: List[str] = []
        if active_modality_index is not None:
            branch_prefixes = (
                f"heads.{active_modality_index}.",
                f"projections.{active_modality_index}.",
                f"projectors.{active_modality_index}.",
            )
            txt_large_embedding_names = {
                f"embeddings.{active_modality_index}.word_embeddings.weight",
                f"embeddings.{active_modality_index}.position_embeddings.weight",
                f"embeddings.{active_modality_index}.token_type_embeddings.weight",
            }

            def _allow_lightweight_branch_param(candidate_name: str) -> bool:
                if self.modality != "txt":
                    return True
                return candidate_name not in txt_large_embedding_names

            def _add_candidates(candidates: List[str], reason: str) -> None:
                for candidate_name in candidates:
                    if candidate_name in set(matched_prompt_names):
                        continue
                    param = name_to_param.get(candidate_name)
                    if param is None:
                        continue
                    param.requires_grad = True
                    prompt_params.append(param)
                    matched_prompt_names.append(candidate_name)
                    lightweight_fallback_names.append(candidate_name)
                    selection_reasons[candidate_name] = reason

            branch_head_candidates = []
            for name, _ in named_params:
                if not _is_active_branch_param(name):
                    continue
                if any(name.startswith(prefix) for prefix in branch_prefixes):
                    if (name.endswith("weight") or name.endswith("bias")) and _allow_lightweight_branch_param(name):
                        branch_head_candidates.append(name)
            branch_head_candidates = sorted(dict.fromkeys(branch_head_candidates))[:8]
            _add_candidates(branch_head_candidates, reason="branch_head_projection")

            norm_candidates = [name for name in ("norm.weight", "norm.bias") if name in name_to_param]
            _add_candidates(norm_candidates, reason="norm_support")

            block_norm_candidates = [
                name
                for name in name_to_param
                if (
                    name.startswith(f"blockses.{active_modality_index}.")
                    and ".norm" in name
                    and (name.endswith("weight") or name.endswith("bias"))
                )
            ]
            block_norm_candidates = sorted(dict.fromkeys(block_norm_candidates))[:8]
            _add_candidates(block_norm_candidates, reason="branch_norm_support")

            cls_candidates = [
                name
                for name in (
                    f"embeddings.{active_modality_index}.cls_token",
                    f"embeddings.{active_modality_index}.reg_token",
                )
                if name in name_to_param
            ]
            _add_candidates(cls_candidates, reason="branch_token")

        used_fallback = False
        if len(prompt_params) == 0:
            if not hasattr(self.model, "_fedtoa_prompt_fallback"):
                self.model.register_parameter(
                    "_fedtoa_prompt_fallback",
                    nn.Parameter(torch.zeros(1, device=self.device))
                )
            fallback = getattr(self.model, "_fedtoa_prompt_fallback")
            fallback.requires_grad = True
            prompt_params.append(fallback)
            used_fallback = True

        if not prompt_only:
            for _, param in self.model.named_parameters():
                if not (freeze_backbone and not param.requires_grad):
                    param.requires_grad = True

        self._fedtoa_prompt_trainable_names = tuple(matched_prompt_names)
        self._fedtoa_prompt_trainable_reasons = dict(selection_reasons)
        self._fedtoa_prompt_used_fallback = used_fallback
        self._fedtoa_prompt_tokens = tuple(prompt_tokens)
        selected_param_elems = int(sum(name_to_param[name].numel() for name in matched_prompt_names if name in name_to_param))

        logger.info(
            "[FEDTOA][PROMPT_MATCH] client=%s modality=%s configured_tokens=%s matched_count=%s matched_param_elems=%s matched_names=%s matched_reasons=%s inactive_branch_matches=%s lightweight_fallback_names=%s fallback_used=%s",
            self.id,
            self.modality,
            list(prompt_tokens),
            len(matched_prompt_names),
            selected_param_elems,
            matched_prompt_names,
            selection_reasons,
            inactive_prompt_matches,
            lightweight_fallback_names,
            used_fallback,
        )

        return prompt_params

    @staticmethod
    def _loss_requires_grad_summary(
        task_loss: torch.Tensor,
        topo_loss: torch.Tensor,
        spec_loss: torch.Tensor,
        lip_loss: torch.Tensor,
        total_loss: torch.Tensor,
    ) -> Dict[str, object]:
        return {
            "task_loss_requires_grad": bool(task_loss.requires_grad),
            "topo_loss_requires_grad": bool(topo_loss.requires_grad),
            "spec_loss_requires_grad": bool(spec_loss.requires_grad),
            "lip_loss_requires_grad": bool(lip_loss.requires_grad),
            "total_loss_requires_grad": bool(total_loss.requires_grad),
            "total_loss_grad_fn": str(total_loss.grad_fn),
        }

    @staticmethod
    def _groupwise_contrastive_task_loss(feats: torch.Tensor, group_ids: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
        """Group-supervised contrastive fallback task loss.

        Args:
            feats: Tensor[B, D] active-branch normalized features.
            group_ids: Tensor[B] group/class ids.
            temperature: Logit temperature for pairwise similarity.

        Returns:
            Scalar contrastive loss with gradient path to ``feats``.
        """

        if feats.ndim != 2 or feats.shape[0] <= 1:
            return feats.sum() * 0.0

        group_ids = group_ids.to(device=feats.device, dtype=torch.long)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        logits = torch.matmul(feats, feats.transpose(0, 1)) / max(float(temperature), 1e-6)
        logits = logits - logits.max(dim=1, keepdim=True).values

        eye_mask = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
        positive_mask = (group_ids.unsqueeze(0) == group_ids.unsqueeze(1)) & (~eye_mask)
        valid_anchor_mask = positive_mask.any(dim=1)
        if not bool(valid_anchor_mask.any()):
            return feats.sum() * 0.0

        comparison_mask = ~eye_mask
        logits = logits.masked_fill(~comparison_mask, float("-inf"))
        row_logsumexp = torch.logsumexp(logits, dim=1, keepdim=True)
        finite_row_mask = torch.isfinite(row_logsumexp.squeeze(1))

        valid_anchor_mask = valid_anchor_mask & finite_row_mask
        if not bool(valid_anchor_mask.any()):
            return feats.sum() * 0.0

        log_prob = logits - row_logsumexp
        safe_log_prob = torch.where(positive_mask, log_prob, torch.zeros_like(log_prob))
        pos_counts = positive_mask.sum(dim=1).clamp_min(1).to(dtype=feats.dtype)
        per_anchor = -safe_log_prob.sum(dim=1) / pos_counts
        finite_anchor_mask = torch.isfinite(per_anchor)
        valid_anchor_mask = valid_anchor_mask & finite_anchor_mask
        if not bool(valid_anchor_mask.any()):
            return feats.sum() * 0.0

        return per_anchor[valid_anchor_mask].mean()

    @staticmethod
    def _task_fallback_loss(feats: torch.Tensor, group_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Retrieval-oriented fallback student task objective.

        Uses groupwise contrastive loss and, when possible, a light entropy term
        on logits so active branch heads/projections remain task-connected.
        """
        contrastive = FedtoaClient._groupwise_contrastive_task_loss(feats=feats, group_ids=group_ids)
        entropy_term = contrastive.new_tensor(0.0)
        if torch.is_tensor(logits) and logits.ndim == 2 and logits.shape[0] > 0:
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1).mean()
            entropy_term = -0.01 * entropy
        return contrastive + entropy_term

    def _task_connected_nonzero_grad_names(self, task_loss: torch.Tensor, selected_names: List[str]) -> List[str]:
        selected_lookup = {name: p for name, p in self.model.named_parameters() if name in set(selected_names)}
        if not task_loss.requires_grad or len(selected_lookup) == 0:
            return []
        grad_inputs = [p for p in selected_lookup.values() if p.requires_grad]
        grad_names = [n for n, p in selected_lookup.items() if p.requires_grad]
        if len(grad_inputs) == 0:
            return []
        grads = torch.autograd.grad(task_loss, grad_inputs, retain_graph=True, allow_unused=True)
        connected = []
        for name, grad in zip(grad_names, grads):
            if grad is None:
                continue
            if bool(torch.any(grad.detach() != 0)):
                connected.append(name)
        return connected

    def _task_path_diagnostics(self, task_loss: torch.Tensor, selected_names: List[str], active_modality_index: Optional[int]) -> Dict[str, object]:
        selected_lookup = {name: p for name, p in self.model.named_parameters() if name in set(selected_names)}
        selected_connected: List[str] = []
        if task_loss.requires_grad and len(selected_lookup) > 0:
            grad_inputs = [p for p in selected_lookup.values() if p.requires_grad]
            grad_names = [n for n, p in selected_lookup.items() if p.requires_grad]
            if len(grad_inputs) > 0:
                grads = torch.autograd.grad(task_loss, grad_inputs, retain_graph=True, allow_unused=True)
                selected_connected = [
                    name for name, grad in zip(grad_names, grads)
                    if grad is not None and bool(torch.any(grad.detach() != 0))
                ]

        active_head_prefix = None if active_modality_index is None else f"heads.{active_modality_index}."
        active_head_names = []
        active_head_selected = []
        if active_head_prefix is not None:
            active_head_names = [name for name, _ in self.model.named_parameters() if name.startswith(active_head_prefix)]
            active_head_selected = [name for name in selected_names if name.startswith(active_head_prefix)]

        return {
            "task_connected_selected_nonzero": selected_connected,
            "active_head_param_count": len(active_head_names),
            "active_head_selected": active_head_selected,
        }

    def _effective_beta_topo(self) -> float:
        target_beta = float(getattr(self.args, "beta_topo", 1.0))
        warmup_rounds = int(getattr(self.args, "fedtoa_topo_warmup_rounds", 0))
        start_beta = float(getattr(self.args, "fedtoa_topo_warmup_start_beta", 0.0))
        warmup_mode = str(getattr(self.args, "fedtoa_topo_warmup_mode", "linear")).lower()
        comm_round = int(getattr(self.args, "fedtoa_comm_round", 0))

        if warmup_rounds <= 0 or comm_round >= warmup_rounds:
            return target_beta
        if warmup_mode != "linear":
            raise ValueError(f"Unsupported fedtoa_topo_warmup_mode={warmup_mode}.")

        progress = max(float(comm_round), 0.0) / max(float(warmup_rounds), 1.0)
        return start_beta + (target_beta - start_beta) * progress

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
                groups, _ = self._resolve_topology_groups(batch, batch_size=x.shape[0], device=self.device)
                if groups is None:
                    groups = torch.arange(fallback_group_offset, fallback_group_offset + x.shape[0], device=self.device)
                    fallback_group_offset += x.shape[0]
                all_groups.append(groups)
            elif self.modality == "txt":
                x, y = batch
                x = x.to(self.device)
                f = self.model([None, x], feat_out=True)[1]
                feat_txt.append(f)
                groups, _ = self._resolve_topology_groups(batch, batch_size=x.shape[0], device=self.device)
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
                groups, _ = self._resolve_topology_groups(batch, batch_size=x_img.shape[0], device=self.device)
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
        self._fedtoa_upload_base_state = {
            name: tensor.detach().cpu().clone() for name, tensor in self.model.state_dict().items()
        }

        prompt_params = self._configure_trainable_params()
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        prompt_only = bool(getattr(self.args, "fedtoa_prompt_only", True))
        optimizer_params = prompt_params if prompt_only else trainable_params
        optimizer = self.optim(optimizer_params, **self._refine_optim_args(self.args))

        criterion = self.criterion()
        beta_effective = self._effective_beta_topo()
        gamma = float(getattr(self.args, "gamma_spec", 1.0))
        eta = float(getattr(self.args, "eta_lip", 1.0))
        tau = float(getattr(self.args, "tau", 0.2))
        eig_k = int(getattr(self.args, "eig_k", self.global_blueprint.spectral_global.shape[0]))
        diagonal_eps = float(getattr(self.args, "diagonal_eps", 1e-4))

        use_topo = bool(getattr(self.args, "use_topo", True))
        use_spec = bool(getattr(self.args, "use_spec", True))
        use_lip = bool(getattr(self.args, "use_lip", True))
        lip_enabled = use_lip and abs(eta) > 0.0
        diagnostics_enabled = bool(getattr(self.args, "fedtoa_enable_diagnostics", False))

        last_metrics = {}
        num_classes = self._num_classes()
        fallback_group_offset = 0
        trainable_named = [name for name, param in self.model.named_parameters() if param.requires_grad]
        trainable_count = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
        prompt_count = sum(param.numel() for param in prompt_params)
        prompt_param_ids = {id(param) for param in prompt_params}
        non_prompt_trainable_named = [
            name
            for name, param in self.model.named_parameters()
            if param.requires_grad and id(param) not in prompt_param_ids
        ]
        backbone_frozen = len(non_prompt_trainable_named) == 0
        logger.info(
            "[FEDTOA][CLIENT %s] student prompt-only=%s freeze_backbone=%s backbone_frozen=%s trainable_params=%s non_prompt_trainable_params=%s trainable_elems=%s prompt_elems=%s effective_beta_topo=%.6f eta_lip=%.6f lip_enabled=%s round=%s",
            self.id,
            prompt_only,
            bool(getattr(self.args, "freeze_backbone", True)),
            backbone_frozen,
            trainable_named,
            non_prompt_trainable_named,
            trainable_count,
            prompt_count,
            beta_effective,
            eta,
            lip_enabled,
            int(getattr(self.args, "fedtoa_comm_round", 0)),
        )

        active_edge_debug_logged = False
        topo_skip_logged = False
        grad_path_logged = False
        task_path_logged = False
        grad_after_backward_logged = False
        task_connectivity_checked = False
        task_connectivity_nonzero: List[str] = []
        task_connectivity_use_fallback = False

        for epoch_idx in range(epochs):
            for batch_idx, batch in enumerate(self.train_loader):
                if batch_idx == 0:
                    logger.info(
                        "[FEDTOA][HEARTBEAT] client=%s round=%s epoch=%s/%s",
                        self.id,
                        int(getattr(self.args, "fedtoa_comm_round", 0)),
                        epoch_idx + 1,
                        epochs,
                    )
                optimizer.zero_grad()
                logits, feats, targets = self._student_forward(batch)

                topology_groups, topology_group_source = self._resolve_topology_groups(batch, batch_size=feats.shape[0], device=feats.device)
                if topology_groups is None:
                    if targets is not None:
                        topology_groups = targets.to(dtype=torch.long)
                        topology_group_source = "targets"
                    else:
                        topology_groups = torch.arange(
                            fallback_group_offset,
                            fallback_group_offset + feats.shape[0],
                            device=feats.device,
                            dtype=torch.long,
                        )
                        topology_group_source = "arange_fallback"
                        fallback_group_offset += feats.shape[0]
                raw_topology_groups = topology_groups
                topology_groups = self._map_groups_to_table(topology_groups, num_classes)

                task_source = "supervised"
                if targets is None:
                    task_loss = self._task_fallback_loss(feats=feats, group_ids=topology_groups, logits=logits)
                    task_source = "fallback_missing_targets"
                    if not self._fedtoa_warned_missing_task_target:
                        warnings.warn(
                            "FedToA student task targets unavailable in current batch; using retrieval-oriented fallback task path.",
                            RuntimeWarning,
                        )
                        self._fedtoa_warned_missing_task_target = True
                else:
                    task_loss = criterion(logits.to(targets.device), targets)
                    selected_names = list(getattr(self, "_fedtoa_prompt_trainable_names", ()))
                    if diagnostics_enabled and not task_connectivity_checked:
                        task_connectivity_nonzero = self._task_connected_nonzero_grad_names(
                            task_loss=task_loss,
                            selected_names=selected_names,
                        )
                        task_connectivity_use_fallback = len(task_connectivity_nonzero) == 0
                        task_connectivity_checked = True
                    if task_connectivity_use_fallback:
                        task_loss = self._task_fallback_loss(feats=feats, group_ids=topology_groups, logits=logits)
                        task_source = "fallback_disconnected_supervised"
                    else:
                        task_source = "supervised_connected"

                if task_source.startswith("fallback") and not torch.isfinite(task_loss):
                    if not self._fedtoa_warned_nonfinite_fallback_task_loss:
                        logger.warning(
                            "[FEDTOA][LOSS_SANITIZE] client=%s round=%s nonfinite fallback task_loss detected; replacing with zero-gradient-safe scalar.",
                            self.id,
                            int(getattr(self.args, "fedtoa_comm_round", 0)),
                        )
                        self._fedtoa_warned_nonfinite_fallback_task_loss = True
                    task_loss = feats.sum() * 0.0
                    task_source = f"{task_source}_sanitized_nonfinite"

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
                blueprint_mask = self.global_blueprint.topology_mask.to(self.device).to(dtype=torch.bool)

                topo_loss = task_loss.new_tensor(0.0)
                active_edge_count = 0
                if use_topo:
                    valid_edges = (
                        blueprint_mask
                        & support_mask.unsqueeze(0)
                        & support_mask.unsqueeze(1)
                    )
                    valid_edges_no_diag = valid_edges.clone()
                    valid_edges_no_diag.fill_diagonal_(False)
                    active_edge_count = int(valid_edges_no_diag.sum().item())
                    if diagnostics_enabled and not active_edge_debug_logged:
                        support_mask_true_count = int(support_mask.sum().item())
                        support_true_indices = torch.nonzero(support_mask, as_tuple=False).flatten().tolist()
                        support_true_indices_sample = support_true_indices[:12]
                        blueprint_mask_true_count = int(blueprint_mask.sum().item())
                        blueprint_true_edges = torch.nonzero(torch.triu(blueprint_mask, diagonal=1), as_tuple=False)
                        blueprint_edge_sample = blueprint_true_edges[:12].tolist()
                        valid_edges_true_count = int(valid_edges.sum().item())
                        valid_edges_no_diag_true_count = int(valid_edges_no_diag.sum().item())
                        valid_edges_no_diag_sample = torch.nonzero(
                            torch.triu(valid_edges_no_diag, diagonal=1),
                            as_tuple=False,
                        )[:12].tolist()
                        support_set = set(support_true_indices)
                        blueprint_edge_overlap_count = 0
                        for e0, e1 in blueprint_true_edges.tolist():
                            if e0 in support_set and e1 in support_set:
                                blueprint_edge_overlap_count += 1
                        raw_group_sample = raw_topology_groups[:12].detach().cpu().tolist()
                        mapped_group_sample = topology_groups[:12].detach().cpu().tolist()
                        remap_applied = raw_group_sample != mapped_group_sample
                        logger.info(
                            "[FEDTOA][ACTIVE_EDGE_DEBUG] client=%s round=%s support_mask_true_count=%s support_true_indices_sample=%s blueprint_mask_true_count=%s blueprint_edge_sample=%s blueprint_edge_overlap_count=%s valid_edges_true_count=%s valid_edges_no_diag_true_count=%s valid_edges_no_diag_sample=%s active_edge_count=%s local_group_count=%s local_class_count=%s group_source=%s raw_group_sample=%s mapped_group_sample=%s remap_applied=%s",
                            self.id,
                            int(getattr(self.args, "fedtoa_comm_round", 0)),
                            support_mask_true_count,
                            support_true_indices_sample,
                            blueprint_mask_true_count,
                            blueprint_edge_sample,
                            blueprint_edge_overlap_count,
                            valid_edges_true_count,
                            valid_edges_no_diag_true_count,
                            valid_edges_no_diag_sample,
                            active_edge_count,
                            int(torch.unique(topology_groups).numel()),
                            int(local_support.sum().item()),
                            topology_group_source,
                            raw_group_sample,
                            mapped_group_sample,
                            remap_applied,
                        )
                        active_edge_debug_logged = True
                    if active_edge_count > 0:
                        topo_loss = masked_topology_loss(
                            local_topology=local_topology,
                            global_topology=self.global_blueprint.topology_mean.to(self.device),
                            edge_mask=valid_edges_no_diag,
                            class_support_mask=support_mask,
                        )
                    elif not topo_skip_logged:
                        logger.info(
                            "[FEDTOA][TOPO_SKIP] client=%s round=%s reason=no_active_edges topo_loss=0.0",
                            self.id,
                            int(getattr(self.args, "fedtoa_comm_round", 0)),
                        )
                        topo_skip_logged = True

                spec_loss = task_loss.new_tensor(0.0)
                if use_spec:
                    spec_loss = spectral_consistency_loss(
                        local_spectral=local_spectral,
                        global_spectral=self.global_blueprint.spectral_global.to(self.device),
                    )

                lip_loss = task_loss.new_tensor(0.0)
                if lip_enabled:
                    lip_loss = prompt_lipschitz_regularization(prompt_params)

                total_loss = fedtoa_total_loss(
                    task_loss=task_loss,
                    topo_loss=topo_loss,
                    spec_loss=spec_loss,
                    lip_loss=lip_loss,
                    beta=beta_effective,
                    gamma=gamma,
                    eta=eta,
                )

                loss_finite_diag = {
                    "task": bool(torch.isfinite(task_loss).all().item()),
                    "topo": bool(torch.isfinite(topo_loss).all().item()),
                    "spec": bool(torch.isfinite(spec_loss).all().item()),
                    "total": bool(torch.isfinite(total_loss).all().item()),
                }

                grad_diag = self._loss_requires_grad_summary(
                    task_loss=task_loss,
                    topo_loss=topo_loss,
                    spec_loss=spec_loss,
                    lip_loss=lip_loss,
                    total_loss=total_loss,
                )
                if diagnostics_enabled and not grad_path_logged:
                    logger.info(
                        "[FEDTOA][GRAD_PATH] client=%s round=%s matched_names=%s task_requires_grad=%s topo_requires_grad=%s spec_requires_grad=%s lip_requires_grad=%s total_requires_grad=%s total_grad_fn=%s",
                        self.id,
                        int(getattr(self.args, "fedtoa_comm_round", 0)),
                        list(getattr(self, "_fedtoa_prompt_trainable_names", ())),
                        grad_diag["task_loss_requires_grad"],
                        grad_diag["topo_loss_requires_grad"],
                        grad_diag["spec_loss_requires_grad"],
                        grad_diag["lip_loss_requires_grad"],
                        grad_diag["total_loss_requires_grad"],
                        grad_diag["total_loss_grad_fn"],
                    )
                    grad_path_logged = True
                active_modality_index = 0 if self.modality == "img" else 1 if self.modality == "txt" else None
                task_path_diag = {
                    "task_connected_selected_nonzero": task_connectivity_nonzero if diagnostics_enabled else [],
                    "active_head_param_count": 0,
                    "active_head_selected": [],
                }
                if diagnostics_enabled and not task_path_logged:
                    task_path_diag = self._task_path_diagnostics(
                        task_loss=task_loss,
                        selected_names=list(getattr(self, "_fedtoa_prompt_trainable_names", ())),
                        active_modality_index=active_modality_index,
                    )
                    logger.info(
                        "[FEDTOA][TASK_PATH] client=%s round=%s task_source=%s task_requires_grad=%s task_connected_selected_nonzero=%s active_head_param_count=%s active_head_selected=%s",
                        self.id,
                        int(getattr(self.args, "fedtoa_comm_round", 0)),
                        task_source,
                        grad_diag["task_loss_requires_grad"],
                        task_path_diag["task_connected_selected_nonzero"],
                        task_path_diag["active_head_param_count"],
                        task_path_diag["active_head_selected"],
                    )
                    task_path_logged = True

                if not total_loss.requires_grad:
                    nonzero_losses = {
                        "task_loss": float(task_loss.detach().cpu()),
                        "topo_loss_used": float(topo_loss.detach().cpu()),
                        "spec_loss": float(spec_loss.detach().cpu()),
                        "lip_loss": float(lip_loss.detach().cpu()),
                    }
                    raise RuntimeError(
                        "[FEDTOA][GRAD_PATH_BLOCKED] total_loss has no gradient path. "
                        f"loss_values={nonzero_losses} requires_grad={grad_diag} "
                        f"selected_trainable_params={list(getattr(self, '_fedtoa_prompt_trainable_names', ()))}. "
                        "Current selected parameters are not connected to active loss graph."
                    )

                total_loss.backward()
                matched_named_params = [
                    (name, p) for name, p in self.model.named_parameters()
                    if name in set(getattr(self, "_fedtoa_prompt_trainable_names", ()))
                ]
                matched_with_grad = [name for name, p in matched_named_params if p.grad is not None]
                matched_with_nonzero_grad = [
                    name for name, p in matched_named_params
                    if p.grad is not None and bool(torch.any(p.grad.detach() != 0))
                ]
                if diagnostics_enabled and not grad_after_backward_logged:
                    logger.info(
                        "[FEDTOA][GRAD_AFTER_BACKWARD] client=%s round=%s matched_with_grad=%s matched_with_nonzero_grad=%s",
                        self.id,
                        int(getattr(self.args, "fedtoa_comm_round", 0)),
                        matched_with_grad,
                        matched_with_nonzero_grad,
                    )
                    grad_after_backward_logged = True

                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.args.max_grad_norm)
                optimizer.step()

                last_metrics = {
                    "task_loss": float(task_loss.detach().cpu()),
                    "topo_loss_raw": float(topo_loss.detach().cpu()),
                    "topo_loss_used": float(topo_loss.detach().cpu()),
                    "active_edge_count": active_edge_count,
                    "effective_beta_topo": float(beta_effective),
                    "scaled_topo_term": float((beta_effective * topo_loss).detach().cpu()),
                    "spec_loss": float(spec_loss.detach().cpu()),
                    "lip_loss": float(lip_loss.detach().cpu()),
                    "total_loss": float(total_loss.detach().cpu()),
                    "total_loss_requires_grad": bool(total_loss.requires_grad),
                    "task_loss_requires_grad": grad_diag["task_loss_requires_grad"],
                    "topo_loss_requires_grad": grad_diag["topo_loss_requires_grad"],
                    "spec_loss_requires_grad": grad_diag["spec_loss_requires_grad"],
                    "lip_loss_requires_grad": grad_diag["lip_loss_requires_grad"],
                    "matched_with_grad": matched_with_grad,
                    "matched_with_nonzero_grad": matched_with_nonzero_grad,
                    "task_source": task_source,
                    "task_connected_selected_nonzero": task_path_diag["task_connected_selected_nonzero"],
                    "loss_finite": loss_finite_diag,
                }

        final_epoch = int(getattr(self.args, "E", epochs))
        loss_value = float(last_metrics.get("total_loss", 0.0))
        metric_payload = {
            "task_loss": float(last_metrics.get("task_loss", 0.0)),
            "topo_loss_raw": float(last_metrics.get("topo_loss_raw", 0.0)),
            "topo_loss_used": float(last_metrics.get("topo_loss_used", 0.0)),
            "active_edge_count": int(last_metrics.get("active_edge_count", 0)),
            "effective_beta_topo": float(last_metrics.get("effective_beta_topo", beta_effective)),
            "scaled_topo_term": float(last_metrics.get("scaled_topo_term", 0.0)),
            "spec_loss": float(last_metrics.get("spec_loss", 0.0)),
            "lip_loss": float(last_metrics.get("lip_loss", 0.0)),
            "total_loss": loss_value,
            "total_loss_requires_grad": bool(last_metrics.get("total_loss_requires_grad", False)),
            "task_loss_requires_grad": bool(last_metrics.get("task_loss_requires_grad", False)),
            "topo_loss_requires_grad": bool(last_metrics.get("topo_loss_requires_grad", False)),
            "spec_loss_requires_grad": bool(last_metrics.get("spec_loss_requires_grad", False)),
            "lip_loss_requires_grad": bool(last_metrics.get("lip_loss_requires_grad", False)),
            "matched_with_grad": list(last_metrics.get("matched_with_grad", [])),
            "matched_with_nonzero_grad": list(last_metrics.get("matched_with_nonzero_grad", [])),
            "task_source": str(last_metrics.get("task_source", "unknown")),
            "task_connected_selected_nonzero": list(last_metrics.get("task_connected_selected_nonzero", [])),
            "loss_finite": dict(last_metrics.get("loss_finite", {})),
        }
        logger.info(
            "[FEDTOA][TRAIN_METRICS] client=%s round=%s task_source=%s task_loss=%.6f topo_loss_used=%.6f scaled_topo_term=%.6f spec_loss=%.6f lip_loss=%.6f active_edge_count=%s effective_beta_topo=%.6f finite(task/topo/spec/total)=%s/%s/%s/%s prompt_only=%s freeze_backbone=%s total_loss_requires_grad=%s matched_with_grad=%s matched_with_nonzero_grad=%s",
            self.id,
            int(getattr(self.args, "fedtoa_comm_round", 0)),
            metric_payload["task_source"],
            metric_payload["task_loss"],
            metric_payload["topo_loss_used"],
            metric_payload["scaled_topo_term"],
            metric_payload["spec_loss"],
            metric_payload["lip_loss"],
            metric_payload["active_edge_count"],
            metric_payload["effective_beta_topo"],
            bool(metric_payload["loss_finite"].get("task", False)),
            bool(metric_payload["loss_finite"].get("topo", False)),
            bool(metric_payload["loss_finite"].get("spec", False)),
            bool(metric_payload["loss_finite"].get("total", False)),
            prompt_only,
            bool(getattr(self.args, "freeze_backbone", True)),
            metric_payload["total_loss_requires_grad"],
            metric_payload["matched_with_grad"],
            metric_payload["matched_with_nonzero_grad"],
        )

        result_payload = {
            final_epoch: {
                "loss": loss_value,
                "task_loss": metric_payload["task_loss"],
                "topo_loss": metric_payload["topo_loss_used"],
                "topo_loss_raw": metric_payload["topo_loss_raw"],
                "topo_loss_used": metric_payload["topo_loss_used"],
                "active_edge_count": metric_payload["active_edge_count"],
                "effective_beta_topo": metric_payload["effective_beta_topo"],
                "scaled_topo_term": metric_payload["scaled_topo_term"],
                "spec_loss": metric_payload["spec_loss"],
                "lip_loss": metric_payload["lip_loss"],
                "total_loss": metric_payload["total_loss"],
                "metrics": metric_payload,
            }
        }

        self.model.to("cpu")
        return result_payload

    def update(self):
        return self.local_train_student(getattr(self.args, "E", 1))

    def upload(self):
        sd = super().upload()
        prompt_only = bool(getattr(self.args, "fedtoa_prompt_only", True))
        if not prompt_only or self._fedtoa_upload_base_state is None:
            return sd

        prompt_tokens = self._prompt_name_tokens()
        normalized_prompt_tokens = self._normalize_name_tokens(prompt_tokens)
        for key in sd.keys():
            if self._param_matches_prompt_tokens(key, normalized_prompt_tokens):
                continue
            if key in self._fedtoa_upload_base_state:
                sd[key] = self._fedtoa_upload_base_state[key].clone()
        return sd
