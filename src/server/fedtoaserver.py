import logging
from collections import defaultdict
from copy import deepcopy
from typing import List, Optional

import torch

from src.methods.fedtoa.server_ops import (
    aggregate_topologies_mean,
    aggregate_topologies_var,
    build_confidence_mask,
    build_global_blueprint,
)
from .fedavgserver import DATASET_2_MODALITY, DATASET_2_TASK, FedavgServer


logger = logging.getLogger(__name__)


class FedtoaServer(FedavgServer):
    """FedToA server orchestration on top of ``FedavgServer``.

    This implementation keeps the existing server stack intact while adding an
    explicit teacher/student flow for topology-centric transfer:
      1) sample participating clients
      2) collect teacher topology payloads
      3) aggregate global topology blueprint
      4) broadcast blueprint to students
      5) run student local updates
      6) aggregate student model updates with existing FedAvg logic
    """

    def __init__(self, args, writer, server_dataset, client_datasets, model_str):
        super().__init__(args, writer, server_dataset, client_datasets, model_str)
        self.latest_blueprint = None



    @staticmethod
    def _tensor_bytes(tensor: torch.Tensor) -> int:
        return int(tensor.numel() * tensor.element_size())

    @classmethod
    def _payload_bytes(cls, payload) -> int:
        """Approximate serialized payload size using raw tensor storage bytes."""

        total = 8  # client_id
        total += cls._tensor_bytes(payload.class_ids)
        total += cls._tensor_bytes(payload.topology)
        total += cls._tensor_bytes(payload.spectral)
        total += cls._tensor_bytes(payload.support_mask)
        total += 8  # num_samples
        return total

    @staticmethod
    def _state_dict_bytes(state_dict) -> int:
        total = 0
        for tensor in state_dict.values():
            if torch.is_tensor(tensor):
                total += int(tensor.numel() * tensor.element_size())
        return total

    def _estimate_round_comm_bytes(self, teacher_payloads, blueprint, student_ids) -> dict:
        """Approximate FedToA communication in bytes for diagnostics.

        Estimation includes:
        - teacher upload payload bytes
        - student upload model state bytes after local update
        - server blueprint broadcast bytes to all students
        """

        teacher_upload = sum(self._payload_bytes(payload) for payload in teacher_payloads)
        student_upload = 0
        student_prompt_upload = 0
        student_frozen_shadow = 0
        for client_id in student_ids:
            client = self.clients[client_id]
            state_dict = client.upload()
            student_upload += self._state_dict_bytes(state_dict)
            stats = getattr(client, "_fedtoa_last_upload_stats", None)
            if isinstance(stats, dict):
                student_prompt_upload += int(stats.get("uploaded_param_bytes", 0))
                student_frozen_shadow += int(stats.get("frozen_param_bytes", 0))
        blueprint_bytes = (
            self._tensor_bytes(blueprint.topology_mean)
            + self._tensor_bytes(blueprint.topology_mask)
            + self._tensor_bytes(blueprint.spectral_global)
            + self._tensor_bytes(blueprint.active_classes)
        )
        server_broadcast = int(blueprint_bytes * len(student_ids))
        total_round = int(teacher_upload + student_upload + server_broadcast)

        return {
            "teacher_payload": int(teacher_upload),
            "student_upload": int(student_upload),
            "student_prompt_upload": int(student_prompt_upload),
            "student_frozen_shadow": int(student_frozen_shadow),
            "server_broadcast": int(server_broadcast),
            "round_total": total_round,
        }

    def _modality_from_layout(self, client_id: int) -> Optional[str]:
        """Resolve client modality from ``args.modalities``/``args.Ks`` layout.

        ``load_datasets`` creates clients in task-order blocks, so this fallback
        can recover the intended per-client modality for smoke/debug settings.
        """

        if not getattr(self.args, "multi_task", False):
            return None
        modalities = getattr(self.args, "modalities", None)
        if not modalities:
            return None

        ks = getattr(self.args, "Ks", None)
        if not ks:
            return None

        num_client_groups = max(len(modalities) - 1, 0)
        if num_client_groups == 0:
            return None

        if len(ks) == 1:
            ks = [int(ks[0])] * num_client_groups
        else:
            ks = [int(v) for v in ks[:num_client_groups]]

        start = 0
        for group_idx, group_size in enumerate(ks):
            end = start + group_size
            if start <= client_id < end:
                return modalities[group_idx]
            start = end
        return None

    def _resolve_client_modality(self, client_id: int) -> Optional[str]:
        client = self.clients[client_id]
        client_modality = getattr(client, "modality", None)
        configured_modality = self._modality_from_layout(client_id)

        if configured_modality is not None and client_modality != configured_modality:
            logger.warning(
                "[FEDTOA][ROLE] modality mismatch for client %s: client.modality=%s configured=%s; using configured value.",
                client_id,
                client_modality,
                configured_modality,
            )
            return configured_modality
        return configured_modality or client_modality

    def _teacher_client_ids(self, selected_ids: List[int]) -> List[int]:
        configured_ids = getattr(self.args, "fedtoa_teacher_ids", None)
        if configured_ids is not None:
            configured = set(int(client_id) for client_id in configured_ids)
            return [client_id for client_id in selected_ids if client_id in configured]

        return [
            client_id
            for client_id in selected_ids
            if self._resolve_client_modality(client_id) == "img+txt"
        ]

    def _student_client_ids(self, selected_ids: List[int], teacher_ids: List[int]) -> List[int]:
        teacher_set = set(teacher_ids)
        return [client_id for client_id in selected_ids if client_id not in teacher_set]

    def _prepare_client_for_round(self, client):
        if client.model is None:
            client.download(self.global_models)

        client.args.lr = self.curr_lr

        if getattr(self.args, "freeze_modality", "none") != "none":
            if client.modality == self.args.freeze_modality:
                if self.round <= (self.args.freeze_rounds + self.args.warmup_rounds) and self.round > self.args.warmup_rounds:
                    self._freeze_shared_params(client)
                elif self.round > (self.args.freeze_rounds + self.args.warmup_rounds):
                    self._unfreeze_params(client)

    def _bind_resolved_client_modality(self, client_id: int) -> Optional[str]:
        """Bind the server-resolved modality onto the live client instance."""

        resolved_modality = self._resolve_client_modality(client_id)
        if resolved_modality is not None:
            self.clients[client_id].modality = resolved_modality
        return resolved_modality

    @staticmethod
    def _expected_model_layout(resolved_modality: Optional[str]):
        if resolved_modality == "img":
            return ["img", None]
        if resolved_modality == "txt":
            return [None, "txt"]
        if resolved_modality == "img+txt":
            return ["img", "txt"]
        return None

    def _align_client_model_layout(self, client, resolved_modality: Optional[str]):
        """Align runtime model modality slots with resolved FedToA role modality.

        FedAvg downloads models per dataset key, which can leave student clients
        with multimodal model layouts even after ``client.modality`` is rebound.
        FedCola unimodal behavior requires the inactive slot to be ``None`` so
        ``self.model([inputs, None])`` / ``self.model([None, inputs])`` is valid.
        """

        if client.model is None or not hasattr(client.model, "modalities"):
            return

        expected_layout = self._expected_model_layout(resolved_modality)
        if expected_layout is None:
            return

        current_layout = list(client.model.modalities)
        if current_layout == expected_layout:
            return

        client.model.modalities = expected_layout
        logger.info(
            "[FEDTOA][LAYOUT] client %s modality=%s model.modalities %s -> %s",
            client.id,
            resolved_modality,
            current_layout,
            expected_layout,
        )

    def _collect_teacher_payloads(self, teacher_ids: List[int]):
        payloads = []
        for client_id in teacher_ids:
            client = self.clients[client_id]
            resolved_modality = self._bind_resolved_client_modality(client_id)
            self._prepare_client_for_round(client)
            self._align_client_model_layout(client, resolved_modality)
            payload = client.extract_teacher_topology()
            payloads.append(payload)
        return payloads

    def _aggregate_teacher_blueprint(self, payloads):
        if len(payloads) == 0:
            return None

        topologies = torch.stack([payload.topology.float() for payload in payloads], dim=0)
        topologies = 0.5 * (topologies + topologies.transpose(-1, -2))
        topologies = topologies.clone()
        topologies.diagonal(dim1=-2, dim2=-1).zero_()

        spectral_list = torch.stack([payload.spectral.float() for payload in payloads], dim=0)
        class_masks = torch.stack([payload.support_mask.to(dtype=torch.bool) for payload in payloads], dim=0)

        topo_mean = aggregate_topologies_mean(topologies)
        topo_var = aggregate_topologies_var(topologies)

        # Restrict blueprint edge candidacy to classes supported by at least one
        # teacher so student support-vs-blueprint overlap can be meaningful.
        active_classes = class_masks.any(dim=0)
        active_pair_mask = active_classes.unsqueeze(0) & active_classes.unsqueeze(1)
        topo_mean_masked = topo_mean.masked_fill(~active_pair_mask, float("-inf"))

        c = topo_mean.shape[0]
        default_topk = max((c * (c - 1)) // 2, 1)
        raw_topk = getattr(self.args, "topk_edges", None)
        topk_edges = default_topk if raw_topk is None else int(raw_topk)
        var_threshold = getattr(self.args, "fedtoa_var_threshold", None)

        confidence_mask = build_confidence_mask(
            topo_mean=topo_mean_masked,
            topo_var=topo_var,
            topk_edges=topk_edges,
            var_threshold=var_threshold,
        )
        confidence_mask = confidence_mask & active_pair_mask

        retained_edges = int(torch.triu(confidence_mask, diagonal=1).sum().item())
        max_edges = max((c * (c - 1)) // 2, 1)
        retained_density = float(retained_edges / max_edges)
        logger.info(
            "[FEDTOA][BLUEPRINT] round=%s teachers=%s topk_edges=%s retained_edges=%s retained_density=%.6f active_classes=%s var_threshold_active=%s support_mask_active=%s confidence_mask_active=%s var_threshold=%s",
            str(self.round).zfill(4),
            len(payloads),
            topk_edges,
            retained_edges,
            retained_density,
            int(active_classes.sum().item()),
            var_threshold is not None,
            bool(class_masks.to(dtype=torch.bool).any().item()),
            bool(confidence_mask.any().item()),
            var_threshold,
        )

        blueprint = build_global_blueprint(
            topo_mean=topo_mean,
            confidence_mask=confidence_mask,
            spectral_list=spectral_list,
            class_masks=class_masks,
        )
        self.results[self.round]["fedtoa_blueprint"] = {
            "teacher_payloads": len(payloads),
            "topk_edges": int(topk_edges),
            "retained_edges": retained_edges,
            "retained_edge_density": retained_density,
            "var_threshold": None if var_threshold is None else float(var_threshold),
            "var_threshold_active": bool(var_threshold is not None),
            "confidence_mask_active": bool(confidence_mask.any().item()),
            "support_mask_active": bool(blueprint.active_classes.any().item()),
        }
        return blueprint

    def _run_student_updates(self, student_ids: List[int], blueprint):
        update_sizes = {}
        update_results = {}

        for client_id in student_ids:
            client = self.clients[client_id]
            resolved_modality = self._bind_resolved_client_modality(client_id)
            self._prepare_client_for_round(client)
            self._align_client_model_layout(client, resolved_modality)
            client.args.fedtoa_comm_round = int(self.round)
            client.set_global_blueprint(blueprint)
            update_results[client_id] = client.local_train_student(getattr(client.args, "E", 1))
            update_sizes[client_id] = len(client.training_set)

        if len(update_sizes) > 0:
            self.results[self.round]["clients_updated"] = self._log_results(
                update_sizes,
                update_results,
                eval=False,
                participated=True,
                save_raw=False,
            )

        return update_sizes

    def update(self):
        # ``equal_sampled`` can yield duplicated IDs when users pass repeated
        # dataset names (e.g., modality partitions of the same dataset). FedToA
        # orchestration is role-based per client, so deduplicate before
        # teacher/student splitting.
        selected_ids = sorted(set(self._sample_clients()))
        teacher_ids = self._teacher_client_ids(selected_ids)
        student_ids = self._student_client_ids(selected_ids, teacher_ids)

        logger.info(
            "[FEDTOA][PRECHECK] round=%s dataset=%s algorithm=fedtoa selected=%s teachers=%s students=%s topk_edges=%s beta_topo=%.6f gamma_spec=%.6f eta_lip=%.6f warmup=(rounds:%s,start_beta:%.6f,mode:%s) prompt_only=%s freeze_backbone=%s",
            str(self.round).zfill(4),
            getattr(self, "dataset", "unknown"),
            len(selected_ids),
            len(teacher_ids),
            len(student_ids),
            getattr(self.args, "topk_edges", None),
            float(getattr(self.args, "beta_topo", 1.0)),
            float(getattr(self.args, "gamma_spec", 1.0)),
            float(getattr(self.args, "eta_lip", 1.0)),
            int(getattr(self.args, "fedtoa_topo_warmup_rounds", 0)),
            float(getattr(self.args, "fedtoa_topo_warmup_start_beta", 0.0)),
            str(getattr(self.args, "fedtoa_topo_warmup_mode", "linear")),
            bool(getattr(self.args, "fedtoa_prompt_only", True)),
            bool(getattr(self.args, "freeze_backbone", True)),
        )

        self.results[self.round]["fedtoa_selected"] = {
            "all": selected_ids,
            "teachers": teacher_ids,
            "students": student_ids,
        }

        role_tokens = []
        teacher_set = set(teacher_ids)
        for client_id in selected_ids:
            resolved_modality = self._resolve_client_modality(client_id)
            role = "teacher" if client_id in teacher_set else "student"
            role_tokens.append(f"{client_id}:{resolved_modality}:{role}")
        logger.info(
            "[FEDTOA] Round %s selected clients => %s",
            str(self.round).zfill(4),
            ", ".join(role_tokens) if role_tokens else "none",
        )

        blueprint = None
        if len(teacher_ids) > 0 and len(student_ids) > 0:
            payloads = self._collect_teacher_payloads(teacher_ids)
            blueprint = self._aggregate_teacher_blueprint(payloads)
            self.latest_blueprint = blueprint
            student_update_sizes = self._run_student_updates(student_ids, blueprint)
            comm = self._estimate_round_comm_bytes(payloads, blueprint, student_ids)
            cumulative = int(self.results[self.round - 1].get("fedtoa_comm", {}).get("cumulative_total", 0)) if self.round > 0 else 0
            comm["cumulative_total"] = int(cumulative + comm["round_total"])
            self.results[self.round]["fedtoa_comm"] = comm
            logger.info(
                "[FEDTOA][COMM] round=%s teacher_payload_bytes=%s student_upload_bytes=%s student_prompt_upload_bytes=%s student_frozen_shadow_bytes=%s server_broadcast_bytes=%s round_total_bytes=%s cumulative_total_bytes=%s",
                str(self.round).zfill(4),
                comm["teacher_payload"],
                comm["student_upload"],
                comm.get("student_prompt_upload", 0),
                comm.get("student_frozen_shadow", 0),
                comm["server_broadcast"],
                comm["round_total"],
                comm["cumulative_total"],
            )
        else:
            logger.warning(
                "[FEDTOA] Round %s skipped FedToA orchestration because teachers=%s students=%s.",
                str(self.round).zfill(4),
                len(teacher_ids),
                len(student_ids),
            )
            student_update_sizes = {}
            self.results[self.round]["fedtoa_comm"] = {"teacher_payload": 0, "student_upload": 0, "student_prompt_upload": 0, "student_frozen_shadow": 0, "server_broadcast": 0, "round_total": 0, "cumulative_total": int(self.results[self.round - 1].get("fedtoa_comm", {}).get("cumulative_total", 0)) if self.round > 0 else 0}

        if self.args.fedavg_eval and len(student_ids) > 0 and len(student_update_sizes) > 0:
            old_models = deepcopy(self.global_models)
            for i, dataset in enumerate(self.global_models.keys()):
                self.global_model = self.global_models[dataset]
                self.task = DATASET_2_TASK[dataset]
                self.modality = DATASET_2_MODALITY[dataset]
                self.dataset = dataset
                self.out_modality_scale = self.args.out_modality_scales[i]
                self._aggregate(student_ids, student_update_sizes, fedavg=True)
                self.global_models[dataset] = self.global_model
            # Keep fedavg-eval aggregation path for parity checks, but avoid
            # evaluating here so round-level evaluation remains centralized in
            # main.py via server.evaluate(...).
            self.global_models = old_models

        if len(student_ids) > 0 and len(student_update_sizes) > 0:
            for i, dataset in enumerate(self.global_models.keys()):
                self.global_model = self.global_models[dataset]
                self.task = DATASET_2_TASK[dataset]
                self.modality = DATASET_2_MODALITY[dataset]
                self.dataset = dataset
                self.out_modality_scale = self.args.out_modality_scales[i]
                self._aggregate(student_ids, student_update_sizes)
                self.global_models[dataset] = self.global_model

        if self.round % self.args.lr_decay_step == 0:
            self.curr_lr *= self.args.lr_decay

        self._empty_client_models()

        self.results[self.round]["fedtoa_blueprint_available"] = blueprint is not None
        return selected_ids
