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

        c = topo_mean.shape[0]
        default_topk = max((c * (c - 1)) // 2, 1)
        raw_topk = getattr(self.args, "topk_edges", None)
        topk_edges = default_topk if raw_topk is None else int(raw_topk)
        var_threshold = getattr(self.args, "fedtoa_var_threshold", None)

        confidence_mask = build_confidence_mask(
            topo_mean=topo_mean,
            topo_var=topo_var,
            topk_edges=topk_edges,
            var_threshold=var_threshold,
        )

        return build_global_blueprint(
            topo_mean=topo_mean,
            confidence_mask=confidence_mask,
            spectral_list=spectral_list,
            class_masks=class_masks,
        )

    def _run_student_updates(self, student_ids: List[int], blueprint):
        update_sizes = {}
        update_results = {}

        for client_id in student_ids:
            client = self.clients[client_id]
            resolved_modality = self._bind_resolved_client_modality(client_id)
            self._prepare_client_for_round(client)
            self._align_client_model_layout(client, resolved_modality)
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
        else:
            logger.warning(
                "[FEDTOA] Round %s skipped FedToA orchestration because teachers=%s students=%s.",
                str(self.round).zfill(4),
                len(teacher_ids),
                len(student_ids),
            )
            student_update_sizes = {}

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
            self._central_evaluate(fedavg=True)
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
