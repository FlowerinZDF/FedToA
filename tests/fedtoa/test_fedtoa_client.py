import pathlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset


sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from methods.fedtoa.payloads import GlobalTopologyBlueprint  # noqa: E402


def _install_client_import_stubs():
    src_stub = types.ModuleType("src")

    class _MetricManager:
        def __init__(self, *_args, **_kwargs):
            self.results = {}

    src_stub.MetricManager = _MetricManager
    src_stub.TqdmToLogger = lambda iterable, **_kwargs: iterable

    criterions_stub = types.ModuleType("src.criterions")
    segloss_stub = types.ModuleType("src.criterions.segmentation_loss")

    class _SegLoss:
        pass

    segloss_stub.SegLoss = _SegLoss

    # Alias fedtoa pure modules under src.methods.* import paths expected by client code.
    import methods.fedtoa.losses as losses
    import methods.fedtoa.payloads as payloads
    import methods.fedtoa.prompt as prompt
    import methods.fedtoa.topology as topology

    methods_stub = types.ModuleType("src.methods")
    fedtoa_stub = types.ModuleType("src.methods.fedtoa")

    sys.modules["src"] = src_stub
    sys.modules["src.criterions"] = criterions_stub
    sys.modules["src.criterions.segmentation_loss"] = segloss_stub
    sys.modules["src.methods"] = methods_stub
    sys.modules["src.methods.fedtoa"] = fedtoa_stub
    sys.modules["src.methods.fedtoa.losses"] = losses
    sys.modules["src.methods.fedtoa.payloads"] = payloads
    sys.modules["src.methods.fedtoa.prompt"] = prompt
    sys.modules["src.methods.fedtoa.topology"] = topology


def _build_args():
    return SimpleNamespace(
        optimizer="SGD",
        lr=0.1,
        momentum=0.0,
        weight_decay=0.0,
        no_shuffle=True,
        B=2,
        distributed=False,
        mm_distributed=False,
        max_grad_norm=0.0,
        debug=False,
        num_classes=3,
        tau=0.5,
        eig_k=2,
        diagonal_eps=1e-4,
        beta_topo=0.3,
        gamma_spec=0.2,
        eta_lip=0.1,
        use_topo=True,
        use_spec=True,
        use_lip=True,
        freeze_backbone=True,
        fedtoa_prompt_only=True,
    )


class TinyPromptModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.prompt = nn.Parameter(torch.zeros(4))
        self.head = nn.Linear(4, 3)

    def forward(self, x, feat_out=False):
        img = x[0]
        feats = self.backbone(img) + self.prompt.unsqueeze(0)
        if feat_out:
            norm = feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            return [feats / norm, None]
        return [self.head(feats), None]




class TinyPromptTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.prompt = nn.Parameter(torch.zeros(4))
        self.head = nn.Linear(4, 3)

    def forward(self, x, feat_out=False):
        txt = x[1]
        feats = self.backbone(txt) + self.prompt.unsqueeze(0)
        if feat_out:
            norm = feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            return [None, feats / norm]
        return [None, self.head(feats)]


class TinyPromptModelWithMissingCheck(TinyPromptModel):
    def __init__(self):
        super().__init__()
        self.last_missing = None

    def forward(self, x, feat_out=False):
        self.last_missing = x[1]
        return super().forward(x, feat_out=feat_out)


class TinyPromptTextModelWithMissingCheck(TinyPromptTextModel):
    def __init__(self):
        super().__init__()
        self.last_missing = None

    def forward(self, x, feat_out=False):
        self.last_missing = x[0]
        return super().forward(x, feat_out=feat_out)


class TinyMMPromptModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_backbone = nn.Linear(4, 4)
        self.txt_backbone = nn.Linear(4, 4)
        self.prompt = nn.Parameter(torch.zeros(4))

    def forward(self, x, feat_out=False):
        img, txt = x
        img_feats = self.img_backbone(img) + self.prompt.unsqueeze(0)
        txt_feats = self.txt_backbone(txt) + self.prompt.unsqueeze(0)
        if feat_out:
            img_norm = img_feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            txt_norm = txt_feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            return [img_feats / img_norm, txt_feats / txt_norm]
        return [img_feats, txt_feats]


def test_local_train_student_updates_prompt_only():
    _install_client_import_stubs()
    from client.fedtoaclient import FedtoaClient

    feats = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    ds = TensorDataset(feats, labels)

    client = FedtoaClient(args=_build_args(), training_set=ds, test_set=ds, modality="img", task="cls", eval_metrics=["acc1"])
    client.id = 7
    client.device = "cpu"
    client.model = TinyPromptModel()

    blueprint = GlobalTopologyBlueprint(
        topology_mean=torch.zeros(3, 3),
        topology_mask=torch.ones(3, 3, dtype=torch.bool),
        spectral_global=torch.zeros(2),
        active_classes=torch.tensor([True, True, False]),
    )
    client.set_global_blueprint(blueprint)

    backbone_before = client.model.backbone.weight.detach().clone()
    prompt_before = client.model.prompt.detach().clone()

    metrics = client.local_train_student(epochs=2)
    epoch_key = 2

    assert epoch_key in metrics
    assert "total_loss" in metrics[epoch_key]
    assert "metrics" in metrics[epoch_key]
    assert torch.allclose(client.model.backbone.weight.detach(), backbone_before)
    assert not torch.allclose(client.model.prompt.detach(), prompt_before)


def test_extract_teacher_topology_returns_class_level_payload():
    _install_client_import_stubs()
    from client.fedtoaclient import FedtoaClient

    feats = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    ds = TensorDataset(feats, labels)

    client = FedtoaClient(args=_build_args(), training_set=ds, test_set=ds, modality="img", task="cls", eval_metrics=["acc1"])
    client.id = 5
    client.device = "cpu"
    client.model = TinyPromptModel()

    payload = client.extract_teacher_topology()

    assert payload.client_id == 5
    assert payload.topology.shape == (3, 3)
    assert payload.spectral.shape == (2,)
    assert payload.support_mask.tolist() == [True, True, False]
    assert payload.class_ids.tolist() == [0, 1]
    assert payload.num_samples == 4


def test_extract_teacher_topology_retrieval_without_class_labels():
    _install_client_import_stubs()
    from client.fedtoaclient import FedtoaClient

    args = _build_args()
    args.num_classes = None
    args.fedtoa_group_count = 4

    img_feats = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.8, 0.2, 0.0]],
        dtype=torch.float32,
    )
    txt_feats = torch.tensor(
        [[0.9, 0.1, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    # labels are placeholders and not used for topology grouping in retrieval path.
    labels = torch.tensor([9, 8, 7, 6], dtype=torch.long)
    group_ids = torch.tensor([10, 10, 17, 17], dtype=torch.long)
    ann_ids = torch.tensor([101, 102, 103, 104], dtype=torch.long)
    indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    ds = TensorDataset(img_feats, txt_feats, labels, group_ids, ann_ids, indices)

    client = FedtoaClient(args=args, training_set=ds, test_set=ds, modality="img+txt", task="img+txt", eval_metrics=[])
    client.id = 11
    client.device = "cpu"
    client.model = TinyMMPromptModel()

    payload = client.extract_teacher_topology()

    assert payload.client_id == 11
    assert payload.topology.shape == (4, 4)
    assert payload.spectral.shape == (2,)
    assert payload.num_samples == 4
    assert payload.class_ids.tolist() == [1, 2]
    assert payload.support_mask.tolist() == [False, True, True, False]


def test_local_train_student_img_retrieval_batch_zero_task_loss_warns_once():
    _install_client_import_stubs()
    from client.fedtoaclient import FedtoaClient

    args = _build_args()
    args.num_classes = None
    args.fedtoa_group_count = 4

    img_feats = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.8, 0.2, 0.0]],
        dtype=torch.float32,
    )
    txt_feats = torch.tensor(
        [[0.9, 0.1, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    labels = torch.tensor([9, 8, 7, 6], dtype=torch.long)
    group_ids = torch.tensor([10, 10, 17, 17], dtype=torch.long)
    ann_ids = torch.tensor([101, 102, 103, 104], dtype=torch.long)
    indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    ds = TensorDataset(img_feats, txt_feats, labels, group_ids, ann_ids, indices)

    client = FedtoaClient(args=args, training_set=ds, test_set=ds, modality="img", task="img+txt", eval_metrics=[])
    client.id = 15
    client.device = "cpu"
    client.model = TinyPromptModelWithMissingCheck()

    blueprint = GlobalTopologyBlueprint(
        topology_mean=torch.zeros(4, 4),
        topology_mask=torch.ones(4, 4, dtype=torch.bool),
        spectral_global=torch.zeros(2),
        active_classes=torch.tensor([True, True, True, True]),
    )
    client.set_global_blueprint(blueprint)

    with pytest.warns(RuntimeWarning, match="task targets unavailable") as caught:
        metrics = client.local_train_student(epochs=2)

    assert len(caught) == 1
    assert metrics[2]["task_loss"] == 0.0
    assert torch.allclose(client.model.last_missing, torch.zeros_like(txt_feats[:2]))


def test_local_train_student_txt_retrieval_batch_uses_text_field():
    _install_client_import_stubs()
    from client.fedtoaclient import FedtoaClient

    args = _build_args()
    args.num_classes = None
    args.fedtoa_group_count = 4

    img_feats = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    txt_feats = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.8, 0.2, 0.0]],
        dtype=torch.float32,
    )
    labels = torch.tensor([9, 8, 7, 6], dtype=torch.long)
    group_ids = torch.tensor([10, 10, 17, 17], dtype=torch.long)
    ann_ids = torch.tensor([101, 102, 103, 104], dtype=torch.long)
    indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    ds = TensorDataset(img_feats, txt_feats, labels, group_ids, ann_ids, indices)

    client = FedtoaClient(args=args, training_set=ds, test_set=ds, modality="txt", task="img+txt", eval_metrics=[])
    client.id = 16
    client.device = "cpu"
    client.model = TinyPromptTextModelWithMissingCheck()

    blueprint = GlobalTopologyBlueprint(
        topology_mean=torch.zeros(4, 4),
        topology_mask=torch.ones(4, 4, dtype=torch.bool),
        spectral_global=torch.zeros(2),
        active_classes=torch.tensor([True, True, True, True]),
    )
    client.set_global_blueprint(blueprint)

    with pytest.warns(RuntimeWarning, match="task targets unavailable"):
        metrics = client.local_train_student(epochs=1)

    assert metrics[1]["task_loss"] == 0.0
    assert "total_loss" in metrics[1]
    assert torch.allclose(client.model.last_missing, torch.zeros_like(img_feats[:2]))
