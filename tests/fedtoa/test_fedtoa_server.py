import pathlib
import sys
import types
from collections import defaultdict
from types import SimpleNamespace

import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from methods.fedtoa.payloads import TeacherTopologyPayload  # noqa: E402


def _install_server_import_stubs():
    timm_stub = types.ModuleType("timm")
    timm_stub.create_model = lambda *args, **kwargs: None

    wandb_stub = types.ModuleType("wandb")
    wandb_stub.log = lambda *_args, **_kwargs: None

    src_stub = types.ModuleType("src")
    src_stub.init_weights = lambda *_args, **_kwargs: None
    src_stub.TqdmToLogger = lambda iterable, **_kwargs: iterable

    class _MetricManager:
        def __init__(self, *_args, **_kwargs):
            self.results = {}

    src_stub.MetricManager = _MetricManager

    mome_stub = types.ModuleType("src.models.mome")
    metrics_stub = types.ModuleType("src.metrics.eval_coco")

    class _COCOEvaluator:
        pass

    metrics_stub.COCOEvaluator = _COCOEvaluator

    sys.modules.setdefault("timm", timm_stub)
    sys.modules.setdefault("wandb", wandb_stub)
    sys.modules.setdefault("src", src_stub)
    sys.modules.setdefault("src.models", types.ModuleType("src.models"))
    sys.modules.setdefault("src.models.mome", mome_stub)
    sys.modules.setdefault("src.metrics", types.ModuleType("src.metrics"))
    sys.modules.setdefault("src.metrics.eval_coco", metrics_stub)

    # Alias fedtoa pure modules under src.methods.* path expected by server code.
    import methods.fedtoa.server_ops as server_ops

    sys.modules.setdefault("src.methods", types.ModuleType("src.methods"))
    sys.modules.setdefault("src.methods.fedtoa", types.ModuleType("src.methods.fedtoa"))
    sys.modules["src.methods.fedtoa.server_ops"] = server_ops


class FakeFedtoaClient:
    def __init__(self, client_id, modality, payload=None):
        self.id = client_id
        self.modality = modality
        self.args = SimpleNamespace(E=1, lr=0.1)
        self.model = object()
        self.training_set = [0, 1, 2]
        self._payload = payload

        self.extract_calls = 0
        self.blueprint_calls = 0
        self.student_updates = 0

    def download(self, _global_models):
        self.model = object()

    def extract_teacher_topology(self):
        self.extract_calls += 1
        return self._payload

    def set_global_blueprint(self, _blueprint):
        self.blueprint_calls += 1

    def local_train_student(self, epochs):
        self.student_updates += 1
        return {"epochs": epochs, "loss": 0.123}

    def upload(self):
        return {"prompt": torch.zeros(2, dtype=torch.float32)}


def _toy_payload(client_id):
    return TeacherTopologyPayload(
        client_id=client_id,
        class_ids=torch.tensor([0, 1]),
        topology=torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32),
        spectral=torch.tensor([0.2, 0.4], dtype=torch.float32),
        support_mask=torch.tensor([True, True]),
        num_samples=8,
    )


def test_role_split_prefers_configured_teacher_ids():
    _install_server_import_stubs()
    from server.fedtoaserver import FedtoaServer

    server = FedtoaServer.__new__(FedtoaServer)
    server.args = SimpleNamespace(fedtoa_teacher_ids=[3])
    server.clients = [
        FakeFedtoaClient(0, "img+txt"),
        FakeFedtoaClient(1, "img"),
        FakeFedtoaClient(2, "txt"),
        FakeFedtoaClient(3, "img"),
    ]

    selected = [0, 1, 2, 3]
    teachers = server._teacher_client_ids(selected)
    students = server._student_client_ids(selected, teachers)

    assert teachers == [3]
    assert students == [0, 1, 2]


def test_update_smoke_path_one_teacher_one_student():
    _install_server_import_stubs()
    from server.fedtoaserver import FedtoaServer

    server = FedtoaServer.__new__(FedtoaServer)
    server.round = 1
    server.curr_lr = 0.1
    server.args = SimpleNamespace(
        fedtoa_teacher_ids=None,
        topk_edges=1,
        fedtoa_var_threshold=None,
        freeze_modality="none",
        freeze_rounds=0,
        warmup_rounds=0,
        fedavg_eval=False,
        lr_decay_step=100,
        lr_decay=0.5,
    )
    server.results = defaultdict(dict)
    server.global_models = {"Toy": object()}

    teacher = FakeFedtoaClient(0, "img+txt", payload=_toy_payload(0))
    student = FakeFedtoaClient(1, "img")
    server.clients = [teacher, student]

    aggregate_calls = []
    server._sample_clients = lambda: [0, 1]
    server._aggregate = lambda ids, sizes, fedavg=False: aggregate_calls.append((ids, sizes, fedavg))
    server._empty_client_models = lambda: None
    server._log_results = lambda *args, **kwargs: {"ok": True}

    selected = server.update()

    assert selected == [0, 1]
    assert teacher.extract_calls == 1
    assert student.blueprint_calls == 1
    assert student.student_updates == 1

    assert len(aggregate_calls) == 1
    ids, sizes, fedavg = aggregate_calls[0]
    assert ids == [1]
    assert sizes == {1: len(student.training_set)}
    assert fedavg is False

    assert server.latest_blueprint is not None
    assert server.results[1]["fedtoa_blueprint_available"] is True
