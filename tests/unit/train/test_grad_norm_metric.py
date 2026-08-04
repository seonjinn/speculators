import logging
from collections.abc import Iterator
from typing import cast
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from speculators.model import SpeculatorModel
from speculators.train.trainer import Trainer, TrainerConfig


class _OneBatchDataset(Dataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, _index: int) -> dict[str, torch.Tensor]:
        return {"document_ids": torch.ones(1)}


class _ScalarModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))

    def forward(
        self, document_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        del document_ids
        loss = self.weight.square()
        return (
            torch.empty(0),
            loss,
            {
                "loss_sum": loss.detach(),
                "loss_total": torch.tensor(1.0),
            },
        )


class _CpuTrainer(Trainer):
    def setup_model(self) -> None:
        pass

    def setup_optimizer(self) -> None:
        self.optimizers = [torch.optim.SGD(self.model.parameters(), lr=0.1)]
        self.schedulers = []


@pytest.fixture(autouse=True)
def _disable_cuda_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr("speculators.train.trainer.tqdm", lambda loader, **_: loader)


def _make_trainer(tmp_path: str) -> _CpuTrainer:
    model = _ScalarModel()
    config = TrainerConfig(
        lr=0.1,
        num_epochs=1,
        save_path=tmp_path,
        scheduler_type="none",
        hidden_states_dtype=torch.bfloat16,
        log_freq=1,
    )
    loader = DataLoader(_OneBatchDataset(), batch_size=1)
    trainer = _CpuTrainer(cast("SpeculatorModel", model), config, loader)
    trainer.local_rank = torch.device("cpu")  # type: ignore[assignment]
    trainer.device_type = "cpu"
    return trainer


def _logged_train_metrics(caplog: pytest.LogCaptureFixture) -> Iterator[dict]:
    for record in caplog.records:
        if record.name == "speculators.metrics" and isinstance(record.msg, dict):
            yield record.msg["train"]


def test_train_epoch_logs_pre_clip_gradient_norm_and_preserves_clipping(
    tmp_path: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    trainer = _make_trainer(tmp_path)

    with caplog.at_level(logging.INFO, logger="speculators.metrics"):
        trainer.train_epoch(0)

    [metrics] = list(_logged_train_metrics(caplog))
    assert metrics["grad_norm"] == pytest.approx(4.0)
    assert cast("_ScalarModel", trainer.model).weight.item() == pytest.approx(1.9)


def test_distributed_gradient_norm_is_rank_averaged(
    tmp_path: str,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _make_trainer(tmp_path)
    trainer.is_distributed = True
    reduce_index = 0

    def reduce_rank_metrics(
        value: torch.Tensor, dst: int, op: torch.distributed.ReduceOp
    ) -> None:
        nonlocal reduce_index
        del dst, op
        reduce_index += 1
        if reduce_index <= 2:
            value.mul_(4)
        else:
            value.fill_(20.0)

    monkeypatch.setattr(torch.distributed, "reduce", reduce_rank_metrics)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)

    with caplog.at_level(logging.INFO, logger="speculators.metrics"):
        trainer.train_epoch(0)

    [metrics] = list(_logged_train_metrics(caplog))
    assert metrics["grad_norm"] == pytest.approx(5.0)


def test_nonfinite_gradient_norm_stops_before_optimizer_step(
    tmp_path: str,
) -> None:
    trainer = _make_trainer(tmp_path)
    model = cast("_ScalarModel", trainer.model)

    with (
        patch(
            "speculators.train.trainer.torch.nn.utils.clip_grad_norm_",
            return_value=torch.tensor(float("inf")),
        ),
        pytest.raises(FloatingPointError, match="gradient norm is not finite"),
    ):
        trainer.train_epoch(0)

    assert model.weight.item() == pytest.approx(2.0)
