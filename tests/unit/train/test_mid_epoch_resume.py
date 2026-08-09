"""Unit tests for mid-epoch checkpoint save and resume."""

import json
import tempfile
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from speculators.model import SpeculatorModel
from speculators.train import entrypoint as entrypoint_module
from speculators.train import trainer as trainer_module
from speculators.train.checkpointer import (
    DistributedCheckpointer,
    SingleGPUCheckpointer,
)
from speculators.train.distributed_batch_sampler import (
    MultipackDistributedBatchSamplerV2,
)
from speculators.train.trainer import Trainer, TrainerConfig

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def trained_steps() -> list[tuple[int, int, int]]:
    """Per-test collection of (epoch, local_step, global_step) tuples."""
    return []


@pytest.fixture(autouse=True)
def patch_checkpointer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub checkpointer I/O that requires real model weights or process groups."""

    def _save_checkpoint(self: object, *args: object, **kwargs: object) -> None:
        epoch = args[2] if len(args) >= 3 else kwargs.get("epoch", "0")
        self.path.joinpath(str(epoch)).mkdir(parents=True, exist_ok=True)  # type: ignore[attr-defined]

    def _noop(*_args: object, **_kwargs: object) -> None:
        return None

    for cls in (SingleGPUCheckpointer, DistributedCheckpointer):
        monkeypatch.setattr(cls, "save_checkpoint", _save_checkpoint)
        monkeypatch.setattr(cls, "save_scheduler_state_dict", _noop)
        monkeypatch.setattr(cls, "load_model_state_dict", _noop)
        monkeypatch.setattr(cls, "load_optimizer_state_dict", _noop)
        monkeypatch.setattr(cls, "load_scheduler_state_dict", _noop)


class _TinyDataset(Dataset):
    def __len__(self) -> int:
        return 100

    def __getitem__(self, i: int) -> dict:
        return {"input_ids": torch.tensor([i]), "loss_mask": torch.tensor([1.0])}


def _make_loader() -> DataLoader:
    return DataLoader(_TinyDataset(), batch_size=10, shuffle=False)


class _BatchSamplerWithSetEpoch(Protocol):
    def set_epoch(self, epoch: int) -> None: ...


def _dummy_model() -> SpeculatorModel:
    return cast("SpeculatorModel", nn.Identity())


class _MockTrainer(Trainer):
    """Trainer subclass that records steps without GPU/model ops."""

    _trained_steps: list[tuple[int, int, int]]

    def setup_model(self) -> None:
        pass

    def setup_optimizer(self) -> None:
        p = nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([p], lr=1e-4)
        self.opt = opt
        self.optimizers = [opt]
        self.scheduler = None
        self.schedulers = []

    def train_epoch(self, epoch: int) -> None:
        if hasattr(self.train_loader.batch_sampler, "set_epoch"):
            batch_sampler = cast(
                "_BatchSamplerWithSetEpoch", self.train_loader.batch_sampler
            )
            batch_sampler.set_epoch(epoch)

        skip_steps = 0
        if epoch == getattr(self, "current_epoch", epoch):
            skip_steps = getattr(self, "_resume_local_step", 0)
            self._resume_local_step = 0

        num_steps = len(self.train_loader)
        step_interval = (
            max(1, round(num_steps * self.config.checkpoint_freq))
            if self.config.checkpoint_freq < 1
            else None
        )

        for local_step, _batch in enumerate(self.train_loader, 1):
            if local_step <= skip_steps:
                continue
            self._trained_steps.append((epoch, local_step, self.global_step))
            self.global_step += 1
            if (
                step_interval
                and not self.config.save_best
                and local_step % step_interval == 0
                and num_steps - local_step >= step_interval * 0.1
            ):
                self.maybe_save_checkpoint(epoch, local_step=local_step)


def _make_trainer(
    save_path: str,
    trained_steps: list[tuple[int, int, int]],
    resume: bool = False,
    epochs: int = 1,
) -> _MockTrainer:
    cfg = TrainerConfig(
        save_path=save_path,
        num_epochs=epochs,
        lr=1e-4,
        resume_from_checkpoint=resume,
        checkpoint_freq=0.3,
        log_freq=1,
        scheduler_type="none",
    )
    trainer = _MockTrainer(_dummy_model(), cfg, _make_loader())
    trainer._trained_steps = trained_steps
    return trainer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mid_epoch_checkpoint_saves_training_state(
    trained_steps: list[tuple[int, int, int]],
) -> None:
    """training_state.json is written with correct epoch/local_step/global_step."""
    with tempfile.TemporaryDirectory() as tmpdir:
        num_steps = len(_make_loader())
        step_interval = max(1, round(num_steps * 0.3))
        t = _make_trainer(tmpdir, trained_steps=trained_steps)
        for local_step, _batch in enumerate(t.train_loader, 1):
            trained_steps.append((0, local_step, t.global_step))
            t.global_step += 1
            if local_step == step_interval:
                t.maybe_save_checkpoint(0, local_step=local_step)
                break

        state_file = Path(tmpdir) / "0" / "training_state.json"
        assert state_file.exists(), "training_state.json was not saved"
        state = json.loads(state_file.read_text())
        expected = {
            "epoch": 0,
            "local_step": step_interval,
            "global_step": step_interval,
        }
        assert state == expected


def test_mid_epoch_resume_restores_epoch_and_step(
    trained_steps: list[tuple[int, int, int]],
) -> None:
    """Resume from mid-epoch checkpoint stays in same epoch and skips batches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        num_steps = len(_make_loader())
        step_fraction = round(num_steps * 0.3)
        step_interval = max(1, step_fraction)

        # Run 1: interrupt after first checkpoint.
        run1_steps = trained_steps
        t1 = _make_trainer(tmpdir, trained_steps=run1_steps)
        for local_step, _batch in enumerate(t1.train_loader, 1):
            run1_steps.append((0, local_step, t1.global_step))
            t1.global_step += 1
            if local_step == step_interval:
                t1.maybe_save_checkpoint(0, local_step=local_step)
                break

        # Run 2: resume.
        run2_steps: list[tuple[int, int, int]] = []
        t2 = _make_trainer(tmpdir, trained_steps=run2_steps, resume=True)
        assert t2.current_epoch == 0, f"Expected epoch 0, got {t2.current_epoch}"
        assert t2._resume_local_step == step_interval
        assert t2.global_step == step_interval

        t2.train_epoch(t2.current_epoch)

        expected = num_steps - step_interval
        assert len(run2_steps) == expected
        assert run2_steps[0][1] == step_interval + 1  # first local_step after skip
        assert run2_steps[0][2] == step_interval  # global_step continues
        assert run2_steps[-1][1] == num_steps


def test_end_of_epoch_checkpoint_advances_epoch(
    trained_steps: list[tuple[int, int, int]],
) -> None:
    """End-of-epoch checkpoint (local_step=0) resumes at next epoch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        t = _make_trainer(tmpdir, trained_steps=trained_steps, epochs=2)
        t.train_epoch(0)
        t.maybe_save_checkpoint(0, local_step=0)

        run2_steps: list[tuple[int, int, int]] = []
        t2 = _make_trainer(tmpdir, trained_steps=run2_steps, resume=True, epochs=2)
        assert t2.current_epoch == 1, f"Expected epoch 1, got {t2.current_epoch}"
        assert t2._resume_local_step == 0


def test_interrupted_checkpoint_has_no_training_state(
    trained_steps: list[tuple[int, int, int]],
) -> None:
    """'interrupted' checkpoint does not write training_state.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        t = _make_trainer(tmpdir, trained_steps=trained_steps)
        t.maybe_save_checkpoint("interrupted")
        state_file = Path(tmpdir) / "interrupted" / "training_state.json"
        assert not state_file.exists()


def test_symlink_created_and_updated(
    trained_steps: list[tuple[int, int, int]],
) -> None:
    """Symlink is created for mid-epoch and updated when overwritten."""
    with tempfile.TemporaryDirectory() as tmpdir:
        num_steps = len(_make_loader())
        step_interval = max(1, round(num_steps * 0.3))
        t = _make_trainer(tmpdir, trained_steps=trained_steps)
        t.maybe_save_checkpoint(0, local_step=step_interval)
        t.maybe_save_checkpoint(0, local_step=step_interval * 2)

        old_link = Path(tmpdir) / f"epoch0_step{step_interval}"
        new_link = Path(tmpdir) / f"epoch0_step{step_interval * 2}"
        assert not old_link.exists(), "old symlink should be removed"
        assert new_link.is_symlink(), "new symlink should exist"

        state = json.loads((Path(tmpdir) / "0" / "training_state.json").read_text())
        assert state["local_step"] == step_interval * 2


def test_end_of_epoch_symlink(
    trained_steps: list[tuple[int, int, int]],
) -> None:
    """End-of-epoch checkpoint creates epoch{N}_end symlink."""
    with tempfile.TemporaryDirectory() as tmpdir:
        t = _make_trainer(tmpdir, trained_steps=trained_steps)
        t.maybe_save_checkpoint(0, local_step=0)
        end_link = Path(tmpdir) / "epoch0_end"
        assert end_link.is_symlink()


def test_distributed_mid_epoch_checkpoint_rank_gate(
    trained_steps: list[tuple[int, int, int]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rank-0 writes state/symlink while nonzero ranks skip side effects."""
    with tempfile.TemporaryDirectory() as tmpdir:
        num_steps = len(_make_loader())
        step_interval = max(1, round(num_steps * 0.3))

        rank0 = _make_trainer(
            tmpdir,
            trained_steps=trained_steps,
        )
        rank0.is_distributed = True
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
        rank0.maybe_save_checkpoint(0, local_step=step_interval)

        state_rank0 = Path(tmpdir) / "0" / "training_state.json"
        link_rank0 = Path(tmpdir) / f"epoch0_step{step_interval}"
        assert state_rank0.exists()
        assert link_rank0.is_symlink()

    with tempfile.TemporaryDirectory() as tmpdir:
        rank1_steps: list[tuple[int, int, int]] = []
        rank1 = _make_trainer(
            tmpdir,
            trained_steps=rank1_steps,
        )
        rank1.is_distributed = True
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
        rank1.maybe_save_checkpoint(0, local_step=step_interval)

        state_rank1 = Path(tmpdir) / "0" / "training_state.json"
        link_rank1 = Path(tmpdir) / f"epoch0_step{step_interval}"
        assert not state_rank1.exists()
        assert not link_rank1.exists()


class _CountingDataset(Dataset):
    def __init__(self, n_items: int):
        self.n_items = n_items
        self.seen_indices: list[int] = []

    def __len__(self) -> int:
        return self.n_items

    def __getitem__(self, idx: int) -> dict:
        self.seen_indices.append(idx)
        return {
            "input_ids": torch.tensor([idx]),
            "loss_mask": torch.tensor([1.0]),
        }


class _FastSkipBatchSampler:
    def __init__(self, n_items: int):
        self.all_batches = [[i] for i in range(n_items)]
        self._resume_once: tuple[int, int] | None = None
        self.generated_for_epoch: int | None = None
        self.current_epoch = 0

    def __len__(self) -> int:
        return len(self.all_batches)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch

    def remaining_batches(
        self, *, epoch: int, completed_batches: int
    ) -> tuple[list[int], ...]:
        self.generated_for_epoch = epoch
        if completed_batches < 0 or completed_batches > len(self.all_batches):
            raise ValueError("invalid completed_batches")
        return tuple(list(batch) for batch in self.all_batches[completed_batches:])

    def resume_from_batch(self, *, epoch: int, completed_batches: int) -> None:
        self.remaining_batches(epoch=epoch, completed_batches=completed_batches)
        self._resume_once = (epoch, completed_batches)

    def __iter__(self):
        if self._resume_once is not None and self._resume_once[0] == self.current_epoch:
            epoch, completed_batches = self._resume_once
            self._resume_once = None
            yield from self.remaining_batches(
                epoch=epoch, completed_batches=completed_batches
            )
            return
        yield from self.remaining_batches(epoch=self.current_epoch, completed_batches=0)


class _FastSkipMockTrainer(_MockTrainer):
    def train_epoch(self, epoch: int) -> None:
        if hasattr(self.train_loader.batch_sampler, "set_epoch"):
            batch_sampler = cast(
                "_BatchSamplerWithSetEpoch", self.train_loader.batch_sampler
            )
            batch_sampler.set_epoch(epoch)

        skip_steps = self._prepare_resume_skip(epoch)

        for local_step_rel, _batch in enumerate(self.train_loader, 1):
            local_step = local_step_rel + skip_steps
            self._trained_steps.append((epoch, local_step, self.global_step))
            self.global_step += 1


def test_fast_resume_sampler_avoids_skipped_getitem(
    trained_steps: list[tuple[int, int, int]],
) -> None:
    """Fast-skip avoids __getitem__ calls for skipped batches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = _CountingDataset(n_items=10)
        sampler = _FastSkipBatchSampler(n_items=10)
        loader = DataLoader(dataset, batch_sampler=sampler)
        cfg = TrainerConfig(
            save_path=tmpdir,
            num_epochs=1,
            lr=1e-4,
            resume_from_checkpoint=False,
            checkpoint_freq=0.3,
            log_freq=1,
            scheduler_type="none",
        )
        trainer = _FastSkipMockTrainer(_dummy_model(), cfg, loader)
        trainer._trained_steps = trained_steps
        trainer._resume_local_step = 3

        trainer.train_epoch(0)

        assert sampler.generated_for_epoch == 0
        assert sampler.remaining_batches(epoch=0, completed_batches=0) == tuple(
            sampler.all_batches
        )
        assert dataset.seen_indices == list(range(3, 10))


class _IndexedDataset(Dataset):
    def __init__(self, n_items: int) -> None:
        self.n_items = n_items
        self.seen_indices: list[int] = []

    def __len__(self) -> int:
        return self.n_items

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        self.seen_indices.append(index)
        return {
            "input_ids": torch.tensor([float(index + 1)]),
            "document_ids": torch.tensor([0]),
            "loss_mask": torch.tensor([1.0]),
        }


class _TwoHeadModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ordinary = nn.Parameter(torch.tensor([1.0]))
        self.confidence_head = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.confidence_head.weight.fill_(2.0)

    def forward(
        self, input_ids: torch.Tensor, **_kwargs: torch.Tensor
    ) -> tuple[None, torch.Tensor, dict[str, torch.Tensor]]:
        values = input_ids.float()
        prediction = values * self.ordinary + self.confidence_head(values)
        loss = prediction.mean()
        return None, loss, {"loss_sum": loss.detach(), "loss_count": loss.new_tensor(1)}


def _make_real_loop_trainer(
    tmp_path: Path,
    *,
    n_items: int,
    max_steps: int,
    num_epochs: int = 3,
    observer: object | None = None,
) -> tuple[Trainer, MultipackDistributedBatchSamplerV2, _IndexedDataset]:
    dataset = _IndexedDataset(n_items)
    sampler = MultipackDistributedBatchSamplerV2(
        batch_max_length=1,
        lengths=np.ones(n_items, dtype=np.int64),
        num_replicas=1,
        rank=0,
        seed=23,
    )
    loader = DataLoader(dataset, batch_sampler=sampler)
    model = _TwoHeadModel()
    trainer = Trainer.__new__(Trainer)
    trainer.model = cast("SpeculatorModel", model)
    trainer.config = TrainerConfig(
        save_path=str(tmp_path),
        num_epochs=num_epochs,
        lr=0.05,
        resume_from_checkpoint=False,
        checkpoint_freq=1,
        log_freq=1,
        scheduler_type="none",
        hidden_states_dtype=torch.bfloat16,
        max_steps=max_steps,
    )
    trainer.local_rank = "cpu"
    trainer.rank = 0
    trainer.train_loader = loader
    trainer.val_loader = loader
    trainer.is_distributed = False
    trainer.resume_from_checkpoint = False
    trainer.device_type = "cpu"
    trainer.checkpointer = SingleGPUCheckpointer(str(tmp_path))
    trainer.current_epoch = 0
    trainer._resume_local_step = 0
    trainer._resume_global_step = 0
    trainer.global_step = 0
    trainer.best_val_loss = float("inf")
    trainer.optimizers = [torch.optim.SGD(model.parameters(), lr=0.05)]
    trainer.schedulers = []
    trainer.observer = observer
    return trainer, sampler, dataset


def test_resume_iterator_never_reads_skipped_dataset_items(tmp_path: Path) -> None:
    """The sampler suffix must bypass every completed dataset item."""
    trainer, sampler, dataset = _make_real_loop_trainer(
        tmp_path, n_items=12, max_steps=12
    )
    del trainer
    expected_suffix = sampler.remaining_batches(epoch=0, completed_batches=4)
    sampler.resume_from_batch(epoch=0, completed_batches=4)

    list(DataLoader(dataset, batch_sampler=sampler))

    assert dataset.seen_indices == [int(batch[0]) for batch in expected_suffix]
    assert len(sampler) == 12
    assert len(sampler.remaining_batches(epoch=0, completed_batches=0)) == 12


def test_max_steps_writes_exact_mid_epoch_checkpoint_and_exact_next_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Step 100 must be resumable without epoch-end publication or step 101."""
    trainer, sampler, dataset = _make_real_loop_trainer(
        tmp_path, n_items=120, max_steps=100
    )

    def fail_validation(_epoch: int) -> dict[str, float]:
        raise AssertionError("partial epochs must not run validation")

    checkpoint_epochs: list[int | str] = []
    save_checkpoint = trainer.checkpointer.save_checkpoint

    def track_checkpoint(
        model: SpeculatorModel,
        optimizers: list[torch.optim.Optimizer],
        epoch: int | str,
    ) -> None:
        checkpoint_epochs.append(epoch)
        save_checkpoint(model, optimizers, epoch)

    monkeypatch.setattr(trainer.checkpointer, "save_checkpoint", track_checkpoint)
    trainer.val_epoch = fail_validation  # type: ignore[method-assign]
    result = trainer.run_training()

    state = json.loads((tmp_path / "0" / "training_state.json").read_text())
    assert state == {"epoch": 0, "local_step": 100, "global_step": 100}
    assert trainer.global_step == 100
    assert len(dataset.seen_indices) == 100
    assert (tmp_path / "0" / SingleGPUCheckpointer.COMPLETE_MARKER_FILENAME).is_file()
    assert (tmp_path / "epoch0_step100").is_symlink()
    assert not (tmp_path / "epoch0_end").exists()
    assert not (tmp_path / "1").exists()
    assert checkpoint_epochs == [0]
    assert result.checkpoint_epoch == 0
    assert result.local_step == 100
    assert result.global_step == 100

    exact_next = sampler.remaining_batches(epoch=0, completed_batches=100)[0]
    sampler.resume_from_batch(epoch=0, completed_batches=100)
    np.testing.assert_array_equal(next(iter(sampler)), exact_next)


def test_max_steps_already_reached_does_not_execute_an_extra_update(
    tmp_path: Path,
) -> None:
    """A resumed process at its configured limit must not run step 101."""
    trainer, _sampler, dataset = _make_real_loop_trainer(
        tmp_path, n_items=120, max_steps=100
    )
    trainer.global_step = 100
    trainer._resume_local_step = 100

    result = trainer.train_epoch(0)

    assert not result.completed_epoch
    assert result.local_step == 100
    assert trainer.global_step == 100
    assert dataset.seen_indices == []


class _RecordingObserver:
    def __init__(self, model: _TwoHeadModel) -> None:
        self.model = model
        self.weights_during_callback: tuple[float, float] | None = None
        self.evidence: list[object] = []

    def after_backward(self, evidence: object) -> None:
        self.weights_during_callback = (
            float(self.model.ordinary.item()),
            float(self.model.confidence_head.weight.item()),
        )
        self.evidence.append(evidence)


def test_observer_receives_detached_post_clip_evidence_before_optimizer_step(
    tmp_path: Path,
) -> None:
    """Evidence is scalar-only and observes clipped gradients before mutation."""
    trainer, sampler, _dataset = _make_real_loop_trainer(
        tmp_path, n_items=2, max_steps=1
    )
    model = cast("_TwoHeadModel", trainer.model)
    observer = _RecordingObserver(model)
    trainer.observer = observer
    expected_sample = tuple(
        int(index)
        for index in sampler.remaining_batches(epoch=0, completed_batches=0)[0]
    )

    trainer.run_training()

    assert observer.weights_during_callback == (1.0, 2.0)
    assert (
        float(model.ordinary.item()),
        float(model.confidence_head.weight.item()),
    ) != (
        1.0,
        2.0,
    )
    assert len(observer.evidence) == 1
    evidence = observer.evidence[0]
    assert evidence.epoch == 0
    assert evidence.local_step == 1
    assert evidence.global_step == 1
    assert evidence.sample_indices == expected_sample
    assert evidence.loss == pytest.approx(3.0 * (expected_sample[0] + 1))
    assert evidence.ordinary_grad_l2 == pytest.approx(2**-0.5, rel=1e-5)
    assert evidence.confidence_grad_l2 == pytest.approx(2**-0.5, rel=1e-5)
    assert all(not isinstance(value, torch.Tensor) for value in vars(evidence).values())


def test_observer_gradient_norms_reject_non_finite_values() -> None:
    """Evidence must fail closed instead of publishing a misleading finite norm."""
    model = _TwoHeadModel()
    model.ordinary.grad = torch.tensor([float("nan")])
    model.confidence_head.weight.grad = torch.tensor([[1.0]])

    with pytest.raises(ValueError, match="non-finite gradient norm"):
        trainer_module._gradient_l2_norms(
            model,
            is_distributed_run=False,
            fsdp_shard=False,
        )


def test_public_run_training_threads_observer_and_returns_training_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The installed API exposes observer injection and its structured result."""
    observer = object()
    expected = object()
    captured: list[object] = []

    class _Session:
        def run(self) -> object:
            return expected

    def build_session(_cfg: object, *, observer: object | None = None) -> _Session:
        captured.append(observer)
        return _Session()

    monkeypatch.setattr(entrypoint_module, "_build_training_session", build_session)

    assert entrypoint_module.run_training(object(), observer=observer) is expected
    assert captured == [observer]
