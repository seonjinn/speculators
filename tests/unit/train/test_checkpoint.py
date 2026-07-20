import os
import signal
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch.utils.data import DataLoader

from speculators.model import SpeculatorModel
from speculators.train.checkpointer import SingleGPUCheckpointer
from speculators.train.trainer import Trainer, TrainerConfig


def _make_minimal_trainer(tmp_path: Path, checkpoint_freq: int, save_best: bool):
    trainer = Trainer.__new__(Trainer)
    trainer.config = TrainerConfig(
        lr=1e-3,
        num_epochs=0,
        save_path=str(tmp_path),
        resume_from_checkpoint=False,
        checkpoint_freq=checkpoint_freq,
        save_best=save_best,
    )
    trainer.best_val_loss = float("inf")
    trainer.current_epoch = 0
    trainer.global_step = 0
    trainer.is_distributed = False
    trainer.rank = 0
    trainer.local_rank = 0
    trainer.resume_from_checkpoint = False
    trainer.train_loader = cast("DataLoader[Any]", [])
    trainer.val_loader = cast("DataLoader[Any]", [])
    trainer.checkpointer = SingleGPUCheckpointer(str(tmp_path))

    trainer.model = cast("SpeculatorModel", object())
    trainer.optimizers = cast("list[torch.optim.Optimizer]", [object()])
    trainer.schedulers = []
    return trainer


def _mark_complete(tmp_path: Path, epoch: int | str) -> None:
    (tmp_path / str(epoch) / SingleGPUCheckpointer.COMPLETE_MARKER_FILENAME).touch()


def test_previous_epoch_ignores_checkpoint_best(tmp_path: Path):
    (tmp_path / "0").mkdir()
    (tmp_path / "2").mkdir()
    _mark_complete(tmp_path, 0)
    _mark_complete(tmp_path, 2)
    (tmp_path / "checkpoint_best").symlink_to("0", target_is_directory=True)

    cp = SingleGPUCheckpointer(str(tmp_path))
    assert cp.previous_epoch == 2


def test_previous_epoch_skips_unmarked_dir(tmp_path: Path):
    """A crash mid-save leaves an epoch dir without the completeness marker;
    resume must fall back to the newest complete checkpoint."""
    (tmp_path / "0").mkdir()
    _mark_complete(tmp_path, 0)
    (tmp_path / "1").mkdir()
    (tmp_path / "1" / "model.safetensors").write_bytes(b"truncated")  # no marker

    cp = SingleGPUCheckpointer(str(tmp_path))
    assert cp.previous_epoch == 0


def test_mark_checkpoint_complete_enables_resume(tmp_path: Path):
    (tmp_path / "3").mkdir()
    (tmp_path / "3" / "model.safetensors").write_bytes(b"weights")

    cp = SingleGPUCheckpointer(str(tmp_path))
    assert cp.previous_epoch == -1

    cp.mark_checkpoint_complete(3)
    assert (tmp_path / "3" / cp.COMPLETE_MARKER_FILENAME).exists()
    assert SingleGPUCheckpointer(str(tmp_path)).previous_epoch == 3


def test_mark_checkpoint_complete_fsyncs_temp_marker_before_publish(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    epoch_dir = tmp_path / "3"
    epoch_dir.mkdir()
    (epoch_dir / "model.safetensors").write_bytes(b"weights")

    fsynced_inodes: set[int] = set()
    real_fsync = os.fsync
    real_replace = os.replace

    def tracking_fsync(fd: int) -> None:
        fsynced_inodes.add(os.fstat(fd).st_ino)
        real_fsync(fd)

    def checking_replace(source: str | Path, destination: str | Path) -> None:
        assert Path(source).stat().st_ino in fsynced_inodes
        real_replace(source, destination)

    monkeypatch.setattr(os, "fsync", tracking_fsync)
    monkeypatch.setattr(os, "replace", checking_replace)

    cp = SingleGPUCheckpointer(str(tmp_path))
    cp.mark_checkpoint_complete(3)

    assert cp.complete_marker_path(3).exists()
    assert not (epoch_dir / ".checkpoint_complete.tmp").exists()


def test_maybe_save_checkpoint_writes_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    trainer = _make_minimal_trainer(tmp_path, checkpoint_freq=1, save_best=False)

    def fake_cp_save_checkpoint(_model, _opt, epoch):
        (tmp_path / str(epoch)).mkdir(exist_ok=True)

    monkeypatch.setattr(
        trainer.checkpointer, "save_checkpoint", fake_cp_save_checkpoint
    )
    trainer.maybe_save_checkpoint(0, local_step=0)

    marker = tmp_path / "0" / SingleGPUCheckpointer.COMPLETE_MARKER_FILENAME
    assert marker.exists()
    assert SingleGPUCheckpointer(str(tmp_path)).previous_epoch == 0


def test_clear_checkpoint_complete_fsyncs_epoch_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    epoch_dir = tmp_path / "3"
    epoch_dir.mkdir()
    marker = epoch_dir / SingleGPUCheckpointer.COMPLETE_MARKER_FILENAME
    marker.touch()

    fsynced_inodes: set[int] = set()
    real_fsync = os.fsync

    def tracking_fsync(fd: int) -> None:
        fsynced_inodes.add(os.fstat(fd).st_ino)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", tracking_fsync)

    SingleGPUCheckpointer(str(tmp_path)).clear_checkpoint_complete(3)

    assert not marker.exists()
    assert epoch_dir.stat().st_ino in fsynced_inodes


def test_failed_same_epoch_rewrite_cannot_retain_completion_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    trainer = _make_minimal_trainer(tmp_path, checkpoint_freq=1, save_best=False)
    save_calls = 0

    def fake_cp_save_checkpoint(_model, _opt, epoch):
        nonlocal save_calls
        save_calls += 1
        epoch_dir = tmp_path / str(epoch)
        epoch_dir.mkdir(exist_ok=True)
        (epoch_dir / "model.safetensors").write_bytes(b"partial")
        if save_calls == 2:
            raise RuntimeError("injected rewrite failure")

    monkeypatch.setattr(
        trainer.checkpointer, "save_checkpoint", fake_cp_save_checkpoint
    )
    trainer.maybe_save_checkpoint(0, local_step=1)

    with pytest.raises(RuntimeError, match="injected rewrite failure"):
        trainer.maybe_save_checkpoint(0, local_step=2)

    marker = tmp_path / "0" / SingleGPUCheckpointer.COMPLETE_MARKER_FILENAME
    assert not marker.exists()
    assert SingleGPUCheckpointer(str(tmp_path)).previous_epoch == -1


def test_update_best_symlink_creates_and_updates(tmp_path: Path):
    (tmp_path / "1").mkdir()
    (tmp_path / "3").mkdir()
    _mark_complete(tmp_path, 1)
    _mark_complete(tmp_path, 3)

    cp = SingleGPUCheckpointer(str(tmp_path))
    cp.update_best_symlink(1)

    best_path = tmp_path / "checkpoint_best"
    assert best_path.exists()
    assert best_path.is_symlink()
    assert best_path.resolve() == (tmp_path / "1").resolve()

    cp.update_best_symlink(3)
    assert best_path.exists()
    assert best_path.is_symlink()
    assert best_path.resolve() == (tmp_path / "3").resolve()


def test_run_training_updates_checkpoint_best_among_saved_checkpoints_save_best_false(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    trainer = _make_minimal_trainer(tmp_path, checkpoint_freq=2, save_best=False)
    trainer.config = trainer.config._replace(num_epochs=4)

    saved_epochs = []
    val_losses = {
        0: 0.9,
        1: 0.6,
        2: 0.1,
        3: 0.7,
    }

    def fake_train_epoch(epoch: int):
        return None

    def fake_val_epoch(epoch: int):
        return {"loss_epoch": val_losses[epoch]}

    def fake_cp_save_checkpoint(_model, _opt, epoch: int):
        saved_epochs.append(epoch)
        (tmp_path / str(epoch)).mkdir(exist_ok=True)

    trainer.train_epoch = fake_train_epoch
    trainer.val_epoch = fake_val_epoch
    monkeypatch.setattr(
        trainer.checkpointer, "save_checkpoint", fake_cp_save_checkpoint
    )
    monkeypatch.setattr(
        trainer.checkpointer,
        "save_scheduler_state_dict",
        lambda *_args, **_kwargs: None,
    )

    trainer.run_training()

    assert saved_epochs == [0, 1, 3]

    best_path = tmp_path / "checkpoint_best"
    assert best_path.exists()
    assert best_path.is_symlink()
    assert best_path.resolve() == (tmp_path / "1").resolve()
    assert trainer.best_val_loss == 0.6


@pytest.mark.parametrize(
    (
        "save_best",
        "checkpoint_freq",
        "expected_saved",
        "expected_remaining_dirs",
        "expected_best_target",
    ),
    [
        (False, 3, [0, 2], {"0", "2"}, "2"),
        (True, 3, [0, 1, 3], {"3"}, "3"),
    ],
)
def test_save_best_flag_changes_checkpoint_behavior(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    save_best: bool,
    checkpoint_freq: int,
    expected_saved: list[int],
    expected_remaining_dirs: set[str],
    expected_best_target: str,
):
    case_dir = tmp_path / ("save_best" if save_best else "save_freq")
    case_dir.mkdir()
    trainer = _make_minimal_trainer(
        case_dir, checkpoint_freq=checkpoint_freq, save_best=save_best
    )
    trainer.config = trainer.config._replace(num_epochs=4)

    saved_epochs: list[int] = []
    val_losses = {0: 0.9, 1: 0.8, 2: 0.85, 3: 0.7}

    trainer.train_epoch = lambda _epoch: None
    trainer.val_epoch = lambda epoch: {"loss_epoch": val_losses[epoch]}

    def fake_cp_save_checkpoint(_model, _opt, epoch: int):
        saved_epochs.append(epoch)
        (case_dir / str(epoch)).mkdir(exist_ok=True)

    monkeypatch.setattr(
        trainer.checkpointer, "save_checkpoint", fake_cp_save_checkpoint
    )
    monkeypatch.setattr(
        trainer.checkpointer,
        "save_scheduler_state_dict",
        lambda *_args, **_kwargs: None,
    )

    trainer.run_training()

    assert saved_epochs == expected_saved

    remaining_dirs = {
        p.name for p in case_dir.iterdir() if p.is_dir() and p.name.isdigit()
    }
    assert remaining_dirs == expected_remaining_dirs

    best_path = case_dir / "checkpoint_best"
    assert best_path.exists()
    assert best_path.is_symlink()
    assert best_path.resolve() == (case_dir / expected_best_target).resolve()


def test_checkpoint_freq_flag_controls_saves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    trainer = _make_minimal_trainer(tmp_path, checkpoint_freq=3, save_best=False)
    trainer.config = trainer.config._replace(num_epochs=7)

    saved_epochs: list[int] = []
    val_losses = {0: 0.9, 1: 0.1, 2: 0.8, 3: 0.2, 4: 0.7, 5: 0.6, 6: 0.3}

    trainer.train_epoch = lambda _epoch: None
    trainer.val_epoch = lambda epoch: {"loss_epoch": val_losses[epoch]}

    def fake_cp_save_checkpoint(_model, _opt, epoch: int):
        saved_epochs.append(epoch)
        (tmp_path / str(epoch)).mkdir(exist_ok=True)

    monkeypatch.setattr(
        trainer.checkpointer, "save_checkpoint", fake_cp_save_checkpoint
    )
    monkeypatch.setattr(
        trainer.checkpointer,
        "save_scheduler_state_dict",
        lambda *_args, **_kwargs: None,
    )

    trainer.run_training()

    assert saved_epochs == [0, 2, 5]

    best_path = tmp_path / "checkpoint_best"
    assert best_path.exists()
    assert best_path.is_symlink()
    assert best_path.resolve() == (tmp_path / "5").resolve()


def test_checkpoint_best_ignores_incomplete_target(tmp_path: Path):
    (tmp_path / "0").mkdir()
    _mark_complete(tmp_path, 0)
    (tmp_path / "1").mkdir()
    (tmp_path / "1" / "val_metrics.json").write_text('{"loss_epoch": 0.01}')
    (tmp_path / "checkpoint_best").symlink_to("1", target_is_directory=True)

    cp = SingleGPUCheckpointer(str(tmp_path))
    assert cp.previous_epoch == 0
    assert cp.read_best_epoch() is None
    assert cp.load_best_val_loss() is None


def test_checkpoint_best_rejects_target_outside_checkpoint_root(tmp_path: Path):
    (tmp_path / "0").mkdir()
    _mark_complete(tmp_path, 0)
    external = tmp_path.parent / f"{tmp_path.name}-external"
    external.mkdir()
    _mark_complete(external.parent, external.name)
    (tmp_path / "checkpoint_best").symlink_to(external, target_is_directory=True)

    cp = SingleGPUCheckpointer(str(tmp_path))
    assert cp.read_best_epoch() is None
    assert cp.load_best_val_loss() is None


def test_checkpoint_best_ignores_symlink_loop(tmp_path: Path):
    (tmp_path / "checkpoint_best").symlink_to("checkpoint_best")

    cp = SingleGPUCheckpointer(str(tmp_path))
    assert cp.read_best_epoch() is None
    assert cp.load_best_val_loss() is None


def test_cleanup_rejects_incomplete_best_target(tmp_path: Path):
    (tmp_path / "0").mkdir()
    _mark_complete(tmp_path, 0)
    (tmp_path / "1").mkdir()
    (tmp_path / "checkpoint_best").symlink_to("1", target_is_directory=True)

    cp = SingleGPUCheckpointer(str(tmp_path))
    with pytest.raises(ValueError, match="requested committed epoch"):
        cp.cleanup_keep_only_best(1)

    assert (tmp_path / "0").is_dir()
    assert (tmp_path / "1").is_dir()


def test_update_best_symlink_requires_complete_epoch(tmp_path: Path):
    (tmp_path / "1").mkdir()
    cp = SingleGPUCheckpointer(str(tmp_path))

    with pytest.raises(FileNotFoundError, match="incomplete epoch 1"):
        cp.update_best_symlink(1)


def test_maybe_update_best_publishes_marker_before_best_symlink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    trainer = _make_minimal_trainer(tmp_path, checkpoint_freq=1, save_best=False)
    calls: list[str] = []
    original_mark = trainer.checkpointer.mark_checkpoint_complete
    original_update = trainer.checkpointer.update_best_symlink

    def mark(epoch: int) -> None:
        calls.append("mark")
        original_mark(epoch)

    def update(epoch: int) -> None:
        calls.append("best")
        assert trainer.checkpointer.complete_marker_path(epoch).is_file()
        original_update(epoch)

    monkeypatch.setattr(trainer.checkpointer, "mark_checkpoint_complete", mark)
    monkeypatch.setattr(trainer.checkpointer, "update_best_symlink", update)

    trainer.maybe_update_best(0, {"loss_epoch": 0.1})

    assert calls == ["mark", "best"]


def test_save_and_load_val_metrics(tmp_path: Path):
    cp = SingleGPUCheckpointer(str(tmp_path))

    # No file yet
    assert cp.load_best_val_loss() is None

    # Save val_metrics for epoch 0 and point checkpoint_best at it
    (tmp_path / "0").mkdir()
    cp.save_val_metrics(0, {"loss_epoch": 0.123456, "full_acc_0_epoch": 0.5})
    _mark_complete(tmp_path, 0)
    cp.update_best_symlink(0)
    assert cp.load_best_val_loss() == pytest.approx(0.123456)

    # Save better metrics for epoch 1 and update best
    (tmp_path / "1").mkdir()
    cp.save_val_metrics(1, {"loss_epoch": 0.05, "full_acc_0_epoch": 0.7})
    _mark_complete(tmp_path, 1)
    cp.update_best_symlink(1)
    assert cp.load_best_val_loss() == pytest.approx(0.05)


def test_best_val_loss_restored_on_resume(tmp_path: Path):
    (tmp_path / "4").mkdir()

    cp = SingleGPUCheckpointer(str(tmp_path))
    cp.save_val_metrics(4, {"loss_epoch": 0.42, "full_acc_0_epoch": 0.6})
    _mark_complete(tmp_path, 4)
    cp.update_best_symlink(4)

    trainer = Trainer.__new__(Trainer)
    trainer.resume_from_checkpoint = True
    trainer.is_distributed = False
    trainer.rank = 0
    trainer.local_rank = 0
    trainer.checkpointer = cp

    trainer.current_epoch = cp.previous_epoch + 1
    trainer.global_step = 0
    trainer.best_val_loss = float("inf")

    # Simulate the load that setup_trainer does
    saved = trainer.checkpointer.load_best_val_loss()
    if saved is not None:
        trainer.best_val_loss = saved

    assert trainer.best_val_loss == pytest.approx(0.42)


def test_graceful_shutdown_saves_interrupted_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    trainer = _make_minimal_trainer(tmp_path, checkpoint_freq=99, save_best=False)
    trainer.config = trainer.config._replace(num_epochs=4)

    saved_labels: list[int | str] = []

    def fake_train_epoch(epoch: int):
        if epoch == 2:
            os.kill(os.getpid(), signal.SIGINT)

    def fake_val_epoch(epoch: int):
        return {"loss_epoch": 0.5}

    def fake_cp_save_checkpoint(_model, _opt, epoch: int | str):
        saved_labels.append(epoch)
        (tmp_path / str(epoch)).mkdir(exist_ok=True)

    trainer.train_epoch = fake_train_epoch
    trainer.val_epoch = fake_val_epoch
    monkeypatch.setattr(
        trainer.checkpointer, "save_checkpoint", fake_cp_save_checkpoint
    )
    monkeypatch.setattr(
        trainer.checkpointer,
        "save_scheduler_state_dict",
        lambda *_args, **_kwargs: None,
    )

    trainer.run_training()

    assert "interrupted" in saved_labels


class _TinyOptimModel(torch.nn.Module):
    """Duck-typed stand-in for a PreTrainedModel: save_pretrained + dtype."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def save_pretrained(self, save_directory, state_dict=None, **kwargs):
        Path(save_directory).mkdir(parents=True, exist_ok=True)


def _stepped_optimizer(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.linear.weight.grad = torch.randn_like(model.linear.weight)
    model.linear.bias.grad = torch.randn_like(model.linear.bias)
    optimizer.step()
    # Simulate a long run: bf16 would round 10001 to a multiple of 64.
    for state in optimizer.state.values():
        if isinstance(state.get("step"), torch.Tensor):
            state["step"] = state["step"] + 10000.0
    return optimizer


def test_optimizer_state_round_trips_at_full_precision(tmp_path):
    """Optimizer state must survive save + load without dtype degradation:
    bf16 casts quantize the Adam moments and round the step counters (bf16 has
    an 8-bit mantissa, so step counts above 256 are no longer exact)."""
    model = _TinyOptimModel()
    optimizer = _stepped_optimizer(model)
    reference = optimizer.state_dict()

    checkpointer = SingleGPUCheckpointer(tmp_path)
    checkpointer.save_checkpoint(model, optimizer, epoch=0)

    saved = torch.load(
        checkpointer.optimizer_path(0), weights_only=True, map_location="cpu"
    )
    for param_id, state in reference["state"].items():
        saved_state = saved["state"][param_id]
        assert torch.equal(saved_state["step"], state["step"])
        for key in ("exp_avg", "exp_avg_sq"):
            assert saved_state[key].dtype == torch.float32
            assert torch.equal(saved_state[key], state[key])

    # Load path: a fresh checkpointer resumes from the saved epoch and must
    # restore the state exactly, without casting to the model dtype.
    resumed_model = _TinyOptimModel()
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    resumed_checkpointer = SingleGPUCheckpointer(tmp_path)
    resumed_checkpointer.load_optimizer_state_dict(resumed_model, resumed_optimizer)
    restored = resumed_optimizer.state_dict()
    for param_id, state in reference["state"].items():
        restored_state = restored["state"][param_id]
        # load_optimizer_state_dict maps onto the current accelerator, so on a GPU
        # host the restored tensors live on cuda while the reference is on cpu.
        # Compare on cpu: this test is about dtype and value, not placement.
        assert torch.equal(restored_state["step"].cpu(), state["step"].cpu())
        for key in ("exp_avg", "exp_avg_sq"):
            assert restored_state[key].dtype == state[key].dtype
            assert torch.equal(restored_state[key].cpu(), state[key].cpu())


def test_legacy_bf16_optimizer_step_is_restored_to_float32(tmp_path):
    """A checkpoint written by the previous bf16 save path stores "step" in
    bf16. The non-capturable loader keeps that dtype, so in-place increments
    would round once the count is large. The load path must repair it."""
    model = _TinyOptimModel()
    optimizer = _stepped_optimizer(model)

    checkpointer = SingleGPUCheckpointer(tmp_path)
    checkpointer.save_checkpoint(model, optimizer, epoch=0)

    # Rewrite the saved file the way the old bf16-casting save path would have.
    legacy = torch.load(
        checkpointer.optimizer_path(0), weights_only=True, map_location="cpu"
    )
    for state in legacy["state"].values():
        state["step"] = state["step"].to(torch.bfloat16)
    torch.save(legacy, checkpointer.optimizer_path(0))

    resumed_model = _TinyOptimModel()
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    SingleGPUCheckpointer(tmp_path).load_optimizer_state_dict(
        resumed_model, resumed_optimizer
    )

    for state in resumed_optimizer.state.values():
        assert state["step"].dtype == torch.float32
