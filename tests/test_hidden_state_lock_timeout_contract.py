"""Regression contracts for hidden-state lock timeout handling."""

import argparse
import math
from pathlib import Path
from types import SimpleNamespace

import hs_connectors.transfer as transfer_module
import pytest
import torch

import speculators.train.data as data_module
from hs_connectors import FileBackend, FileTransfer
from speculators.train.data import ArrowDataset


def test_file_transfer_propagates_configured_lock_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller-selected timeout must reach the lock wait operation."""
    generated_path = tmp_path / "generated.safetensors"
    generated_path.touch()
    lock_path = Path(f"{generated_path}.lock")
    lock_path.touch()
    expected_payload = {"hidden_states": torch.zeros(1, 1)}
    observed: dict[str, object] = {}

    def record_lock_wait(
        path: str,
        timeout: float = 10.0,
        poll_interval: float = 0.1,
    ) -> None:
        observed.update(
            path=path,
            timeout=timeout,
            poll_interval=poll_interval,
        )

    monkeypatch.setattr(transfer_module, "wait_for_lock", record_lock_wait)
    monkeypatch.setattr(
        transfer_module,
        "load_file",
        lambda path: expected_payload,
    )

    transfer = FileTransfer(tmp_path, lock_timeout=0.25)

    assert transfer.get_generated(str(generated_path)) is expected_payload
    assert observed["path"] == str(lock_path)
    assert observed["timeout"] == 0.25


def test_file_backend_builds_transfer_with_cli_lock_timeout(tmp_path: Path) -> None:
    """The file backend must carry the explicit CLI timeout into the reader."""
    parser = argparse.ArgumentParser()
    FileBackend.add_train_args(parser)
    args = parser.parse_args(
        [
            "--hidden-states-path",
            str(tmp_path),
            "--hidden-states-lock-timeout",
            "120",
        ]
    )

    transfer = FileBackend.from_train_args(args, data_path="/unused")

    assert transfer.hidden_states_path == tmp_path
    assert transfer.lock_timeout == 120


@pytest.mark.parametrize("lock_timeout", [0.0, -1.0, math.nan, math.inf])
def test_file_transfer_rejects_non_finite_or_non_positive_timeout(
    tmp_path: Path,
    lock_timeout: float,
) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        FileTransfer(tmp_path, lock_timeout=lock_timeout)


def test_file_backend_preserves_default_for_legacy_namespace(tmp_path: Path) -> None:
    args = SimpleNamespace(hidden_states_path=str(tmp_path))

    transfer = FileBackend.from_train_args(args, data_path="/unused")

    assert transfer.lock_timeout == 10.0


def test_arrow_dataset_reraises_hidden_state_lock_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lock timeout must abort the sample instead of becoming ``None``."""
    dataset = object.__new__(ArrowDataset)
    dataset.client = object()
    dataset.data = [{"input_ids": torch.tensor([1, 2, 3])}]
    dataset.model = "test-model"
    dataset.request_timeout = 17.0
    dataset.max_retries = 2

    class TimeoutTransfer:
        def get_generated(self, handle: str) -> dict[str, torch.Tensor]:
            raise TimeoutError(f"hidden-state lock timed out: {handle}")

    dataset.transfer = TimeoutTransfer()
    monkeypatch.setattr(
        data_module,
        "generate_hidden_states",
        lambda *args, **kwargs: "/hidden/generated.safetensors",
    )

    with pytest.raises(TimeoutError, match="hidden-state lock timed out"):
        dataset._maybe_generate_hs(0)
