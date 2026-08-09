"""Behavioral coverage for the installed training entrypoint."""

import importlib
import subprocess
import sys


class _RecordingSession:
    def __init__(self) -> None:
        self.ran = False

    def run(self) -> None:
        self.ran = True


def test_main_resolves_argv_and_runs_the_training_session(monkeypatch):
    """The installed entrypoint must resolve CLI arguments before running once."""
    entrypoint = importlib.import_module("speculators.train.entrypoint")
    session = _RecordingSession()
    captured = []

    def build_session(cfg):
        captured.append(cfg.flatten())
        return session

    monkeypatch.setattr(entrypoint, "_build_training_session", build_session)

    assert entrypoint.main(["--verifier-name-or-path", "dummy-model"]) == 0
    assert len(captured) == 1
    assert captured[0]["verifier_name_or_path"] == "dummy-model"
    assert session.ran


def test_module_entrypoint_displays_training_help():
    """Users can invoke training through ``python -m speculators.train``."""
    completed = subprocess.run(
        [sys.executable, "-m", "speculators.train", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--verifier-name-or-path" in completed.stdout


def test_compatibility_script_exports_existing_training_helpers():
    """Existing callers retain access to the helpers moved into the package."""
    compatibility = importlib.import_module("scripts.train")

    for name in (
        "create_transformer_layer_config",
        "_build_from_config_only",
        "build_draft_model",
        "parse_vocab_mappings",
    ):
        assert callable(getattr(compatibility, name))
