"""Compatibility wrapper for the installed training entrypoint."""

import sys

from speculators.train import entrypoint as _entrypoint
from speculators.train.entrypoint import (
    _build_from_config_only,
    build_draft_model,
    create_transformer_layer_config,
    parse_vocab_mappings,
    run_training,
    set_seed,
)

__all__ = [
    "_build_from_config_only",
    "build_draft_model",
    "create_transformer_layer_config",
    "parse_vocab_mappings",
    "run_training",
    "set_seed",
]


if __name__ == "__main__":
    raise SystemExit(_entrypoint.main())

# Keep importers' historical module-level monkeypatch targets working.
sys.modules[__name__] = _entrypoint
