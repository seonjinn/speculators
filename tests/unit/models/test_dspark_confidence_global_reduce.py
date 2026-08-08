"""Tests for global confidence-loss normalization in DSpark."""

import math
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.functional import binary_cross_entropy_with_logits

from speculators.models.dspark.metrics import _masked_decayed_mean


def _rank_features(rank: int) -> torch.Tensor:
    values = [1.0, 3.0] if rank == 0 else [2.0, 4.0, 6.0, 8.0]
    return torch.tensor([values])


def _worker(rank: int, world_size: int, init_file: str, ret: dict) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    theta = torch.nn.Parameter(torch.zeros(()))
    features = _rank_features(rank)
    logits = theta * features
    targets = torch.zeros_like(logits)
    elementwise = binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss_mask = torch.ones_like(elementwise)
    pos_idx = torch.zeros_like(elementwise)
    _masked_decayed_mean(elementwise, loss_mask, pos_idx, decay_fn=None).backward()

    assert theta.grad is not None
    grad = theta.grad.detach().clone()
    dist.all_reduce(grad, op=dist.ReduceOp.SUM)
    grad /= world_size
    if rank == 0:
        ret["averaged_grad"] = grad.item()
    dist.destroy_process_group()


def test_single_process_confidence_mean_is_unchanged() -> None:
    logits = torch.zeros(1, 4)
    targets = torch.zeros_like(logits)
    elementwise = binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss_mask = torch.ones_like(elementwise)
    pos_idx = torch.zeros_like(elementwise)

    got = _masked_decayed_mean(elementwise, loss_mask, pos_idx, decay_fn=None)

    assert abs(got.item() - math.log(2.0)) < 1e-6


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="requires torch.distributed with the gloo backend",
)
def test_ddp_averaged_confidence_grad_equals_global_token_weighted_objective() -> None:
    world_size = 2
    with tempfile.TemporaryDirectory() as tmp:
        init_file = f"{tmp}/pg_init"
        ctx = mp.get_context("spawn")
        with ctx.Manager() as manager:
            ret = manager.dict()
            mp.spawn(
                _worker,
                args=(world_size, init_file, ret),
                nprocs=world_size,
                join=True,
            )
            assert "averaged_grad" in ret, "rank-0 worker did not report a result"
            averaged_grad = ret["averaged_grad"]
    assert abs(averaged_grad - 2.0) < 1e-6
