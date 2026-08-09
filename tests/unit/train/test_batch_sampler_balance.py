"""Balance and partition guarantees for MultipackDistributedBatchSamplerV2."""

import numpy as np
import pytest

from speculators.train.distributed_batch_sampler import (
    MultipackDistributedBatchSamplerV2,
)

MAX_LEN = 8192


def _skewed_lengths(n: int = 4000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.lognormal(mean=7.6, sigma=0.9, size=n).astype(int), 64, MAX_LEN)


def _shards(lengths: np.ndarray, replicas: int) -> list[list[np.ndarray]]:
    return [
        list(
            iter(
                MultipackDistributedBatchSamplerV2(
                    batch_max_length=MAX_LEN,
                    lengths=lengths,
                    num_replicas=replicas,
                    rank=rank,
                )
            )
        )
        for rank in range(replicas)
    ]


@pytest.mark.parametrize("replicas", [2, 3, 4, 8])
def test_sample_counts_are_balanced_across_ranks(replicas):
    lengths = _skewed_lengths()
    counts = [sum(len(b) for b in batches) for batches in _shards(lengths, replicas)]

    assert min(counts) > 0
    assert max(counts) / min(counts) < 1.15, f"sample counts unbalanced: {counts}"


@pytest.mark.parametrize("replicas", [2, 3, 4, 8])
def test_token_counts_stay_balanced_across_ranks(replicas):
    lengths = _skewed_lengths()
    tokens = [
        sum(int(lengths[b].sum()) for b in batches)
        for batches in _shards(lengths, replicas)
    ]

    assert max(tokens) / min(tokens) < 1.15, f"token counts unbalanced: {tokens}"


@pytest.mark.parametrize("replicas", [2, 3, 4, 8])
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_ranks_form_a_disjoint_partition(replicas, seed):
    lengths = _skewed_lengths(seed=seed)
    shards = _shards(lengths, replicas)

    seen: list[set[int]] = []
    for batches in shards:
        idxs = [int(i) for b in batches for i in b]
        assert len(idxs) == len(set(idxs)), "duplicate index within a single rank"
        seen.append(set(idxs))

    for i in range(replicas):
        for j in range(i + 1, replicas):
            assert not (seen[i] & seen[j]), f"ranks {i}/{j} share samples"
    assert set().union(*seen) == set(range(len(lengths)))


@pytest.mark.parametrize("replicas", [2, 3, 4, 8])
def test_equal_batch_count_per_rank(replicas):
    counts = {len(batches) for batches in _shards(_skewed_lengths(), replicas)}
    assert len(counts) == 1, f"ranks disagree on batch count: {counts}"


@pytest.mark.parametrize("replicas", [2, 3])
def test_batches_respect_token_budget(replicas):
    lengths = _skewed_lengths()
    for batches in _shards(lengths, replicas):
        for b in batches:
            assert int(lengths[b].sum()) <= MAX_LEN


def test_rotation_actually_moves_the_largest_sample_off_rank_zero():
    lengths = _skewed_lengths()
    replicas = 3
    shards = _shards(lengths, replicas)
    nbatches = len(shards[0])

    owners = set()
    for i in range(min(nbatches, 30)):
        best_rank, best_len = None, -1
        for rank in range(replicas):
            if len(shards[rank][i]) == 0:
                continue
            m = int(lengths[shards[rank][i]].max())
            if m > best_len:
                best_rank, best_len = rank, m
        owners.add(best_rank)

    assert len(owners) > 1, "largest sample always lands on the same rank"


def test_final_tail_is_rebalanced_without_dropping_samples():
    lengths = np.ones(10, dtype=np.int64)
    replicas = 4
    shards = [
        list(
            iter(
                MultipackDistributedBatchSamplerV2(
                    batch_max_length=2,
                    lengths=lengths,
                    num_replicas=replicas,
                    rank=rank,
                )
            )
        )
        for rank in range(replicas)
    ]

    flattened = [
        int(index)
        for rank_batches in shards
        for batch in rank_batches
        for index in batch
    ]
    assert sorted(flattened) == list(range(len(lengths)))
    assert len(flattened) == len(set(flattened))
    assert {len(rank_batches) for rank_batches in shards} == {2}
    for rank_batches in shards:
        for batch in rank_batches:
            assert len(batch) > 0
            assert int(lengths[batch].sum()) <= 2


def test_unrepresentable_final_tail_fails_instead_of_dropping_samples():
    sampler = MultipackDistributedBatchSamplerV2(
        batch_max_length=1,
        lengths=np.ones(6, dtype=np.int64),
        num_replicas=4,
        rank=0,
    )

    with pytest.raises(ValueError, match="cannot rebalance final sampler tail"):
        list(iter(sampler))


def _small_sampler() -> MultipackDistributedBatchSamplerV2:
    return MultipackDistributedBatchSamplerV2(
        batch_max_length=3,
        lengths=np.ones(18, dtype=np.int64),
        num_replicas=1,
        rank=0,
        seed=17,
    )


def _assert_batches_equal(
    actual: tuple[np.ndarray, ...] | list[np.ndarray],
    expected: tuple[np.ndarray, ...] | list[np.ndarray],
) -> None:
    assert len(actual) == len(expected)
    for actual_batch, expected_batch in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_batch, expected_batch)


def test_remaining_batches_returns_exact_defensive_copies_without_cache_mutation():
    """A suffix mutation must not alter the deterministic full epoch."""
    sampler = _small_sampler()
    full_epoch = tuple(batch.copy() for batch in sampler)

    suffix = sampler.remaining_batches(epoch=0, completed_batches=2)
    _assert_batches_equal(suffix, full_epoch[2:])
    assert all(
        actual is not cached
        for actual, cached in zip(suffix, full_epoch[2:], strict=True)
    )

    suffix[0][0] = -1

    _assert_batches_equal(
        sampler.remaining_batches(epoch=0, completed_batches=2), full_epoch[2:]
    )
    _assert_batches_equal(list(sampler), full_epoch)
    assert len(sampler) == len(full_epoch)


def test_remaining_batches_is_repeatable_after_generating_another_epoch():
    """Evicting the one-epoch cache must not change an earlier epoch's suffix."""
    sampler = _small_sampler()
    first = sampler.remaining_batches(epoch=3, completed_batches=1)

    sampler.remaining_batches(epoch=4, completed_batches=0)

    second = sampler.remaining_batches(epoch=3, completed_batches=1)
    _assert_batches_equal(second, first)


@pytest.mark.parametrize("completed_batches", [-1, 7])
def test_remaining_batches_rejects_invalid_offsets(completed_batches: int):
    """Offsets outside the closed full-epoch boundary are invalid."""
    sampler = _small_sampler()

    with pytest.raises(ValueError, match="completed_batches"):
        sampler.remaining_batches(epoch=0, completed_batches=completed_batches)


def test_remaining_batches_accepts_the_exact_epoch_length():
    """A checkpoint after the final batch has an empty valid suffix."""
    sampler = _small_sampler()

    assert sampler.remaining_batches(epoch=0, completed_batches=len(sampler)) == ()


def test_resume_from_batch_is_one_shot_and_preserves_full_epoch_cache():
    """Only the next matching iterator consumes the suffix request."""
    sampler = _small_sampler()
    full_epoch = tuple(batch.copy() for batch in sampler)

    sampler.resume_from_batch(epoch=0, completed_batches=2)

    _assert_batches_equal(list(sampler), full_epoch[2:])
    _assert_batches_equal(list(sampler), full_epoch)
    _assert_batches_equal(
        sampler.remaining_batches(epoch=0, completed_batches=0), full_epoch
    )
    assert len(sampler) == len(full_epoch)
