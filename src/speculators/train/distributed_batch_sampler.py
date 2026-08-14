"""
MIT License

Copyright (c) 2023 One

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Adapted from https://github.com/imoneoi/multipack_sampler.
"""

# Standard
import warnings
from heapq import heappush, heapreplace
from typing import NamedTuple

import numpy as np

# Third Party
from numpy.typing import ArrayLike, NDArray
from torch.utils.data import Sampler


## Multipack Distributed Batch Sampler
class _Bin(NamedTuple):
    """Helper named tuple for `lpt_packed_batch`"""

    fill: int  # sum of items in _Bin
    slot: int  # heap slot id (0..num_replicas-1)


class _TailDonor(NamedTuple):
    length: int
    distance_from_tail: int
    logical_slot: int
    shuffled_index: int
    rotation: int


class _PackingSearchExhaustedError(RuntimeError):
    pass


def _lpt_packed_batch(
    lengths: np.ndarray,
    max_len: int,
    num_replicas: int,
    start_index: int,
    rank: int,
    rotation: int,
) -> None | list:
    """
    Check if lengths can be distributed into `num_replicas` machines with at most
    `max_len` tokens per machine and return this rank's batch.

    Uses the LPT (Longest processing time first scheduling) algorithm
    Time: O(|lengths| log |lengths| + |lengths| log replicas)

    Bins are packed to a token budget, which leaves sample counts skewed: with every
    bin empty the heap tie-breaks on slot id, so slot 0 always takes the largest sample
    and the highest slot absorbs the small ones. `rotation` re-maps slots to ranks so
    that role cycles across batches.

    Returns:
    `None` if unable to find a valid packing. Otherwise, return the batch indices that
    correspond to `rank`.
    """

    local_batch = []
    heap = [_Bin(0, i) for i in range(num_replicas)]

    target_slot = (rank - rotation) % num_replicas

    indices = np.argsort(lengths)[::-1]
    for idx, size in zip(indices, lengths[indices], strict=True):
        new_fill = heap[0].fill + size
        if new_fill > max_len:
            return None

        if heap[0].slot == target_slot:
            local_batch.append(start_index + int(idx))

        _ = heapreplace(heap, _Bin(new_fill, heap[0].slot))

    return local_batch


def _lpt_packed_slots(
    lengths: np.ndarray,
    max_len: int,
    num_replicas: int,
    start_index: int,
) -> None | list[list[int]]:
    """Pack one global batch and return every logical replica slot."""

    local_batches: list[list[int]] = [[] for _ in range(num_replicas)]
    heap = [_Bin(0, i) for i in range(num_replicas)]

    # sort in descending order
    indices = np.argsort(lengths)[::-1]

    for idx, size in zip(indices, lengths[indices], strict=True):
        new_fill = heap[0].fill + size
        if new_fill > max_len:
            # Size doesn't fit in least full batch (or any others), report failure.
            return None

        local_batches[heap[0].slot].append(start_index + int(idx))

        _ = heapreplace(heap, _Bin(new_fill, heap[0].slot))

    return local_batches


def _exact_packed_groups(
    *,
    lengths: np.ndarray,
    indices: list[int],
    max_len: int,
    num_groups: int,
    node_limit: int = 100_000,
) -> None | list[list[int]]:
    """Exactly pack a small deterministic subset into nonempty token-limited groups."""

    ordered = sorted(indices, key=lambda index: (-int(lengths[index]), index))
    groups: list[list[int]] = [[] for _ in range(num_groups)]
    fills = [0] * num_groups
    visited: set[tuple[int, tuple[tuple[int, bool], ...]]] = set()
    nodes = 0

    def search(position: int) -> bool:
        nonlocal nodes
        nodes += 1
        if nodes > node_limit:
            raise _PackingSearchExhaustedError
        if position == len(ordered):
            return all(groups)

        remaining = len(ordered) - position
        if remaining < sum(not group for group in groups):
            return False

        state = (
            position,
            tuple(
                sorted(
                    (fills[slot], bool(groups[slot]))
                    for slot in range(num_groups)
                )
            ),
        )
        if state in visited:
            return False
        visited.add(state)

        index = ordered[position]
        size = int(lengths[index])
        equivalent_slots: set[tuple[int, bool]] = set()
        for slot in sorted(range(num_groups), key=lambda item: (-fills[item], item)):
            signature = (fills[slot], bool(groups[slot]))
            if signature in equivalent_slots or fills[slot] + size > max_len:
                continue
            equivalent_slots.add(signature)
            groups[slot].append(index)
            fills[slot] += size
            if search(position + 1):
                return True
            fills[slot] -= size
            groups[slot].pop()
        return False

    if not search(0):
        return None
    return groups


def _reconstruct_completed_slots(
    *,
    lengths: np.ndarray,
    max_len: int,
    replicas: int,
    batch_range: tuple[int, int],
) -> list[list[int]]:
    batch_indices = np.arange(*batch_range)
    local_slots = _lpt_packed_slots(
        lengths=lengths[batch_indices],
        max_len=max_len,
        num_replicas=replicas,
        start_index=0,
    )
    if local_slots is None:
        raise ValueError("cannot reconstruct a completed sampler batch")
    slots = [
        [int(batch_indices[position]) for position in slot] for slot in local_slots
    ]
    if any(not slot for slot in slots):
        raise ValueError("completed sampler batch has an empty rank")
    return slots


def _donor_candidates(
    *,
    lengths: np.ndarray,
    slots: list[list[int]],
    distance_from_tail: int,
    rotation: int,
) -> list[_TailDonor]:
    candidates: list[_TailDonor] = []
    for slot_index, slot in enumerate(slots):
        keeper = max(
            slot,
            key=lambda index: (int(lengths[index]), -index),
        )
        candidates.extend(
            _TailDonor(
                length=int(lengths[index]),
                distance_from_tail=distance_from_tail,
                logical_slot=slot_index,
                shuffled_index=index,
                rotation=rotation,
            )
            for index in slot
            if index != keeper
        )
    return candidates


def _retained_rank_batch(
    *,
    slots: list[list[int]],
    donor_indices: frozenset[int],
    rank: int,
    rotation: int,
    replicas: int,
    dtype: np.dtype,
) -> NDArray:
    target_slot = (rank - rotation) % replicas
    retained = [index for index in slots[target_slot] if index not in donor_indices]
    if not retained:
        raise ValueError(
            "cannot rebalance final sampler tail while keeping every rank nonempty"
        )
    return np.asarray(retained, dtype=dtype)


def _retain_smallest_index(
    *,
    smallest: list[tuple[int, int, int]],
    lengths: np.ndarray,
    index: int,
    limit: int,
) -> None:
    entry = (-int(lengths[index]), -index, index)
    if len(smallest) < limit:
        heappush(smallest, entry)
        return
    largest_key = (-smallest[0][0], -smallest[0][1])
    if (int(lengths[index]), index) < largest_key:
        heapreplace(smallest, entry)


def _fold_tail_into_completed_suffix(
    *,
    lengths: np.ndarray,
    max_len: int,
    rank: int,
    replicas: int,
    completed_batch_ranges: list[tuple[int, int]],
    tail_rotation: int,
    dtype: np.dtype,
) -> list[tuple[int, NDArray]]:
    """Fold a tail into the smallest feasible completed suffix.

    A suffix with ``K`` slots and ``K + E`` samples needs only ``E`` multi-sample
    groups. If a packing exists, its grouped samples can be replaced by the ``2E``
    smallest samples without increasing any group sum; every other sample is a
    singleton. This keeps the exact search bounded by the replica count, not corpus
    size.
    """

    smallest_limit = 2 * (replicas - 1)
    smallest: list[tuple[int, int, int]] = []

    tail_start = completed_batch_ranges[-1][1]
    for index in range(tail_start, len(lengths)):
        _retain_smallest_index(
            smallest=smallest,
            lengths=lengths,
            index=index,
            limit=smallest_limit,
        )

    suffix_size = len(lengths) - tail_start
    search_exhausted = False
    for start_rotation in range(len(completed_batch_ranges) - 1, -1, -1):
        suffix_start, suffix_end = completed_batch_ranges[start_rotation]
        suffix_size += suffix_end - suffix_start
        for index in range(suffix_start, suffix_end):
            _retain_smallest_index(
                smallest=smallest,
                lengths=lengths,
                index=index,
                limit=smallest_limit,
            )

        step_count = tail_rotation - start_rotation
        slot_count = step_count * replicas
        excess_count = suffix_size - slot_count
        if excess_count <= 0 or excess_count >= replicas:
            continue

        selected = sorted(
            (entry[2] for entry in smallest),
            key=lambda index: (int(lengths[index]), index),
        )[: 2 * excess_count]
        try:
            packed_groups = _exact_packed_groups(
                lengths=lengths,
                indices=selected,
                max_len=max_len,
                num_groups=excess_count,
            )
        except _PackingSearchExhaustedError:
            search_exhausted = True
            continue
        if packed_groups is None:
            continue

        selected_set = frozenset(selected)
        packed_slots = packed_groups + [
            [index]
            for index in range(suffix_start, len(lengths))
            if index not in selected_set
        ]
        if len(packed_slots) != slot_count:
            raise ValueError("invalid sampler suffix fold slot count")
        packed_slots.sort(
            key=lambda slot: (
                -sum(int(lengths[index]) for index in slot),
                tuple(slot),
            )
        )

        replacements: list[tuple[int, NDArray]] = []
        for step_offset in range(step_count):
            rotation = start_rotation + step_offset
            logical_slot = (rank - rotation) % replicas
            flat_slot = step_offset * replicas + logical_slot
            replacements.append(
                (rotation, np.asarray(packed_slots[flat_slot], dtype=dtype))
            )
        return replacements

    if search_exhausted:
        raise ValueError(
            "cannot rebalance final sampler tail because the bounded exact packing "
            "search was exhausted"
        )
    raise ValueError("cannot rebalance final sampler tail within the token budget")


def _rebalance_final_tail(
    lengths: np.ndarray,
    max_len: int,
    rank: int,
    replicas: int,
    completed_batch_ranges: list[tuple[int, int]],
    tail_indices: NDArray,
    tail_rotation: int,
) -> tuple[list[tuple[int, NDArray]], NDArray | None]:
    if not completed_batch_ranges:
        raise ValueError("cannot rebalance final sampler tail without a previous batch")

    move_count = replicas - len(tail_indices)
    donor_rotations: list[int] = []
    donor_count = 0
    for rotation in range(len(completed_batch_ranges) - 1, -1, -1):
        batch_start, batch_end = completed_batch_ranges[rotation]
        donor_rotations.append(rotation)
        donor_count += batch_end - batch_start - replicas
        if donor_count >= move_count:
            break

    if donor_count < move_count:
        return (
            _fold_tail_into_completed_suffix(
                lengths=lengths,
                max_len=max_len,
                rank=rank,
                replicas=replicas,
                completed_batch_ranges=completed_batch_ranges,
                tail_rotation=tail_rotation,
                dtype=tail_indices.dtype,
            ),
            None,
        )

    donor_candidates: list[_TailDonor] = []
    for rotation in reversed(donor_rotations):
        slots = _reconstruct_completed_slots(
            lengths=lengths,
            max_len=max_len,
            replicas=replicas,
            batch_range=completed_batch_ranges[rotation],
        )
        distance_from_tail = tail_rotation - rotation - 1
        donor_candidates.extend(
            _donor_candidates(
                lengths=lengths,
                slots=slots,
                distance_from_tail=distance_from_tail,
                rotation=rotation,
            )
        )

    donor_indices = tuple(
        candidate.shuffled_index
        for candidate in sorted(donor_candidates)[:move_count]
    )
    donor_set = frozenset(donor_indices)
    rebalanced_tail = np.concatenate(
        (tail_indices, np.asarray(donor_indices, dtype=tail_indices.dtype))
    )
    tail_slots = _lpt_packed_slots(
        lengths[rebalanced_tail],
        max_len,
        replicas,
        0,
    )
    if tail_slots is None:
        raise ValueError("cannot rebalance final sampler tail within the token budget")
    if any(not slot for slot in tail_slots):
        raise ValueError(
            "cannot rebalance final sampler tail while keeping every rank nonempty"
        )

    selected_donors = sorted(donor_candidates)[:move_count]
    replacement_rotations = sorted(
        {candidate.rotation for candidate in selected_donors}
    )
    replacement_batches = [
        (
            rotation,
            _retained_rank_batch(
                slots=_reconstruct_completed_slots(
                    lengths=lengths,
                    max_len=max_len,
                    replicas=replicas,
                    batch_range=completed_batch_ranges[rotation],
                ),
                donor_indices=donor_set,
                rank=rank,
                rotation=rotation,
                replicas=replicas,
                dtype=tail_indices.dtype,
            ),
        )
        for rotation in replacement_rotations
    ]

    tail_target_slot = (rank - tail_rotation) % replicas
    tail_batch = rebalanced_tail[
        np.asarray(tail_slots[tail_target_slot], dtype=np.int64)
    ]
    return replacement_batches, tail_batch


def _assign_to_packed_batches(
    lengths: np.ndarray, max_len: int, rank: int, replicas: int
) -> list[NDArray]:
    """Distribute lengths to batches across all ranks, while respecting max_length.
    Uses a binary search + LPT algorithm.

    Args:
        lengths (np.ndarray): array of dataset sample lengths
        max_len (int): maximum allowed sum of lengths in batch
        rank (int): global rank to collect batches for
        replicas (int): world size to distribute batches to

    Returns:
        A list of index arrays, one per global batch, for this rank. Every
        valid sample is included exactly once across ranks. A final tail that
        cannot be rebalanced without empty ranks or exceeding ``max_len``
        raises ``ValueError`` instead of being silently dropped.
    """

    lengths_so_far: int = 0
    ind: int = 0
    result: list = []
    lengths_cumsum = np.cumsum(lengths)
    completed_batch_ranges: list[tuple[int, int]] = []

    # binary search for max integer x such that the next x elements in shuffled lengths
    # array can be packed into `replicas` batches.
    # Add this rank's batch to `result` and repeat until end of dataset
    while True:
        if len(lengths) - ind < replicas:
            tail_indices = np.arange(ind, len(lengths))
            if len(tail_indices) == 0:
                break
            replacement_batches, tail_batch = _rebalance_final_tail(
                lengths,
                max_len,
                rank,
                replicas,
                completed_batch_ranges,
                tail_indices,
                len(result),
            )
            for rotation, batch in replacement_batches:
                result[rotation] = batch
            if tail_batch is not None:
                result.append(tail_batch)
            break

        # binary search in [1, 1 + upper bound for x)
        batch_start = ind
        left = 1
        right = 1 + int(
            np.searchsorted(
                lengths_cumsum[ind:], lengths_so_far + max_len * replicas, "right"
            )
        )

        # Cycle the slot->rank mapping so no rank is permanently the many-samples one.
        rotation = len(result)

        batch = None
        while right - left > 1 and right > replicas:
            mid = (left + right) // 2
            batch = _lpt_packed_batch(
                lengths[ind : ind + mid], max_len, replicas, ind, rank, rotation
            )
            if batch is None:
                right = mid
            else:
                left = mid

        if batch is None:
            batch = _lpt_packed_batch(
                lengths[ind : ind + left], max_len, replicas, ind, rank, rotation
            )

        ind += left
        lengths_so_far = int(lengths_cumsum[ind - 1])
        completed_batch_ranges.append((batch_start, ind))

        # append only result for this rank (already filtered in lpt_packed_batch)
        result.append(batch)

    return result


class MultipackDistributedBatchSamplerV2(Sampler):
    def __init__(
        self,
        batch_max_length: int,
        lengths: ArrayLike,
        num_replicas: int,
        rank: int,
        truncate_long_samples: bool = True,
        seed: int = 0,
    ):
        """Efficient distributed packing sampler for linear attention style models

        Args:
            batch_max_length (int): max number of tokens in a single batch per device
            lengths (ArrayLike[int]): the lengths of each sample in the dataset
            num_replicas (int): The number of replicas to split the dataset across.
            rank (int): The global rank to collect batches for.
            truncate_long_samples (bool, optional): Whether to truncate long samples
            (True) or drop them (False). Default is True.
            seed (int, optional): Seed for RNG, must be the same on all ranks. Default 0
        """
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.batch_max_length = batch_max_length
        self.lengths = np.array(lengths)

        self.valid_indices = np.nonzero(self.lengths <= self.batch_max_length)[0]
        if len(self.valid_indices) < len(self.lengths):
            if truncate_long_samples:
                msg = (
                    f"Found {len(self.lengths) - len(self.valid_indices)}"
                    f"/{len(self.lengths)} samples longer than batch_max_length. "
                    "These samples will be truncated to batch_max_length."
                )
                self.valid_indices = np.arange(len(self.lengths))
                self.lengths = np.clip(self.lengths, 0, self.batch_max_length)
            else:
                msg = (
                    f"Dropping {len(self.lengths) - len(self.valid_indices)}"
                    f"/{len(self.lengths)} samples longer than batch_max_length. Ensure"
                    " that the right max_batch_length is used during data processing."
                )

            if self.rank == 0:
                warnings.warn(msg, stacklevel=1)

        self._cached_generated_batches: tuple[int, list[NDArray]] = (-1, [])
        self._resume_once: tuple[int, int] | None = None

    def __iter__(self):
        if self._resume_once is not None and self._resume_once[0] == self.epoch:
            epoch, completed_batches = self._resume_once
            self._resume_once = None
            return iter(
                self.remaining_batches(epoch=epoch, completed_batches=completed_batches)
            )
        batches = self._generate_batches(self.epoch)
        return iter(batches)

    def __len__(self):
        batches = self._generate_batches(self.epoch)
        return len(batches)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def remaining_batches(
        self, *, epoch: int, completed_batches: int
    ) -> tuple[NDArray, ...]:
        batches = self._generate_batches(epoch)
        if completed_batches < 0 or completed_batches > len(batches):
            raise ValueError(
                "completed_batches must be between 0 and the number of epoch "
                f"batches ({len(batches)}), got {completed_batches}"
            )
        return tuple(batch.copy() for batch in batches[completed_batches:])

    def resume_from_batch(self, *, epoch: int, completed_batches: int) -> None:
        self.remaining_batches(epoch=epoch, completed_batches=completed_batches)
        self._resume_once = (epoch, completed_batches)

    def _generate_batches(self, epoch: int) -> list[NDArray]:
        """Generate batches for this rank

        Returns:
            list[NDArray]: list of np.arrays containing the indices for each batch on
            this rank
        """
        if self._cached_generated_batches[0] == epoch:
            return self._cached_generated_batches[1]

        rng = np.random.default_rng(seed=self.seed + epoch)
        indices = rng.permutation(self.valid_indices)

        batches = _assign_to_packed_batches(
            self.lengths[indices], self.batch_max_length, self.rank, self.num_replicas
        )

        # The indices in batches are relative to the shuffled self.lengths[indices]
        # Translate them so that they are instead relative to the overall unshuffled
        # self.lengths array.
        batches = [indices[batch] for batch in batches]

        # Cache result
        self._cached_generated_batches = (epoch, batches)
        return batches
