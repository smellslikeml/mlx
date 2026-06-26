# Copyright © 2023-2024 Apple Inc.

"""Distributed orthogonalization helpers for the :class:`Muon` optimizer.

Muon's per-parameter update applies a Newton-Schulz orthogonalization to every
2D weight matrix. The iteration is matmul-heavy and, in a data-parallel setup,
every rank recomputes the *same* orthogonalization for *every* matrix -- the
work is fully replicated across the group. For models with many large matrices
this dominates the optimizer step.

These helpers shard that redundant work across the group: each weight matrix is
orthogonalized by exactly one rank and the result is shared with the others via
a single ``all_sum``. The compute per rank drops to roughly ``1 / group_size``
of the replicated cost while the communicated update stays the same size it
would be under standard gradient averaging.

Adapted from "DMuon: Efficient Distributed Muon Training with Near-Adam
Overhead" (https://arxiv.org/abs/2606.27153), which distributes the
Newton-Schulz orthogonalization to bring Muon's optimizer-step latency down to
near-AdamW levels.
"""

from typing import Callable, List, Sequence, Tuple

import mlx.core as mx


def newton_schulz_cost(shape: Sequence[int]) -> int:
    """Relative cost of orthogonalizing a matrix of the given 2D ``shape``.

    The Newton-Schulz iteration is dominated by the ``X @ X.T`` and ``B @ X``
    matmuls, so the per-step work scales like ``rows * cols * min(rows, cols)``.
    The constant Newton-Schulz step count is the same for every matrix and is
    dropped -- only the relative ordering matters for load balancing.
    """
    rows, cols = int(shape[-2]), int(shape[-1])
    return rows * cols * min(rows, cols)


def balance_orthogonalization(
    shapes: Sequence[Sequence[int]], world_size: int
) -> List[int]:
    """Assign each matrix to the rank that will orthogonalize it.

    Uses greedy longest-processing-time scheduling: the most expensive matrices
    are placed first, each onto the currently least-loaded rank. This keeps the
    slowest rank's load close to the optimum so no single rank becomes the
    straggler for the whole group.

    Args:
        shapes: The 2D shapes of the matrices that need orthogonalization, in a
            stable order shared by every rank.
        world_size: The number of ranks in the distributed group.

    Returns:
        A list with one entry per matrix giving the rank that owns it. The list
        is identical on every rank because the inputs are identical.
    """
    if world_size <= 1:
        return [0] * len(shapes)

    costs = [(newton_schulz_cost(shape), idx) for idx, shape in enumerate(shapes)]
    # Heaviest first; ties broken by index so the order is deterministic.
    costs.sort(key=lambda c: (-c[0], c[1]))

    loads = [0] * world_size
    owners = [0] * len(shapes)
    for cost, idx in costs:
        rank = min(range(world_size), key=lambda r: (loads[r], r))
        owners[idx] = rank
        loads[rank] += cost
    return owners


def orthogonalize_distributed(
    update: mx.array,
    orthogonalize: Callable[[mx.array], mx.array],
    owner: int,
    group: "mx.distributed.Group",
) -> mx.array:
    """Orthogonalize ``update`` on its owning rank and share the result.

    Only ``owner`` runs the (expensive) ``orthogonalize`` callable; the other
    ranks contribute zeros. A single ``all_sum`` then reconstructs the full
    orthogonalized matrix on every rank, since exactly one rank produced it.

    Args:
        update: The momentum update matrix to orthogonalize.
        orthogonalize: The Newton-Schulz callable (e.g. the optimizer's own
            ``_zeropower_via_newtonschulz5`` bound to ``ns_steps``).
        owner: The rank responsible for this matrix.
        group: The distributed group to reduce over.

    Returns:
        The orthogonalized update, identical on every rank.
    """
    if owner == group.rank():
        result = orthogonalize(update)
    else:
        result = mx.zeros_like(update)
    return mx.distributed.all_sum(result, group=group)


def orthogonalization_shapes(updates: Sequence[mx.array]) -> List[Tuple[int, int]]:
    """Collect the flattened 2D shapes of the ``updates`` Muon will orthogonalize.

    Parameters with fewer than two dimensions are skipped (Muon leaves those to
    an element-wise optimizer); higher-rank tensors are flattened to 2D exactly
    as the Newton-Schulz step does.
    """
    shapes = []
    for u in updates:
        if u.ndim >= 2:
            shapes.append((u.shape[0], int(u.size // u.shape[0])))
    return shapes
