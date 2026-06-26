# Copyright © 2023-2024 Apple Inc.

import unittest

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import mlx_tests
from mlx.utils import tree_flatten
from mlx.optimizers.distributed_muon import (
    balance_orthogonalization,
    newton_schulz_cost,
    orthogonalization_shapes,
    orthogonalize_distributed,
)


class TestDistributedMuon(mlx_tests.MLXTestCase):
    def test_balance_orthogonalization_single_rank(self):
        shapes = [(8, 8), (16, 4), (4, 4)]
        self.assertEqual(balance_orthogonalization(shapes, 1), [0, 0, 0])

    def test_balance_orthogonalization_spreads_load(self):
        # Two equal-cost matrices over two ranks -> one each.
        owners = balance_orthogonalization([(8, 8), (8, 8)], 2)
        self.assertEqual(sorted(owners), [0, 1])

        # The heaviest matrix is placed first; the lighter ones backfill the
        # other rank so the max per-rank load stays balanced.
        shapes = [(64, 64), (8, 8), (8, 8), (8, 8)]
        owners = balance_orthogonalization(shapes, 2)
        costs = [newton_schulz_cost(s) for s in shapes]
        loads = [0, 0]
        for owner, cost in zip(owners, costs):
            loads[owner] += cost
        # The big matrix dominates one rank; the three small ones share the
        # other, and no rank exceeds the heaviest single matrix by much.
        self.assertEqual(max(loads), costs[0])

    def test_orthogonalization_shapes_filters_low_rank(self):
        updates = [mx.zeros((4,)), mx.zeros((8, 6)), mx.zeros((2, 3, 5))]
        self.assertEqual(orthogonalization_shapes(updates), [(8, 6), (2, 15)])

    def test_orthogonalize_distributed_matches_local(self):
        # On the owning rank the helper must reproduce the optimizer's own
        # Newton-Schulz result exactly.
        group = mx.distributed.init()
        m = opt.Muon(learning_rate=0.1)
        x = mx.random.normal((6, 4))
        local = m._zeropower_via_newtonschulz5(x, steps=m.ns_steps)
        shared = orthogonalize_distributed(
            x,
            lambda u: m._zeropower_via_newtonschulz5(u, steps=m.ns_steps),
            group.rank(),
            group,
        )
        self.assertTrue(mx.allclose(local, shared, atol=1e-5))

    def test_muon_distributed_matches_dense(self):
        # The distributed path is wired through the public Muon.update API and
        # must produce the same parameters as the standard Muon, since sharding
        # only changes *which* rank computes each orthogonalization.
        def make_model():
            mx.random.seed(0)
            return nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 4))

        def loss_fn(model, x):
            return model(x).square().mean()

        x = mx.random.normal((5, 8))

        dense_model = make_model()
        dense_opt = opt.Muon(learning_rate=0.1, distributed=False)
        dist_model = make_model()
        dist_opt = opt.Muon(learning_rate=0.1, distributed=True)

        for _ in range(3):
            g1 = nn.value_and_grad(dense_model, loss_fn)(dense_model, x)[1]
            dense_opt.update(dense_model, g1)
            g2 = nn.value_and_grad(dist_model, loss_fn)(dist_model, x)[1]
            dist_opt.update(dist_model, g2)
            mx.eval(dense_model.parameters(), dist_model.parameters())

        # The distributed planner ran and assigned every 2D matrix to a rank.
        self.assertIsNotNone(dist_opt._ortho_owners)
        self.assertEqual(len(dist_opt._ortho_owners), 2)

        dense_flat = dict(tree_flatten(dense_model.parameters()))
        dist_flat = dict(tree_flatten(dist_model.parameters()))
        for key, value in dense_flat.items():
            self.assertTrue(
                mx.allclose(value, dist_flat[key], atol=1e-5),
                msg=f"parameter {key} diverged",
            )


if __name__ == "__main__":
    unittest.main()
