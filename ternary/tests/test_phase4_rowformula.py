"""Phase 4 tests for the row-formula matmul (reference implementation)."""
import numpy as np
import pytest

from ternary.rowformula import (build_plan, naive_adds, random_ternary, PatchablePlan, Plan)


@pytest.mark.parametrize("shape", [(1, 5), (7, 3), (40, 24), (96, 64), (64, 96)])
@pytest.mark.parametrize("zf", [0.0, 0.5, 0.9])
def test_plan_output_exactly_equals_matmul(shape, zf):
    rng = np.random.default_rng(hash((shape, zf)) % 2**32)
    W = random_ternary(*shape, zf, rng)
    plan = build_plan(W)
    X = rng.integers(-128, 128, size=(17, shape[1]))
    assert np.array_equal(plan.evaluate(X), X @ W.astype(np.int64).T)


def test_counted_ops_match_cost_model_and_naive():
    rng = np.random.default_rng(0)
    W = random_ternary(80, 48, 0.5, rng)
    x = rng.integers(-128, 128, size=48)
    plan = build_plan(W)
    out, ops = plan.evaluate_counting(x)
    assert np.array_equal(np.array(out), W.astype(np.int64) @ x)
    assert ops == plan.adds()
    # a plan with no sharing costs exactly the naive count
    plain = build_plan(W, max_temps=0)
    assert plain.adds() == naive_adds(W) == plain.evaluate_counting(x)[1]
    assert plan.adds() < naive_adds(W)


def test_sharing_is_used_on_obvious_example():
    # rows share x0+x1 and x2-x3
    W = np.array([[1, 1, 1, -1],
                  [1, 1, 1, -1],
                  [1, 1, 0, 0],
                  [-1, -1, -1, 1]])
    plan = build_plan(W)
    assert naive_adds(W) == 3 + 3 + 1 + 3
    # shared: t0 = x0+x1, t1 = x2-x3, t2 = t0+t1; rows 0,1 = t2, row 2 = t0, row 3 = -t2 (1 negation)
    assert plan.adds() == 4
    X = np.arange(12).reshape(3, 4) - 5
    assert np.array_equal(plan.evaluate(X), X @ W.T)


def test_all_zero_and_empty_rows():
    W = np.zeros((4, 6), dtype=np.int8)
    plan = build_plan(W)
    assert plan.adds() == 0 and np.array_equal(plan.evaluate(np.ones((2, 6))), np.zeros((2, 4)))


@pytest.mark.parametrize("flip_frac", [0.001, 0.02, 0.2])
def test_patching_stays_exact(flip_frac):
    rng = np.random.default_rng(1)
    W = random_ternary(120, 64, 0.5, rng)
    pp = PatchablePlan(build_plan(W))
    for _ in range(3):
        W2 = W.copy()
        n = max(1, int(flip_frac * W.size))
        idx = rng.choice(W.size, n, replace=False)
        flat = W2.reshape(-1)
        step = rng.choice([-1, 1], n)
        flat[idx] = np.clip(flat[idx] + step, -1, 1)
        changed = sorted(set((np.nonzero((W2 != W).any(axis=1))[0]).tolist()))
        pp.patch(W2, changed)
        W = W2
        X = rng.integers(-128, 128, size=(9, 64))
        assert np.array_equal(pp.p.evaluate(X), X @ W.astype(np.int64).T)
        x = X[0]
        out, ops = pp.p.evaluate_counting(x)
        assert ops == pp.p.adds()


@pytest.mark.parametrize("N", [1, 5, 64])
def test_fast_kernels_exact(N):
    from ternary.rowformula_fast import FastPlanMatmul, FastNaiveMatmul
    rng = np.random.default_rng(3)
    W = random_ternary(96, 64, 0.5, rng)
    X = rng.integers(-128, 128, size=(N, 64)).astype(np.int32)
    ref = (X.astype(np.int64) @ W.astype(np.int64).T).T
    X_T = np.ascontiguousarray(X.T)
    out = np.zeros((96, N), dtype=np.int32)
    assert np.array_equal(FastNaiveMatmul(W)(X_T, out), ref)
    pp = PatchablePlan(build_plan(W))
    out2 = np.zeros((96, N), dtype=np.int32)
    assert np.array_equal(FastPlanMatmul(pp.p, N)(X_T, out2), ref)
    # also after patching (dead temporaries are dropped when compiling)
    W2 = W.copy(); W2[:10] = 0
    pp.patch(W2, list(range(10)))
    out3 = np.zeros((96, N), dtype=np.int32)
    assert np.array_equal(FastPlanMatmul(pp.p, N)(X_T, out3), (X.astype(np.int64) @ W2.astype(np.int64).T).T)
