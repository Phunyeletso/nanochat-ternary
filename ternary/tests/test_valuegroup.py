"""Tests for the value-grouped integer matmul."""
import numpy as np
import pytest

from ternary.valuegroup import (quantized_weights, naive_counts, grouped_counts, grouped_matmul_reference,
                                GroupedMatmul, NaiveIntMatmul)


def test_users_3x3_example():
    A = np.array([[3, 2, 4], [6, 1, 2], [4, 5, 6]])
    B = np.array([[3, 4, 1], [2, 2, 3], [4, 5, 2]])
    # grouped_matmul_reference computes X @ W.T, so W = A and X = B.T gives (A @ B).T
    assert np.array_equal(grouped_matmul_reference(A, B.T).T, A @ B)
    assert np.array_equal(A @ B, [[29, 36, 17], [28, 36, 13], [46, 56, 31]])


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("N", [1, 7, 64])
def test_reference_and_kernels_exact(bits, N):
    rng = np.random.default_rng(bits * 100 + N)
    W = quantized_weights(48, 80, bits, rng)
    X = rng.integers(-128, 128, size=(N, 80)).astype(np.int32)
    ref = X.astype(np.int64) @ W.astype(np.int64).T
    assert np.array_equal(grouped_matmul_reference(W, X), ref)
    X_T = np.ascontiguousarray(X.T)
    o = np.zeros((48, N), dtype=np.int32)
    assert np.array_equal(GroupedMatmul(W, N)(X_T, o).T, ref)
    o2 = np.zeros((48, N), dtype=np.int32)
    assert np.array_equal(NaiveIntMatmul(W)(X_T, o2).T, ref)


def test_counts():
    W = np.array([[3, 3, 3, 0], [1, -1, 2, 2]])
    assert naive_counts(W) == (3 + 2, 2 + 3)
    assert grouped_counts(W) == (1 + 1, 2 + 3)          # one multiply per distinct |v|>=2 value
    rng = np.random.default_rng(0)
    W4 = quantized_weights(64, 256, 4, rng)
    assert grouped_counts(W4)[0] < naive_counts(W4)[0] / 10   # int4: at most 14 multiplies per row
