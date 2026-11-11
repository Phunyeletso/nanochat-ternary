"""
"Group by value" integer matmul — the generalised version of the row-minimum / column-total (T) idea.

For one output row r of W (integers, e.g. int4 or int8) and an input vector x:
    y_r = sum_j W[r,j] * x_j
        = sum over distinct values v in row r of  v * S_{r,v},   S_{r,v} = sum of x_j where W[r,j] == v
So each output needs one multiply per DISTINCT non-zero value in its row (values +1/-1 need none: just add or
subtract), instead of one multiply per weight. Additions are unchanged: every non-zero weight still adds its x_j
into some group sum, then the group results are combined.

Peeling the row minimum (your T trick) is the first step of this: v = min, S = whole column total. Repeating the
peel for every level of the row ends at exactly this grouping.

Cost model (same for naive and grouped):
    naive   : multiplies = weights with |w| >= 2 ;  additions = (non-zeros - 1) per row
    grouped : multiplies = distinct values with |v| >= 2 per row ; additions = (non-zeros - 1) per row
"""
import numpy as np
import numba as nb


# ----------------------------------------------------------------------------- weights
def quantized_weights(n_out, n_in, bits, rng):
    """Realistic quantised weights: Gaussian, symmetric absmax quantisation to `bits` bits."""
    w = rng.standard_normal((n_out, n_in))
    qmax = 2 ** (bits - 1) - 1
    scale = np.abs(w).max(axis=1, keepdims=True) / qmax          # per-row absmax, like int8/int4 LLM quantisers
    return np.clip(np.round(w / scale), -qmax - 1, qmax).astype(np.int32)


# ----------------------------------------------------------------------------- counting
def naive_counts(W):
    W = np.asarray(W)
    nnz = (W != 0).sum(axis=1)
    return int((np.abs(W) >= 2).sum()), int(np.maximum(nnz - 1, 0).sum())


def grouped_counts(W):
    W = np.asarray(W)
    mul = 0
    for r in range(W.shape[0]):
        vals = np.unique(W[r])
        mul += int((np.abs(vals) >= 2).sum())
    nnz = (W != 0).sum(axis=1)
    return mul, int(np.maximum(nnz - 1, 0).sum())


# ----------------------------------------------------------------------------- reference (plain NumPy)
def grouped_matmul_reference(W, X):
    """X: (N, n_in) ints -> (N, n_out) int64, computed group by group."""
    W = np.asarray(W); X = np.asarray(X, dtype=np.int64)
    out = np.zeros((X.shape[0], W.shape[0]), dtype=np.int64)
    for r in range(W.shape[0]):
        for v in np.unique(W[r]):
            if v == 0:
                continue
            S = X[:, W[r] == v].sum(axis=1)                       # additions only
            out[:, r] += S if v == 1 else (-S if v == -1 else v * S)
    return out


# ----------------------------------------------------------------------------- compiled kernels
def compile_grouped(W):
    """CSR-like layout: for each row, its groups; for each group, its value and input indices."""
    W = np.asarray(W)
    row_ptr, grp_val, grp_ptr, idx = [0], [], [0], []
    for r in range(W.shape[0]):
        for v in np.unique(W[r]):
            if v == 0:
                continue
            js = np.nonzero(W[r] == v)[0]
            idx.extend(js.tolist()); grp_val.append(int(v)); grp_ptr.append(len(idx))
        row_ptr.append(len(grp_val))
    a = lambda z: np.asarray(z, dtype=np.int32)
    return a(row_ptr), a(grp_val), a(grp_ptr), a(idx)


def compile_naive_int(W):
    W = np.asarray(W)
    ptr, idx, val = [0], [], []
    for r in range(W.shape[0]):
        js = np.nonzero(W[r])[0]
        idx.extend(js.tolist()); val.extend(W[r, js].tolist()); ptr.append(len(idx))
    a = lambda z: np.asarray(z, dtype=np.int32)
    return a(ptr), a(idx), a(val)


@nb.njit(cache=True)
def grouped_kernel(X_T, row_ptr, grp_val, grp_ptr, idx, S, out):
    """X_T: (n_in, N) int32; out: (n_out, N) int32. Inner loops run over the N tokens (SIMD-friendly)."""
    N = X_T.shape[1]
    for r in range(row_ptr.shape[0] - 1):
        o = out[r]
        for t in range(N):
            o[t] = 0
        for g in range(row_ptr[r], row_ptr[r + 1]):
            for t in range(N):
                S[t] = 0
            for p in range(grp_ptr[g], grp_ptr[g + 1]):
                x = X_T[idx[p]]
                for t in range(N):
                    S[t] += x[t]
            v = grp_val[g]
            if v == 1:
                for t in range(N):
                    o[t] += S[t]
            elif v == -1:
                for t in range(N):
                    o[t] -= S[t]
            else:
                for t in range(N):
                    o[t] += v * S[t]


@nb.njit(cache=True)
def naive_int_kernel(X_T, ptr, idx, val, out):
    N = X_T.shape[1]
    for r in range(ptr.shape[0] - 1):
        o = out[r]
        for t in range(N):
            o[t] = 0
        for p in range(ptr[r], ptr[r + 1]):
            x = X_T[idx[p]]; w = val[p]
            for t in range(N):
                o[t] += w * x[t]


class GroupedMatmul:
    def __init__(self, W, max_tokens):
        self.c = compile_grouped(W)
        self.S = np.zeros(max_tokens, dtype=np.int32)

    def __call__(self, X_T, out):
        grouped_kernel(X_T, *self.c, self.S[:X_T.shape[1]], out)
        return out


class NaiveIntMatmul:
    def __init__(self, W):
        self.c = compile_naive_int(W)

    def __call__(self, X_T, out):
        naive_int_kernel(X_T, *self.c, out)
        return out
