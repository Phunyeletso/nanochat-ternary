"""
Phase 4: fast (compiled) versions of the row-formula matmul, for timing against plain matmul.

Both compiled kernels use the same layout and loop style so the comparison is fair:
  * inputs are given transposed, X_T: (n_in, N) int32, so the inner loop runs over the N tokens
    contiguously (lets the compiler use SIMD);
  * naive_kernel:   out[r] = sum over nonzeros of +/- x[j]           (add/sub only, no sharing)
  * plan_kernel:    first every shared temporary t = sym[a] +/- sym[b], then out[r] = sum +/- sym[s]
Single-threaded (no parallel loops), compiled with Numba.
"""
import numpy as np
import numba as nb


def compile_plan(plan):
    """Turn a Plan into flat arrays. Dead temporaries are dropped and ids renumbered."""
    n_in = plan.n_in
    remap = {i: i for i in range(n_in)}
    ta, tb, trel = [], [], []
    for i, (a, b, rel) in enumerate(plan.temps):
        if not plan.alive[i]:
            continue
        remap[n_in + i] = n_in + len(ta)
        ta.append(remap[a]); tb.append(remap[b]); trel.append(rel)
    ptr, idx, sgn = [0], [], []
    for row in plan.rows:
        items = sorted(row.items(), key=lambda kv: -kv[1])
        for s, g in items:
            idx.append(remap[s]); sgn.append(g)
        ptr.append(len(idx))
    as_i32 = lambda v: np.asarray(v, dtype=np.int32)
    return dict(n_in=n_in, n_sym=n_in + len(ta), ta=as_i32(ta), tb=as_i32(tb), trel=as_i32(trel),
                ptr=as_i32(ptr), idx=as_i32(idx), sgn=as_i32(sgn))


def compile_naive(W):
    W = np.asarray(W)
    ptr, idx, sgn = [0], [], []
    for r in range(W.shape[0]):
        nz = np.nonzero(W[r])[0]
        order = sorted(nz.tolist(), key=lambda j: -W[r, j])
        idx += order; sgn += [int(W[r, j]) for j in order]
        ptr.append(len(idx))
    as_i32 = lambda v: np.asarray(v, dtype=np.int32)
    return dict(ptr=as_i32(ptr), idx=as_i32(idx), sgn=as_i32(sgn))


@nb.njit(cache=True, fastmath=False)
def _rows(src, ptr, idx, sgn, out):
    N = src.shape[1]
    for r in range(ptr.shape[0] - 1):
        o = out[r]
        for t in range(N):
            o[t] = 0
        for p in range(ptr[r], ptr[r + 1]):
            s = src[idx[p]]
            if sgn[p] > 0:
                for t in range(N):
                    o[t] += s[t]
            else:
                for t in range(N):
                    o[t] -= s[t]


@nb.njit(cache=True)
def naive_kernel(X_T, ptr, idx, sgn, out):
    _rows(X_T, ptr, idx, sgn, out)


@nb.njit(cache=True)
def plan_kernel(X_T, n_in, ta, tb, trel, ptr, idx, sgn, syms, out):
    N = X_T.shape[1]
    for j in range(n_in):
        sj = syms[j]; xj = X_T[j]
        for t in range(N):
            sj[t] = xj[t]
    for i in range(ta.shape[0]):
        d = syms[n_in + i]; a = syms[ta[i]]; b = syms[tb[i]]
        if trel[i] > 0:
            for t in range(N):
                d[t] = a[t] + b[t]
        else:
            for t in range(N):
                d[t] = a[t] - b[t]
    _rows(syms, ptr, idx, sgn, out)


class FastPlanMatmul:
    def __init__(self, plan, max_tokens):
        self.c = compile_plan(plan)
        self.syms = np.zeros((self.c["n_sym"], max_tokens), dtype=np.int32)

    def __call__(self, X_T, out):
        c = self.c
        plan_kernel(X_T, c["n_in"], c["ta"], c["tb"], c["trel"], c["ptr"], c["idx"], c["sgn"],
                    self.syms[:, :X_T.shape[1]], out)
        return out


class FastNaiveMatmul:
    def __init__(self, W):
        self.c = compile_naive(W)

    def __call__(self, X_T, out):
        c = self.c
        naive_kernel(X_T, c["ptr"], c["idx"], c["sgn"], out)
        return out
