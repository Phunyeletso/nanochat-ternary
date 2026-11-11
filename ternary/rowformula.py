"""
Phase 4: "row formula" matmul for ternary matrices (reference implementation, plain Python + NumPy).

Idea: y = W x with W in {-1,0,+1}. Row r is a signed sum of inputs. If many rows contain the same
signed pair (e.g. x1 + x2, or x3 - x7), compute it once as a temporary and reuse it.
This is greedy common-subexpression elimination (in the spirit of Paar's algorithm for XOR circuits):

    repeat:
        find the signed pair (a, b, rel) that occurs in the most rows (rel = +1: a+b, -1: a-b)
        if it occurs in fewer than 2 rows: stop
        create temporary t = s_a + rel * s_b                         (costs 1 addition)
        in every row containing s_a with sign g and s_b with sign g*rel: replace both by (t, sign g)
Temporaries can themselves be paired later, so formulas nest.

Cost model (used identically for the naive method and the plan):
    a temporary costs 1 add/sub;
    a row with k terms costs k-1 add/subs, plus 1 negation if all its terms have sign -1 (k >= 1).
Naive = the same formula applied to the raw rows of W (no sharing).

A plan is:
    n_in                    number of inputs; symbol ids 0..n_in-1 are the inputs
    temps[i] = (a, b, rel)  symbol id n_in+i = sym[a] + rel * sym[b]   (a < b, created in order)
    rows[r] = {sym: sign}   output r = sum(sign * sym)
"""
import heapq
import time
from collections import defaultdict

import numpy as np


# ----------------------------------------------------------------------------- cost model
def row_cost(signs):
    k = len(signs)
    if k == 0:
        return 0
    return k - 1 + (0 if any(s > 0 for s in signs) else 1)


def naive_adds(W):
    W = np.asarray(W)
    total = 0
    for r in range(W.shape[0]):
        nz = W[r][W[r] != 0]
        total += row_cost(nz.tolist())
    return total


# ----------------------------------------------------------------------------- plan
class Plan:
    def __init__(self, n_in, n_out):
        self.n_in, self.n_out = n_in, n_out
        self.temps = []          # list of (a, b, rel)
        self.rows = []           # list of dict sym -> sign
        self.alive = []          # temp still used (patching can orphan temps)

    def sym_count(self):
        return self.n_in + len(self.temps)

    def adds(self):
        return sum(self.alive) + sum(row_cost(list(r.values())) for r in self.rows)

    def n_temps(self):
        return sum(self.alive)

    # ---- evaluation (reference: straightforward NumPy, exact for integer inputs)
    def evaluate(self, X):
        """X: (N, n_in) integer array -> (N, n_out) int64, computed only with adds/subs/negations."""
        X = np.asarray(X, dtype=np.int64)
        N = X.shape[0]
        syms = np.zeros((N, self.sym_count()), dtype=np.int64)
        syms[:, :self.n_in] = X
        for i, (a, b, rel) in enumerate(self.temps):
            if self.alive[i]:
                syms[:, self.n_in + i] = syms[:, a] + syms[:, b] if rel > 0 else syms[:, a] - syms[:, b]
        out = np.zeros((N, self.n_out), dtype=np.int64)
        for r, row in enumerate(self.rows):
            acc = np.zeros(N, dtype=np.int64)
            for s, g in row.items():
                acc = acc + syms[:, s] if g > 0 else acc - syms[:, s]
            out[:, r] = acc
        return out

    def evaluate_counting(self, x):
        """Evaluate ONE input vector in pure Python, literally counting every add/sub/negation.
        Used to check that adds() is an honest count."""
        syms = list(int(v) for v in x) + [0] * len(self.temps)
        ops = 0
        for i, (a, b, rel) in enumerate(self.temps):
            if self.alive[i]:
                syms[self.n_in + i] = syms[a] + syms[b] if rel > 0 else syms[a] - syms[b]
                ops += 1
        out = []
        for row in self.rows:
            items = sorted(row.items(), key=lambda kv: -kv[1])   # start from a positive term if any
            if not items:
                out.append(0); continue
            s0, g0 = items[0]
            acc = syms[s0] if g0 > 0 else -syms[s0]
            ops += 0 if g0 > 0 else 1
            for s, g in items[1:]:
                acc = acc + syms[s] if g > 0 else acc - syms[s]
                ops += 1
            out.append(acc)
        return out, ops

    def expansion(self, sym, cache=None):
        """dict input -> coefficient for a symbol (inputs or temporaries)."""
        if cache is None:
            cache = {}
        if sym < self.n_in:
            return {sym: 1}
        if sym in cache:
            return cache[sym]
        a, b, rel = self.temps[sym - self.n_in]
        e = dict(self.expansion(a, cache))
        for j, c in self.expansion(b, cache).items():
            e[j] = e.get(j, 0) + rel * c
        e = {j: c for j, c in e.items() if c != 0}
        cache[sym] = e
        return e


def _key(s1, g1, s2, g2):
    return (s1, s2, g1 * g2) if s1 < s2 else (s2, s1, g1 * g2)


def build_plan(W, min_count=2, max_temps=None):
    """Greedy pair sharing. W: (n_out, n_in) array with values in {-1,0,1}."""
    W = np.asarray(W)
    n_out, n_in = W.shape
    plan = Plan(n_in, n_out)
    rows = []
    occ = defaultdict(set)                       # sym -> set of rows containing it
    for r in range(n_out):
        nz = np.nonzero(W[r])[0]
        row = {int(j): int(W[r, j]) for j in nz}
        rows.append(row)
        for j in row:
            occ[j].add(r)

    count = defaultdict(int)
    for row in rows:
        items = sorted(row.items())
        for i in range(len(items)):
            s1, g1 = items[i]
            for s2, g2 in items[i + 1:]:
                count[(s1, s2, g1 * g2)] += 1
    heap = [(-c, k) for k, c in count.items() if c >= min_count]
    heapq.heapify(heap)

    while heap:
        negc, key = heapq.heappop(heap)
        cur = count.get(key, 0)
        if cur != -negc:                          # stale entry: re-push with current count
            if cur >= min_count:
                heapq.heappush(heap, (-cur, key))
            continue
        if cur < min_count:
            break
        if max_temps is not None and len(plan.temps) >= max_temps:
            break
        a, b, rel = key
        t = n_in + len(plan.temps)
        plan.temps.append((a, b, rel))
        plan.alive.append(True)
        hit = [r for r in (occ[a] & occ[b]) if rows[r][b] == rows[r][a] * rel]
        new_keys = set()
        for r in hit:
            row = rows[r]
            ga, gb = row[a], row[b]
            for s, g in row.items():
                if s == a or s == b:
                    continue
                count[_key(a, ga, s, g)] -= 1
                count[_key(b, gb, s, g)] -= 1
            count[key] -= 1
            del row[a], row[b]
            occ[a].discard(r); occ[b].discard(r)
            for s, g in row.items():
                k2 = _key(t, ga, s, g)
                count[k2] += 1
                new_keys.add(k2)
            row[t] = ga
            occ[t].add(r)
        for k2 in new_keys:
            if count[k2] >= min_count:
                heapq.heappush(heap, (-count[k2], k2))
    plan.rows = rows
    plan._occ = occ
    return plan


# ----------------------------------------------------------------------------- incremental patching
class PatchablePlan:
    """Wraps a Plan so that rows can be re-expressed after weights flip, WITHOUT a full rebuild.
    Patching a row: take its new raw terms (from the new W row) and re-apply EXISTING temporaries
    greedily (oldest first, i.e. the same order the builder created them). No new temporaries are
    created, so quality can drift; temporaries no longer used by anything are marked dead (not computed)."""

    def __init__(self, plan):
        self.p = plan
        n_in = plan.n_in
        self.users = defaultdict(list)            # sym -> temps that use it as an operand
        for i, (a, b, rel) in enumerate(plan.temps):
            self.users[a].append(n_in + i)
            self.users[b].append(n_in + i)
        self.refs = defaultdict(int)              # how many rows/temps reference each temp
        for i, (a, b, rel) in enumerate(plan.temps):
            self.refs[a] += 1
            self.refs[b] += 1
        for row in plan.rows:
            for s in row:
                self.refs[s] += 1

    def _deref(self, s):
        n_in = self.p.n_in
        self.refs[s] -= 1
        if s >= n_in and self.refs[s] == 0:
            i = s - n_in
            if self.p.alive[i]:
                self.p.alive[i] = False
                a, b, _ = self.p.temps[i]
                self._deref(a); self._deref(b)

    def _ref(self, s):
        n_in = self.p.n_in
        self.refs[s] += 1
        if s >= n_in and self.refs[s] == 1:
            i = s - n_in
            if not self.p.alive[i]:
                self.p.alive[i] = True
                a, b, _ = self.p.temps[i]
                self._ref(a); self._ref(b)

    def _cover(self, row):
        """Greedily re-apply existing temporaries to a raw row {input: sign}."""
        n_in = self.p.n_in
        cand = []
        for s in row:
            for t in self.users[s]:
                heapq.heappush(cand, t)
        seen = set()
        while cand:
            t = heapq.heappop(cand)
            if t in seen:
                continue
            a, b, rel = self.p.temps[t - n_in]
            if a in row and b in row and row[b] == row[a] * rel:
                g = row.pop(a); row.pop(b)
                row[t] = g
                seen.add(t)
                for t2 in self.users[t]:
                    heapq.heappush(cand, t2)
        return row

    def patch(self, W_new, changed_rows):
        for r in changed_rows:
            for s in self.p.rows[r]:
                self._deref(s)
            raw = {int(j): int(W_new[r, j]) for j in np.nonzero(W_new[r])[0]}
            row = self._cover(raw)
            for s in row:
                self._ref(s)
            self.p.rows[r] = row


# ----------------------------------------------------------------------------- helpers
def random_ternary(n_out, n_in, zero_frac, rng):
    W = rng.choice(np.array([-1, 1], dtype=np.int8), size=(n_out, n_in))
    W[rng.random((n_out, n_in)) < zero_frac] = 0
    return W


def timed_build(W):
    t0 = time.perf_counter()
    p = build_plan(W)
    return p, time.perf_counter() - t0
