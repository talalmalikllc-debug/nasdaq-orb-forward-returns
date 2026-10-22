"""Small stats helpers (no scipy in this environment)."""
from __future__ import annotations
import numpy as np

RNG = np.random.default_rng(20260813)


def pf(r):
    r = np.asarray(r, float)
    g = r[r > 0].sum()
    b = -r[r < 0].sum()
    return np.inf if b == 0 else g / b


def summary(r):
    r = np.asarray(r, float)
    n = len(r)
    if n == 0:
        return dict(n=0, wr=np.nan, pf=np.nan, avg=np.nan, se=np.nan, t=np.nan)
    avg = r.mean()
    se = r.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return dict(n=n, wr=100.0 * (r > 0).mean(), pf=pf(r), avg=avg, se=se,
                t=avg / se if se and se > 0 else np.nan)


def boot_diff(a, b, iters=20000):
    """Bootstrap P(mean(a) - mean(b) <= 0) and the CI of the difference."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 5 or len(b) < 5:
        return np.nan, (np.nan, np.nan)
    da = RNG.choice(a, (iters, len(a)), replace=True).mean(1)
    db = RNG.choice(b, (iters, len(b)), replace=True).mean(1)
    d = da - db
    return float((d <= 0).mean()), (float(np.percentile(d, 2.5)),
                                    float(np.percentile(d, 97.5)))


def spearman(x, y):
    """Rank correlation + a permutation p-value (two-sided)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 10:
        return np.nan, np.nan
    rx, ry = _rank(x), _rank(y)
    r = np.corrcoef(rx, ry)[0, 1]
    null = np.empty(5000)
    for i in range(5000):
        null[i] = np.corrcoef(rx, RNG.permutation(ry))[0, 1]
    return float(r), float((np.abs(null) >= abs(r)).mean())


def _rank(x):
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    r[o] = np.arange(len(x), dtype=float)
    return r


def bucket(v, q=5, edges=None):
    """Full-sample q-tiles -> integer bucket 0..q-1 (nan-safe)."""
    v = np.asarray(v, float)
    out = np.full(len(v), -1, int)
    m = np.isfinite(v)
    if edges is None:
        edges = np.percentile(v[m], np.linspace(0, 100, q + 1)[1:-1])
    out[m] = np.searchsorted(edges, v[m], side="right")
    return out, edges
