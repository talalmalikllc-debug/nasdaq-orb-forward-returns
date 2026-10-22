#!/usr/bin/env python3
"""P4 - the selection-aware placebo for the weekday result.

P3 reported "TUESDAY or FRIDAY" as validated: diff +0.0413 ATR, placebo
P=0.004, 4/5 years, IS and OOS the same sign. That test is wrong, and the way
it is wrong is the subject of this file.

"Tuesday or Friday" was never a hypothesis. It is the union of the two winners
of a five-way search. P3's placebo shuffled the labels for a FIXED grouping, so
it answers "given that I care about Tue+Fri, could this arise by chance?" - but
the reason anyone cares about Tue+Fri is that they won. The null has to include
the search.

So: for each shuffle, permute the weekday labels across sessions, then run the
SAME procedure - find the best 2 of 5 days and measure that pair against the
rest. The real +0.0413 is then compared to a null of best-of-search results
rather than to a null of one fixed comparison.

The difference between those two p-values is the entire lesson.
"""
from __future__ import annotations
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("BROKER_DATA_DIR", _os.path.join(_REPO, "data", "raw"))
import sys, os
import itertools
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import fwd_harness as fh
import stats_util as su
from p1_batch import add_features

ITERS = 5000
DAYS = [1, 2, 3, 4, 5]          # Mon..Fri


def best_pair(y, dow, days=DAYS):
    """Best 2-of-5 grouping by (group mean - rest mean)."""
    best, which = -np.inf, None
    for combo in itertools.combinations(days, 2):
        m = np.isin(dow, combo)
        if m.sum() < 50 or (~m).sum() < 50:
            continue
        d = y[m].mean() - y[~m].mean()
        if d > best:
            best, which = d, combo
    return best, which


def best_single(y, dow, days=DAYS):
    best, which = -np.inf, None
    for d_ in days:
        m = dow == d_
        if m.sum() < 30:
            continue
        v = y[m].mean() - y[~m].mean()
        if v > best:
            best, which = v, d_
    return best, which


if __name__ == "__main__":
    bars, ses, rs = fh.sessions_and_breakouts()
    rs = add_features(bars, ses, rs)
    y = fh.fwd(rs, 60)
    dow = np.array([int(r["dow"]) for r in rs])
    m = np.isfinite(y)
    y, dow = y[m], dow[m]
    names = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri"}

    real_pair, which_pair = best_pair(y, dow)
    real_one, which_one = best_single(y, dow)
    print(f"n={len(y)}   baseline fwd60 {y.mean():+.4f} ATR")
    print(f"\n  best SINGLE day  : {names[which_one]}  diff {real_one:+.4f}")
    print(f"  best PAIR of days: {names[which_pair[0]]}+{names[which_pair[1]]}"
          f"  diff {real_pair:+.4f}")

    print(f"\n  running {ITERS} shuffles, redoing the search inside each one...")
    null_pair = np.empty(ITERS)
    null_one = np.empty(ITERS)
    for i in range(ITERS):
        s = su.RNG.permutation(dow)
        null_pair[i] = best_pair(y, s)[0]
        null_one[i] = best_single(y, s)[0]

    print("\n" + "=" * 78)
    print("THE TWO NULLS, SIDE BY SIDE")
    print("=" * 78)
    # naive: fixed grouping, labels shuffled
    fixed = np.isin(dow, which_pair)
    naive = np.empty(ITERS)
    for i in range(ITERS):
        s = su.RNG.permutation(fixed)
        naive[i] = y[s].mean() - y[~s].mean()
    p_naive = float((naive >= real_pair).mean())
    p_sel = float((null_pair >= real_pair).mean())
    print(f"  naive placebo (grouping fixed in advance)   P = {p_naive:.4f}")
    print(f"  selection-aware placebo (search redone)     P = {p_sel:.4f}")
    print(f"    null of best-pair searches: mean {null_pair.mean():+.4f}, "
          f"p95 {np.percentile(null_pair,95):+.4f}, max {null_pair.max():+.4f}")
    print(f"    the real result {real_pair:+.4f} sits at the "
          f"{100*(null_pair < real_pair).mean():.1f}th percentile of that null")

    p_one = float((null_one >= real_one).mean())
    print(f"\n  best single day, selection-aware            P = {p_one:.4f}")
    print(f"    null of best-single searches: mean {null_one.mean():+.4f}, "
          f"p95 {np.percentile(null_one,95):+.4f}")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if p_sel < 0.05:
        print("  The weekday effect survives its own search. Worth carrying.")
    else:
        print("  The weekday effect does NOT survive its own search.")
        print(f"  Picking the best 2 of 5 days from PURE NOISE produces a gap of")
        print(f"  {null_pair.mean():+.4f} on average and {np.percentile(null_pair,95):+.4f} "
              f"at the 95th percentile.")
        print(f"  The observed {real_pair:+.4f} is inside that range, so it is what the")
        print(f"  search procedure manufactures, not what the market did.")
