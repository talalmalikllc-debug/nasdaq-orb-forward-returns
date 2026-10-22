#!/usr/bin/env python3
"""P7 - is the weekday result just the Monday effect?

P6 showed Tue+Fri positive AND Monday negative at every horizon, 5/5 years.
Those are not two findings. If Monday is the bad day, then the best two of the
remaining four will look good almost mechanically, and "Tue or Fri" is a
laundered version of "not Monday".

That matters for the multiple-comparison problem. "Best 2 of 5 days" is a
search and needs the selection-aware null. "Monday is weak" is a documented
prior in the equity literature (the weekend effect), so it is closer to a
pre-registered hypothesis than to a data-mined one.

Tested here: NOT-MONDAY as a single grouping, and the selection-aware null
re-run at the horizon where the effect is strongest rather than at fwd60.
"""
from __future__ import annotations
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("BROKER_DATA_DIR", _os.path.join(_REPO, "data", "raw"))
import sys, os, itertools
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import orb_core as oc
import fwd_harness as fh
import stats_util as su
from p1_batch import add_features
from p6_horizons import grade_h, fwd_h, HORIZONS

if __name__ == "__main__":
    bars, ses, rs = fh.sessions_and_breakouts()
    rs = add_features(bars, ses, rs)

    print("### NOT MONDAY  (single grouping, mechanism-backed prior)")
    print(f"  {'horizon':<10}{'diff':>10}{'placebo':>10}{'years':>8}{'IS':>10}{'OOS':>10}")
    for h in HORIZONS:
        d = grade_h(rs, lambda r: int(r["dow"]) != 1, h)
        if d:
            print(f"  fwd{str(h):<7}{d['diff']:>+10.4f}{d['placebo']:>10.3f}"
                  f"{str(d['agree'])+'/'+str(d['tot']):>8}{d['IS']:>+10.4f}{d['OOS']:>+10.4f}")

    print("\n### per-day means at the long horizon (fwd_flat)")
    names = {1:"Mon",2:"Tue",3:"Wed",4:"Thu",5:"Fri"}
    y = fwd_h(rs, "flat")
    dow = np.array([int(r["dow"]) for r in rs])
    m = np.isfinite(y); y, dow = y[m], dow[m]
    for i in range(1, 6):
        v = y[dow == i]
        st = su.summary(v)
        print(f"  {names[i]:<5} n={st['n']:<5} {st['avg']:>+9.4f}  t={st['t']:>+5.2f}")

    print("\n### selection-aware null at fwd_flat (best 2 of 5, search redone)")
    def best_pair(yy, dd):
        best = -np.inf
        for c in itertools.combinations(range(1, 6), 2):
            mm = np.isin(dd, c)
            if mm.sum() < 50 or (~mm).sum() < 50: continue
            best = max(best, yy[mm].mean() - yy[~mm].mean())
        return best
    def worst_single(yy, dd):
        w = np.inf
        for d_ in range(1, 6):
            mm = dd == d_
            if mm.sum() < 30: continue
            w = min(w, yy[mm].mean() - yy[~mm].mean())
        return w
    real_pair = best_pair(y, dow)
    real_worst = worst_single(y, dow)
    N = 5000
    np_, nw_ = np.empty(N), np.empty(N)
    for i in range(N):
        s = su.RNG.permutation(dow)
        np_[i] = best_pair(y, s); nw_[i] = worst_single(y, s)
    print(f"  best pair          real {real_pair:+.4f}   selection-aware P = "
          f"{(np_ >= real_pair).mean():.4f}   (noise mean {np_.mean():+.4f}, "
          f"p95 {np.percentile(np_,95):+.4f})")
    print(f"  worst single day   real {real_worst:+.4f}   selection-aware P = "
          f"{(nw_ <= real_worst).mean():.4f}   (noise mean {nw_.mean():+.4f}, "
          f"p05 {np.percentile(nw_,5):+.4f})")
