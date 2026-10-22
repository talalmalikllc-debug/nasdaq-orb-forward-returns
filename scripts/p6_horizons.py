#!/usr/bin/env python3
"""P6 - the same candidates across horizons.

P5 graded everything at fwd60. That is the wrong ruler for a slow effect: the
pre-session drift builds over the session (fwd15 +0.002 -> fwd120 +0.082 in the
original study), so a 60-minute window truncates most of it. Judging a
candidate at one arbitrary horizon is itself a hidden choice, so here every
candidate is graded at every horizon and the profile is read as a whole.

A real effect should show a coherent shape across horizons. Noise jumps around.
"""
from __future__ import annotations
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("BROKER_DATA_DIR", _os.path.join(_REPO, "data", "raw"))
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import orb_core as oc
import fwd_harness as fh
import stats_util as su
from p1_batch import add_features
from p5_ledger import attach_drift

HORIZONS = [15, 30, 60, 120, "flat"]


def fwd_h(rs, h):
    if h == "flat":
        return fh.col(rs, "fwd_flat") - fh.col(rs, "cost")
    return fh.fwd(rs, h)


def grade_h(rs, sel, h, iters=3000):
    y = fwd_h(rs, h)
    lab = np.array([1 if sel(r) else 0 for r in rs])
    m = np.isfinite(y)
    y, lab = y[m], lab[m]
    sub = [r for r, ok in zip(rs, m) if ok]
    a, b = y[lab == 1], y[lab == 0]
    if len(a) < 40:
        return None
    real = a.mean() - b.mean()
    null = np.empty(iters)
    for i in range(iters):
        s = su.RNG.permutation(lab)
        null[i] = y[s == 1].mean() - y[s == 0].mean()
    pl = float((np.abs(null) >= abs(real)).mean())
    yr = np.array([int(r["year"]) for r in sub])
    ts = np.array([r["t"] for r in sub])
    agree = tot = 0
    for yy in sorted(set(yr)):
        sa, sb = y[(yr == yy) & (lab == 1)], y[(yr == yy) & (lab == 0)]
        if len(sa) < 8 or len(sb) < 8:
            continue
        tot += 1
        if np.sign(sa.mean() - sb.mean()) == np.sign(real):
            agree += 1
    iso = []
    for cond in (ts < fh.SPLIT, ts >= fh.SPLIT):
        sa, sb = y[cond & (lab == 1)], y[cond & (lab == 0)]
        iso.append(sa.mean() - sb.mean() if len(sa) > 8 and len(sb) > 8 else np.nan)
    return dict(diff=real, placebo=pl, agree=agree, tot=tot, IS=iso[0], OOS=iso[1])


if __name__ == "__main__":
    bars, ses, rs = fh.sessions_and_breakouts()
    rs = add_features(bars, ses, rs)
    rs = attach_drift(bars, ses, rs)
    sod = oc.sec_of_day(bars["t"])
    r_end = (oc.RANGE_START_H*3600 + oc.RANGE_START_M*60 + oc.RANGE_MIN*60)
    for r in rs:
        r["immediate"] = bool(sod[r["i_fill"]] == r_end)

    CANDS = [
        ("PRE-SESSION DRIFT 5-min aligned",
         lambda r: np.isfinite(r["drift5"]) and r["drift5"] > 0),
        ("LATER BREAK (inside range at 17:00)", lambda r: not r["immediate"]),
        ("TUESDAY or FRIDAY", lambda r: int(r["dow"]) in (2, 5)),
        ("MONDAY", lambda r: int(r["dow"]) == 1),
    ]
    for name, sel in CANDS:
        print(f"\n### {name}")
        print(f"  {'horizon':<10}{'diff':>10}{'placebo':>10}{'years':>8}"
              f"{'IS':>10}{'OOS':>10}")
        for h in HORIZONS:
            d = grade_h(rs, sel, h)
            if d is None:
                continue
            print(f"  fwd{str(h):<7}{d['diff']:>+10.4f}{d['placebo']:>10.3f}"
                  f"{str(d['agree'])+'/'+str(d['tot']):>8}{d['IS']:>+10.4f}"
                  f"{d['OOS']:>+10.4f}")
