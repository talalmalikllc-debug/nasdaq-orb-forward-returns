#!/usr/bin/env python3
"""P2 - the one candidate from the batch, tested properly.

Two of P1's ten features are discrete and broke its quantile bucketing (many
tied values pile into one bucket, so Q5-Q1 came back NaN and the placebo
degenerated to 0.000 - an artefact, not a result). Both pointed the same way:

    breakouts that trigger IMMEDIATELY at the range end do worse than
    breakouts that arrive some minutes later

and they are largely the same statement - a later trigger means more bars in
which the edge could be touched. There is a plain mechanism to check: if price
is already beyond the edge when the range closes at 17:00, there is no breakout
EVENT at all, just the continuation of a move that happened during the range.
The order fills at 17:00 because it was placed inside the market, not because
anything broke.

So the clean binary is: was price already outside the range at 17:00?
That is known exactly at 17:00, before any order exists.

Day of week is re-run too - P1's labels were shifted a day (epoch day 0 is a
Thursday, so the modulo indexing put Monday under "Tue").
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

SPLIT = fh.SPLIT
HOR = fh.HOR


def group_report(rs, groups, title, note):
    print(f"\n### {title}   [{note}]")
    print(f"  {'group':<22}{'n':>6}" + "".join(f"{'fwd'+str(x):>9}" for x in HOR)
          + f"{'flat':>9}{'t@60':>7}")
    out = {}
    for lab, sel in groups:
        g = [r for r in rs if sel(r)]
        if len(g) < 25:
            print(f"  {lab:<22}{len(g):>6}   (too few)")
            continue
        v = fh.fwd(g, 60); v = v[np.isfinite(v)]
        st = su.summary(v)
        print(f"  {lab:<22}{len(g):>6}" +
              "".join(f"{np.nanmean(fh.fwd(g,x)):>+9.4f}" for x in HOR) +
              f"{np.nanmean(fh.col(g,'fwd_flat'))-np.nanmean(fh.col(g,'cost')):>+9.4f}"
              f"{st['t']:>7.2f}")
        out[lab] = g
    return out


def contrast(a, b, la, lb, rs_all, selfn):
    va, vb = fh.fwd(a, 60), fh.fwd(b, 60)
    va, vb = va[np.isfinite(va)], vb[np.isfinite(vb)]
    p, ci = su.boot_diff(va, vb)
    print(f"\n  {la} minus {lb}: {va.mean()-vb.mean():+.4f} ATR  "
          f"95% CI [{ci[0]:+.4f},{ci[1]:+.4f}]  P(no gain)={p:.3f}")
    # placebo: shuffle the group label across sessions
    y = fh.fwd(rs_all, 60)
    lab = np.array([1 if selfn(r) else 0 for r in rs_all])
    m = np.isfinite(y)
    y2, lab2 = y[m], lab[m]
    null = np.empty(3000)
    for i in range(3000):
        s = su.RNG.permutation(lab2)
        null[i] = y2[s == 1].mean() - y2[s == 0].mean()
    real = y2[lab2 == 1].mean() - y2[lab2 == 0].mean()
    print(f"  placebo (label shuffled): P(|shuffled| >= |real|) = "
          f"{(np.abs(null) >= abs(real)).mean():.3f}")
    # per-year and IS/OOS
    yr = np.array([int(r["year"]) for r in rs_all])[m]
    print(f"  {'year':<8}{'n a':>6}{'a':>10}{'n b':>6}{'b':>10}{'diff':>10}")
    agree = 0; tot = 0
    for yy in sorted(set(yr)):
        sa = y2[(yr == yy) & (lab2 == 1)]; sb = y2[(yr == yy) & (lab2 == 0)]
        if len(sa) < 10 or len(sb) < 10:
            continue
        tot += 1
        d = sa.mean() - sb.mean()
        if np.sign(d) == np.sign(real):
            agree += 1
        print(f"  {yy:<8}{len(sa):>6}{sa.mean():>+10.4f}{len(sb):>6}"
              f"{sb.mean():>+10.4f}{d:>+10.4f}")
    ts = np.array([r["t"] for r in rs_all])[m]
    for nm, cond in (("IS ", ts < SPLIT), ("OOS", ts >= SPLIT)):
        sa = y2[cond & (lab2 == 1)]; sb = y2[cond & (lab2 == 0)]
        if len(sa) > 10 and len(sb) > 10:
            print(f"  {nm:<8}{len(sa):>6}{sa.mean():>+10.4f}{len(sb):>6}"
                  f"{sb.mean():>+10.4f}{sa.mean()-sb.mean():>+10.4f}")
    print(f"  years agreeing with the pooled sign: {agree}/{tot}")


if __name__ == "__main__":
    bars, ses, rs = fh.sessions_and_breakouts()
    rs = add_features(bars, ses, rs)
    sod = oc.sec_of_day(bars["t"])
    r_end = (oc.RANGE_START_H * 3600 + oc.RANGE_START_M * 60
             + oc.RANGE_MIN * 60)
    for r in rs:
        r["immediate"] = bool(sod[r["i_fill"]] == r_end)
    y = fh.fwd(rs, 60)
    print(f"US100.cash breakouts {len(rs)}   baseline fwd60 "
          f"{np.nanmean(y):+.4f} ATR (t={su.summary(y[np.isfinite(y)])['t']:+.2f})")

    g = group_report(rs, [
        ("0 min (first bar)", lambda r: r["trig_min"] == 0),
        ("5-10 min", lambda r: 0 < r["trig_min"] <= 10),
        ("15-30 min", lambda r: 10 < r["trig_min"] <= 30),
        ("35-60 min", lambda r: r["trig_min"] > 30),
    ], "A. Minutes from the range end to the trigger", "known at the trigger")

    print("\n### B. The mechanism: was price ALREADY outside the range at 17:00?")
    imm = [r for r in rs if r["immediate"]]
    lat = [r for r in rs if not r["immediate"]]
    group_report(rs, [("already outside", lambda r: r["immediate"]),
                      ("broke out later", lambda r: not r["immediate"])],
                 "already outside at 17:00 vs a genuine later break",
                 "known exactly at 17:00")
    contrast(lat, imm, "later break", "already outside", rs,
             lambda r: not r["immediate"])

    print("\n### C. Bars touching the edge before the break")
    group_report(rs, [
        ("0 touches", lambda r: r["touches"] == 0),
        ("1 touch", lambda r: r["touches"] == 1),
        ("2+ touches", lambda r: r["touches"] >= 2),
    ], "edge touches", "known at the trigger")

    print("\n### D. Day of week (labels corrected)")
    names = {0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday",
             4: "Thursday", 5: "Friday", 6: "Saturday"}
    group_report(rs, [(names[i], (lambda r, i=i: int(r["dow"]) == i))
                      for i in range(7)], "day of week", "known in advance")
