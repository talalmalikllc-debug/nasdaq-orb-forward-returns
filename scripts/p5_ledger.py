#!/usr/bin/env python3
"""P5 - every candidate in ONE metric: fwd60 on US100, binary group vs rest.

Earlier studies reported their results in different currencies - R under a
chandelier exit, expectancy per unit risk, ATR forward return - which makes
them hard to rank against each other. This puts all of them through the
identical grader:

    group mean fwd60 (ATR, net of spread) minus the rest of the sample
    + bootstrap CI, placebo, per-year sign agreement, IS/OOS

so the ledger is comparable end to end. Nothing new is tested here; this is a
common scale for what has already been found.
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
from p3_validate import grade


def attach_drift(bars, ses, rs):
    import mom_core as mc
    r_start = oc.RANGE_START_H * 3600 + oc.RANGE_START_M * 60
    for r in rs:
        mo = mc.momentum(bars, ses, r["k"], r_start)
        for w in (5, 15, 30, 60):
            v = mo[w] / r["atr"] if (mo and np.isfinite(mo[w])) else np.nan
            r[f"drift{w}"] = r["side"] * v
    return rs


if __name__ == "__main__":
    bars, ses, rs = fh.sessions_and_breakouts()
    rs = add_features(bars, ses, rs)
    rs = attach_drift(bars, ses, rs)
    sod = oc.sec_of_day(bars["t"])
    r_end = (oc.RANGE_START_H * 3600 + oc.RANGE_START_M * 60 + oc.RANGE_MIN * 60)
    for r in rs:
        r["immediate"] = bool(sod[r["i_fill"]] == r_end)

    y = fh.fwd(rs, 60)
    st = su.summary(y[np.isfinite(y)])
    print(f"US100.cash, NY 16:30 +30m opening range, 60-min entry window")
    print(f"n={len(rs)}   baseline fwd60 {st['avg']:+.4f} ATR (t={st['t']:+.2f})")
    print("every row: group mean minus the rest of the sample, fwd60, net of spread")

    out = []
    out.append(grade(rs, lambda r: np.isfinite(r["drift5"]) and r["drift5"] > 0,
                     "PRE-SESSION DRIFT, 5-min, aligned"))
    out.append(grade(rs, lambda r: np.isfinite(r["drift15"]) and r["drift15"] > 0,
                     "PRE-SESSION DRIFT, 15-min, aligned"))
    out.append(grade(rs, lambda r: np.isfinite(r["drift30"]) and r["drift30"] > 0,
                     "PRE-SESSION DRIFT, 30-min, aligned"))
    out.append(grade(rs, lambda r: np.isfinite(r["drift60"]) and r["drift60"] > 0,
                     "PRE-SESSION DRIFT, 60-min, aligned (expected dead)"))
    out.append(grade(rs, lambda r: not r["immediate"],
                     "LATER BREAK (inside the range at 17:00)"))
    out.append(grade(rs, lambda r: int(r["dow"]) in (2, 5), "TUESDAY or FRIDAY"))
    out.append(grade(rs, lambda r: int(r["dow"]) == 1, "MONDAY"))
    out.append(grade(rs, lambda r: (np.isfinite(r["drift5"]) and r["drift5"] > 0
                                    and not r["immediate"]),
                     "DRIFT-aligned AND later break (combined)"))

    print("\n" + "=" * 104)
    print("US100 FORWARD-RETURN LEDGER  (fwd60, ATR units, net of spread)")
    print("=" * 104)
    print(f"  {'candidate':<44}{'n':>6}{'diff':>10}{'placebo':>9}"
          f"{'years':>8}{'IS':>10}{'OOS':>10}   status")
    print("  " + "-" * 100)
    for r in out:
        if r:
            print(f"  {r['name'][:43]:<44}{r['n']:>6}{r['diff']:>+10.4f}"
                  f"{r['placebo']:>9.3f}{r['years']:>8}{r['IS']:>+10.4f}"
                  f"{r['OOS']:>+10.4f}   {'PASS' if r['ok'] else '-'}")
    print("\n  PASS = placebo < 0.05, >=4/5 years agree, IS and OOS share the sign.")
    print("  Weekday rows additionally need the selection-aware placebo in p4")
    print("  (best-2-of-5 from noise averages +0.023 ATR, p95 +0.040).")
