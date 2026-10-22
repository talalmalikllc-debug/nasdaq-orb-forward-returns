#!/usr/bin/env python3
"""P8 - the Friday result at the session-close horizon, and the checks on it.

P7 found the weekday effect is strongest at fwd_flat (exit at the 16:00 NY
flat), not at fwd60. This file re-runs the selection-aware null for the best
SINGLE day at that horizon, then asks the obvious follow-up questions:

  1. Is it an option-expiry artefact (3rd Friday of the month)?
  2. Is it stable year by year?
  3. Is it the pre-session drift signal from P5 in disguise? (2 x 2 cells)
"""
from __future__ import annotations
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fwd_harness as fh          # noqa: E402
import stats_util as su           # noqa: E402
from p1_batch import add_features  # noqa: E402
from p4_selection_aware import best_single  # noqa: E402
from p5_ledger import attach_drift  # noqa: E402
from p6_horizons import fwd_h      # noqa: E402

ITERS = 100_000


def _date(r):
    return datetime.fromtimestamp(int(r["t"]), tz=timezone.utc)


if __name__ == "__main__":
    bars, ses, rs = fh.sessions_and_breakouts()
    rs = add_features(bars, ses, rs)
    rs = attach_drift(bars, ses, rs)

    y_all = fwd_h(rs, "flat")
    keep = np.isfinite(y_all)
    rs = [r for r, k in zip(rs, keep) if k]
    y = y_all[keep]
    dow = np.array([int(r["dow"]) for r in rs])
    dt = [_date(r) for r in rs]

    real, which = best_single(y, dow)
    null = np.array([best_single(y, su.RNG.permutation(dow))[0] for _ in range(ITERS)])
    print(f"BEST SINGLE DAY at fwd_flat: real {real:+.4f}  selection-aware P = "
          f"{(null >= real).mean():.4f}")
    print(f"  (noise mean {null.mean():+.4f}, p95 {np.percentile(null, 95):+.4f})")

    fri = dow == 5
    nth = np.array([(d.day - 1) // 7 + 1 for d in dt])
    print("\n### Is the Friday effect concentrated in option-expiry Fridays?")
    for k in range(1, 6):
        st = su.summary(y[fri & (nth == k)])
        tag = " <- monthly OpEx week" if k == 3 else ""
        print(f"  Friday #{k} of month  n={st['n']:<6} {st['avg']:+.4f}  t={st['t']:+.2f}{tag}")
    st = su.summary(y[fri & (nth != 3)])
    print(f"  Fridays EXCLUDING the 3rd  n={st['n']:<6} {st['avg']:+.4f}  t={st['t']:+.2f}")

    print("\n### Friday per-year at fwd_flat")
    yr = np.array([d.year for d in dt])
    for Y in sorted(set(yr)):
        v = y[fri & (yr == Y)]
        print(f"  {Y}: n={len(v):<5} {v.mean():+.4f}")

    print("\n### Friday x pre-session drift (5-min, aligned with the trade)")
    al = np.array([np.isfinite(r["drift5"]) and r["drift5"] > 0 for r in rs])
    cells = [
        ("all breakouts", np.ones(len(y), bool)),
        ("Friday only", fri),
        ("drift-aligned only", al),
        ("Friday AND drift-aligned", fri & al),
        ("Friday, drift against", fri & ~al),
        ("non-Friday, drift-aligned", ~fri & al),
        ("non-Friday, drift against", ~fri & ~al),
    ]
    print(f"{'cell':<30}{'n':>6}{'fwd_flat':>11}{'t':>7}")
    for name, m in cells:
        st = su.summary(y[m])
        print(f"  {name:<28}{st['n']:>6}{st['avg']:>+11.4f}{st['t']:>7.2f}")
    a, b = y[fri & al], y[~fri & ~al]
    p, (lo, hi) = su.boot_diff(a, b)
    print(f"\n  best cell minus worst cell: {a.mean() - b.mean():+.4f} ATR  "
          f"95% CI [{lo:+.4f},{hi:+.4f}]  P={p:.3f}")

    # per-breakout ledger (normalised returns only, no prices) for the
    # figures and the notebook, so they run without the broker export
    hz = {h: fwd_h(rs, h) for h in (15, 30, 60, 120, "flat")}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "results", "breakout_ledger.csv")
    with open(path, "w") as fh_:
        fh_.write("date,dow,side,drift5_aligned," +
                  ",".join(f"fwd_{h}" for h in hz) + "\n")
        for i, r in enumerate(rs):
            fh_.write(f"{dt[i]:%Y-%m-%d},{dow[i]},{int(r['side'])},{int(al[i])}," +
                      ",".join(f"{hz[h][i]:.6f}" for h in hz) + "\n")
    print("\nwrote results/breakout_ledger.csv")
