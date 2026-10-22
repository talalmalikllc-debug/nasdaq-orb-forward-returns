#!/usr/bin/env python3
"""P3 - validation of the three candidates that survived P1/P2.

    LATER-BREAK   price was still inside the range at 17:00
    FRIDAY        fwd60 +0.0516, t=2.43
    TUESDAY       fwd60 +0.0360, t=2.24

Day of week is five simultaneous looks with no prior reason to favour any of
them, so two hits at t > 2 is roughly what noise produces. The only thing that
separates a real weekday effect from that is out-of-sample persistence and
per-year consistency, so those are what decide it here - not the t-statistic
that put the candidate on the list in the first place.

Each candidate is graded as a binary group against the rest of the sample, with
the group label shuffled for the placebo.
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


def grade(rs, sel, name, horizon=60):
    y = fh.fwd(rs, horizon)
    lab = np.array([1 if sel(r) else 0 for r in rs])
    m = np.isfinite(y)
    y, lab = y[m], lab[m]
    sub = [r for r, ok in zip(rs, m) if ok]
    a, b = y[lab == 1], y[lab == 0]
    if len(a) < 40:
        print(f"\n### {name}: too few ({len(a)})")
        return None
    real = a.mean() - b.mean()
    p, ci = su.boot_diff(a, b)

    null = np.empty(4000)
    for i in range(4000):
        s = su.RNG.permutation(lab)
        null[i] = y[s == 1].mean() - y[s == 0].mean()
    pl = float((np.abs(null) >= abs(real)).mean())

    yr = np.array([int(r["year"]) for r in sub])
    ts = np.array([r["t"] for r in sub])
    agree, tot, per = 0, 0, []
    for yy in sorted(set(yr)):
        sa, sb = y[(yr == yy) & (lab == 1)], y[(yr == yy) & (lab == 0)]
        if len(sa) < 8 or len(sb) < 8:
            continue
        tot += 1
        d = sa.mean() - sb.mean()
        per.append((yy, len(sa), d))
        if np.sign(d) == np.sign(real):
            agree += 1
    iso = []
    for cond in (ts < SPLIT, ts >= SPLIT):
        sa, sb = y[cond & (lab == 1)], y[cond & (lab == 0)]
        iso.append(sa.mean() - sb.mean() if len(sa) > 8 and len(sb) > 8 else np.nan)

    st = su.summary(a)
    print(f"\n### {name}")
    print(f"  in-group n={len(a)}  fwd{horizon} {a.mean():+.4f} (t={st['t']:+.2f})"
          f"   rest n={len(b)} {b.mean():+.4f}")
    print(f"  difference {real:+.4f} ATR  95% CI [{ci[0]:+.4f},{ci[1]:+.4f}]"
          f"  boot P={p:.3f}   placebo P={pl:.3f}")
    print("  per-year: " + "  ".join(f"{y_}:{d:+.4f}(n={n_})" for y_, n_, d in per))
    print(f"  years agreeing {agree}/{tot}    IS {iso[0]:+.4f}   OOS {iso[1]:+.4f}")
    ok = (pl < 0.05 and agree >= tot - 1
          and np.isfinite(iso[0]) and np.isfinite(iso[1])
          and np.sign(iso[0]) == np.sign(iso[1]) == np.sign(real))
    print(f"  --> {'VALIDATED' if ok else 'NOT validated'}"
          + ("" if ok else "  (needs placebo<0.05, >=4/5 years, and both halves same sign)"))
    return dict(name=name, n=len(a), diff=real, placebo=pl,
                years=f"{agree}/{tot}", IS=iso[0], OOS=iso[1], ok=ok)


if __name__ == "__main__":
    bars, ses, rs = fh.sessions_and_breakouts()
    rs = add_features(bars, ses, rs)
    sod = oc.sec_of_day(bars["t"])
    r_end = (oc.RANGE_START_H * 3600 + oc.RANGE_START_M * 60 + oc.RANGE_MIN * 60)
    for r in rs:
        r["immediate"] = bool(sod[r["i_fill"]] == r_end)

    y = fh.fwd(rs, 60)
    print(f"US100.cash breakouts {len(rs)}   baseline fwd60 {np.nanmean(y):+.4f} ATR"
          f" (t={su.summary(y[np.isfinite(y)])['t']:+.2f})")
    print("candidates are graded against the REST of the sample, not against zero")

    out = []
    out.append(grade(rs, lambda r: not r["immediate"],
                     "LATER BREAK (price still inside the range at 17:00)"))
    out.append(grade(rs, lambda r: int(r["dow"]) == 5, "FRIDAY"))
    out.append(grade(rs, lambda r: int(r["dow"]) == 2, "TUESDAY"))
    out.append(grade(rs, lambda r: int(r["dow"]) in (2, 5), "TUESDAY or FRIDAY"))
    out.append(grade(rs, lambda r: int(r["dow"]) == 1, "MONDAY (the weak one)"))

    print("\n" + "=" * 96)
    print("LEDGER - forward-return predictive value on US100, fwd60")
    print("=" * 96)
    print(f"  {'candidate':<46}{'n':>6}{'diff':>10}{'placebo':>9}"
          f"{'years':>8}{'IS':>10}{'OOS':>10}")
    for r in out:
        if r:
            print(f"  {r['name'][:45]:<46}{r['n']:>6}{r['diff']:>+10.4f}"
                  f"{r['placebo']:>9.3f}{r['years']:>8}{r['IS']:>+10.4f}"
                  f"{r['OOS']:>+10.4f}  {'VALIDATED' if r['ok'] else 'no'}")
