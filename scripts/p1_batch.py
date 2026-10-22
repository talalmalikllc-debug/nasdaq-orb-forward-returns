#!/usr/bin/env python3
"""P1 - a batch of ORB hypotheses on US100, graded purely by forward return.

Ten features, none of them previously tested on this desk, each computable
strictly before the breakout it grades. Directional features are aligned with
the trade (side x feature) so "high" always means "the feature agrees with the
trade"; magnitude features are left unsigned.

Every one gets the identical five-test treatment from fwd_harness, so the
output is a ranked ledger rather than ten separate arguments.
"""
from __future__ import annotations
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("BROKER_DATA_DIR", _os.path.join(_REPO, "data", "raw"))
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import orb_core as oc
import fwd_harness as fh

OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def add_features(bars, ses, rs):
    sod = oc.sec_of_day(bars["t"])
    r_start = oc.RANGE_START_H * 3600 + oc.RANGE_START_M * 60
    r_end = r_start + oc.RANGE_MIN * 60

    # relative volume needs a per-session history of the OR window's volume
    orvol = {}
    for r in rs:
        a, z = r["i0"], r["i1"]
        s = sod[a:z]
        m = (s >= r_start) & (s < r_end)
        orvol[r["k"]] = bars["v"][a:z][m].sum() if m.any() else np.nan
    ks = sorted(orvol)
    hist = {}
    for idx, k in enumerate(ks):
        prior = [orvol[j] for j in ks[max(0, idx - 20):idx]
                 if np.isfinite(orvol[j])]
        hist[k] = np.median(prior) if len(prior) >= 10 else np.nan

    for r in rs:
        a, z, k, side, A = r["i0"], r["i1"], r["k"], r["side"], r["atr"]
        s = sod[a:z]

        # --- 1. relative volume in the opening range (known 17:00) ----------
        hv = hist.get(k, np.nan)
        r["relvol"] = orvol[k] / hv if (np.isfinite(hv) and hv > 0) else np.nan

        # --- 2. overnight gap, aligned (known 16:30) ------------------------
        # the session's own open vs the prior day's close
        r["gap"] = side * (r["day_o"] - r["pd_c"]) / A if np.isfinite(r["pd_c"]) else np.nan

        # --- 3. prior day's return, aligned (known 16:30) -------------------
        r["pdret"] = (side * (r["pd_c"] - r["pd_o"]) / A
                      if np.isfinite(r["pd_o"]) else np.nan)

        # --- 4. prior day's close location (known 16:30) --------------------
        rngp = r["pd_h"] - r["pd_l"]
        if np.isfinite(rngp) and rngp > 0:
            clv = (r["pd_c"] - r["pd_l"]) / rngp
            r["pdclv"] = clv if side > 0 else (1 - clv)
        else:
            r["pdclv"] = np.nan

        # --- 5. is the breakout level beyond the prior day's range? ---------
        if np.isfinite(r["pd_h"]) and np.isfinite(r["pd_l"]):
            r["beyond_pd"] = (side * (r["entry"] - (r["pd_h"] if side > 0
                                                    else r["pd_l"])) / A)
        else:
            r["beyond_pd"] = np.nan

        # --- 6. minutes from range end to the trigger (known at the break) --
        r["trig_min"] = (sod[r["i_fill"]] - r_end) / 60.0

        # --- 7. how many bars touched the broken edge before it broke -------
        m = (s >= r_end) & (np.arange(z - a) < (r["i_fill"] - a))
        if m.any():
            edge = r["or_hi"] if side > 0 else r["or_lo"]
            if side > 0:
                r["touches"] = int((bars["h"][a:z][m] >= edge - 0.15 * (r["or_w"])).sum())
            else:
                r["touches"] = int((bars["l"][a:z][m] <= edge + 0.15 * (r["or_w"])).sum())
        else:
            r["touches"] = 0

        # --- 8. overnight (pre-NY) range vs the opening range ---------------
        m2 = (s >= 1 * 3600) & (s < r_start)
        if m2.sum() > 12:
            onr = bars["h"][a:z][m2].max() - bars["l"][a:z][m2].min()
            r["on_range"] = onr / A
            r["or_vs_on"] = r["or_w"] / onr if onr > 0 else np.nan
        else:
            r["on_range"] = r["or_vs_on"] = np.nan

        # --- 9. where the OR sits inside the overnight range, aligned -------
        if m2.sum() > 12:
            on_h = bars["h"][a:z][m2].max(); on_l = bars["l"][a:z][m2].min()
            if on_h > on_l:
                pos = ((r["or_hi"] + r["or_lo"]) / 2 - on_l) / (on_h - on_l)
                r["or_pos"] = pos if side > 0 else (1 - pos)
            else:
                r["or_pos"] = np.nan
        else:
            r["or_pos"] = np.nan

        # --- 10. day of week (0=Mon) ---------------------------------------
        r["dow"] = float(((r["t"] // 86400) + 4) % 7)
    return rs


FEATURES = [
    ("relvol", "1. Relative volume in the opening range (vs prior 20 sessions)"),
    ("gap", "2. Overnight gap, aligned with the trade"),
    ("pdret", "3. Prior day's return, aligned"),
    ("pdclv", "4. Prior day's close location, aligned"),
    ("beyond_pd", "5. Breakout level beyond the prior day's high/low"),
    ("trig_min", "6. Minutes from range end to the trigger"),
    ("touches", "7. Bars that touched the edge before it broke"),
    ("on_range", "8. Overnight range / ATR"),
    ("or_vs_on", "9. Opening range / overnight range"),
    ("or_pos", "10. Where the OR sits in the overnight range, aligned"),
]

NOTES = {
    "relvol": "known at 17:00, from the range window's own volume",
    "gap": "known at 16:30", "pdret": "known at 16:30",
    "pdclv": "known at 16:30", "beyond_pd": "known at 17:00",
    "trig_min": "known at the trigger itself",
    "touches": "known at the trigger", "on_range": "known at 16:30",
    "or_vs_on": "known at 17:00", "or_pos": "known at 17:00",
}

if __name__ == "__main__":
    bars, ses, rs = fh.sessions_and_breakouts()
    rs = add_features(bars, ses, rs)
    y = fh.fwd(rs, 60)
    print(f"US100.cash breakouts {len(rs)}  (NY 16:30 +30min OR, 60-min entry window)")
    print(f"baseline forward return fwd60 = {np.nanmean(y):+.4f} ATR "
          f"(t={fh.su.summary(y[np.isfinite(y)])['t']:+.2f})")
    print("all figures net of the per-bar spread at the fill; signed by the trade\n")

    res = []
    for key, lab in FEATURES:
        r = fh.evaluate(rs, key, f"{lab}   [{NOTES[key]}]")
        res.append(r)

    # day of week gets its own treatment - it is categorical
    print("\n### 11. Day of week   [known in advance]")
    names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    print(f"  {'day':<6}{'n':>6}" + "".join(f"{'fwd'+str(x):>9}" for x in fh.HOR)
          + f"{'t@60':>7}")
    for i, nm in enumerate(names):
        g = [r for r in rs if int(r["dow"]) == i]
        if len(g) < 30:
            continue
        v = fh.fwd(g, 60); v = v[np.isfinite(v)]
        st = fh.su.summary(v)
        print(f"  {nm:<6}{len(g):>6}" +
              "".join(f"{np.nanmean(fh.fwd(g,x)):>+9.4f}" for x in fh.HOR)
              + f"{st['t']:>7.2f}")

    fh.summarise(res)
    json.dump([r for r in res if r], open(os.path.join(OUT, "p1.json"), "w"),
              indent=1, default=float)
