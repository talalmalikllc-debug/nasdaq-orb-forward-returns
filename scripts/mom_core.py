"""Pre-breakout momentum: feature construction.

Returns over the 5 / 15 / 30 / 60 minutes before the breakout, scaled by the
prior day's ATR so a 60-point move means the same thing in a quiet January and
a violent April.

ANCHOR CHOICE IS THE WHOLE EXPERIMENT. "Before the breakout" has three
possible end points and they are not equivalent:

  OPEN  window ends at the range start, 16:30. Pure pre-session drift. Fixed
        long before any order exists, and completely independent of which side
        eventually triggers.
  PLACE window ends at the range end, 17:00 - the moment the EA computes the
        range and places the OCO pendings. Still strictly before entry, and
        this is the latest anchor an ENTRY_PENDING EA can legally read.
  FILL  window ends at the open of the bar the breakout filled in.
        MECHANICALLY CONTAMINATED and included only to prove it: price had to
        rise to reach a buy stop, so the last few minutes before a long fill
        are positive almost by construction. Any "edge" measured on it is the
        breakout predicting itself. It is reported, and then discarded.

The tradeable feature is therefore OPEN or PLACE. Both are computed from bar
closes at or before the anchor, never from the bar in progress.
"""
from __future__ import annotations
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("BROKER_DATA_DIR", _os.path.join(_REPO, "data", "raw"))
import sys
import numpy as np

sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import orb_core as oc            # noqa: E402

WINDOWS = (5, 15, 30, 60)        # minutes
BAR = 300                        # M5


def _close_at(bars, i0, i1, sod_target):
    """Close of the bar that ENDS at sod_target within [i0,i1). The M5 bar
    stamped 16:25 covers 16:25-16:30, so its close is the 16:30 price."""
    t = oc.sec_of_day(bars["t"][i0:i1])
    j = np.where(t == sod_target - BAR)[0]
    if len(j) == 0:
        return np.nan
    return bars["c"][i0 + j[0]]


def momentum(bars, ses, k, anchor_sod):
    """{window: return / ATR} ending at anchor_sod for session k."""
    i0, i1 = ses["i0"][k], ses["i1"][k]
    end = _close_at(bars, i0, i1, anchor_sod)
    if not np.isfinite(end):
        return None
    out = {}
    for w in WINDOWS:
        start = _close_at(bars, i0, i1, anchor_sod - w * 60)
        out[w] = (end - start) if np.isfinite(start) else np.nan
    return out


def build(t0=1640995200, t1=1782950400):
    """Every breakout with OPEN / PLACE / FILL momentum, forward returns and
    the trade-level outcomes already validated in the OR-width study."""
    sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    import r1_predictive as r1
    import quality_core as qc

    bars, ses, f, recs = r1.build()
    r_start = oc.RANGE_START_H * 3600 + oc.RANGE_START_M * 60
    r_end = r_start + oc.RANGE_MIN * 60
    flat = oc.FLAT_H * 3600 + oc.FLAT_M * 60

    out = []
    for rec in recs:
        k = rec["k"]
        if not (t0 <= rec["t"] < t1):
            continue
        atr = f["atr_pre"][k]
        if not np.isfinite(atr) or atr <= 0:
            continue
        side = rec["side"]

        mo = momentum(bars, ses, k, r_start)
        mp = momentum(bars, ses, k, r_end)
        fill_sod = int(oc.sec_of_day(bars["t"][rec["i_fill"]]))
        mf = momentum(bars, ses, k, fill_sod)
        if mo is None or mp is None or mf is None:
            continue

        d = dict(rec)
        d["atr"] = atr
        d["cost_R"] = rec["spread"] / rec["R"]
        d["reg_ok"] = r1.regime_ok(f, k, side > 0)
        d["orr"] = f["orr"][k]
        d["year"] = 1970 + (rec["t"] // 86400) // 365.2425

        for tag, m in (("open", mo), ("place", mp), ("fill", mf)):
            for w in WINDOWS:
                v = m[w] / atr if np.isfinite(m[w]) else np.nan
                d[f"m_{tag}_{w}"] = v                 # raw, signed by the market
                d[f"a_{tag}_{w}"] = side * v          # aligned with the trade

        # ---- forward returns from the fill, in ATR units, signed by side ----
        i0, i1 = ses["i0"][k], ses["i1"][k]
        for h in (15, 30, 60, 120):
            px = _close_at(bars, i0, i1, min(fill_sod + h * 60, flat))
            d[f"fwd{h}"] = (side * (px - rec["entry"]) / atr
                            if np.isfinite(px) else np.nan)
        d["fwd_flat"] = side * (bars["c"][rec["i_end"] - 1] - rec["entry"]) / atr

        # ---- trade-level outcomes under the live exit and a 1R bracket ----
        d["trail"] = qc.trail_from(bars, side, rec["entry"], rec["sl"], rec["R"],
                                   rec["i_fill"], rec["i_end"])[0] - d["cost_R"]
        d["br1R"] = qc.bracket_from(bars, side, rec["entry"], rec["sl"], rec["R"],
                                    rec["i_fill"], rec["i_end"], 1.0)[0] - d["cost_R"]
        out.append(d)
    return bars, ses, f, out
