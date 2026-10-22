"""Breakout-candle quality: features and exit resolution.

Data layer is shared with the OR-width study (same instrument, same session,
same opening range), so `orb_core` is imported from there rather than forked.

THE TIMING PROBLEM, stated once because everything here depends on it:

  The live EA runs ENTRY_PENDING. A buy stop at ORH+buffer fills the moment
  price touches it, part-way through some M5 bar. The close of that bar - and
  therefore its body, its wick and where it closed relative to its range - is
  not knowable until AFTER the position is already open. Screening entries on
  it is lookahead, full stop.

  So quality is only usable two ways, and both are implemented here:

    ABORT  the fill happens as it does today; at the close of the fill bar the
           position is either kept or dumped at that close. Costs a second
           spread on every abort.
    CLOSE  ENTRY_CANDLE_CLOSE mode (the EA already has it): wait for an M5 bar
           to CLOSE beyond the range, screen it, and only then enter at market.
           Legal by construction, but the entry is later and worse, and R
           changes because the stop stays at the opposite edge.

  A third frame, DESCRIPTIVE, measures quality against the outcome of the trade
  as it is filled today. It answers "is quality correlated with what follows"
  and is NOT tradeable. It is reported first, and labelled, because a strong
  descriptive number with a dead abort test is the signature of this trap.
"""
from __future__ import annotations
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("BROKER_DATA_DIR", _os.path.join(_REPO, "data", "raw"))
import sys
import numpy as np

sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import orb_core as oc            # noqa: E402

# live chandelier settings (InpChandTF=M1 live; M5 here, see r4 in the OR study)
ARM_R = 0.4
BE_LOCK_PTS = 5.0
CHAND_MULT = 1.5
CHAND_PERIOD = 10
CHAND_ATR = 14


# ------------------------------------------------------------- features ----
def quality(bars, rec):
    """Quality of the bar the breakout filled in, from that bar's own OHLC.

    All three are orientation-corrected so 'high is good' for both sides.
    """
    i = rec["i_fill"]
    o, h, l, c = bars["o"][i], bars["h"][i], bars["l"][i], bars["c"][i]
    rng = h - l
    side = rec["side"]
    if rng <= 0:
        return None
    # where in its own range did it close (Close Location Value)
    clv = (c - l) / rng if side > 0 else (h - c) / rng
    # how far beyond the range edge did it close, in opening-range widths
    orw = rec["or_w"]
    disp = ((c - rec["or_hi"]) if side > 0 else (rec["or_lo"] - c)) / orw
    # body share of the bar, and the wick that argues against the trade
    body = abs(c - o) / rng
    adverse_wick = ((h - max(o, c)) if side > 0 else (min(o, c) - l)) / rng
    return dict(clv=clv, disp=disp, body=body, wick=adverse_wick,
                bar_range_R=rng / rec["R"])


# --------------------------------------------------------------- exits ----
def _m5_atr(bars, i, period=CHAND_ATR):
    h, l, c = bars["h"], bars["l"], bars["c"]
    a = max(1, i - period)
    if i <= a:
        return np.nan
    tr = np.maximum(h[a:i] - l[a:i],
                    np.maximum(np.abs(h[a:i] - c[a - 1:i - 1]),
                               np.abs(l[a:i] - c[a - 1:i - 1])))
    return tr.mean() if len(tr) else np.nan


def trail_from(bars, side, entry, sl0, R, i_start, i_end, armed=False):
    """The EA's exit, resolved bar by bar from i_start. Ties inside a bar go
    against the trade. Never scores a timeout as zero - it marks to market."""
    h, l, c = bars["h"], bars["l"], bars["c"]
    sl = sl0
    be = entry + side * BE_LOCK_PTS * oc.POINT
    for i in range(i_start, i_end):
        if side > 0 and l[i] <= sl:
            return (sl - entry) / R, i
        if side < 0 and h[i] >= sl:
            return (entry - sl) / R, i
        if not armed and side * (c[i] - entry) / R >= ARM_R:
            armed = True
        if armed:
            a = max(0, i - CHAND_PERIOD + 1)
            atr = _m5_atr(bars, i + 1)
            if np.isfinite(atr) and atr > 0:
                if side > 0:
                    sl = max(sl, h[a:i + 1].max() - CHAND_MULT * atr, be)
                else:
                    sl = min(sl, l[a:i + 1].min() + CHAND_MULT * atr, be)
    return side * (c[i_end - 1] - entry) / R, i_end - 1


def bracket_from(bars, side, entry, sl0, R, i_start, i_end, tp_R):
    h, l, c = bars["h"], bars["l"], bars["c"]
    tp = entry + side * tp_R * R
    for i in range(i_start, i_end):
        if side > 0:
            if l[i] <= sl0:
                return (sl0 - entry) / R, i
            if h[i] >= tp:
                return tp_R, i
        else:
            if h[i] >= sl0:
                return (entry - sl0) / R, i
            if l[i] <= tp:
                return tp_R, i
    return side * (c[i_end - 1] - entry) / R, i_end - 1


def died_in_fill_bar(bars, rec):
    """Did the stop go inside the very bar that filled? Then there is no
    decision to make at that bar's close - the trade is already over."""
    i, side = rec["i_fill"], rec["side"]
    if side > 0:
        return bars["l"][i] <= rec["sl"]
    return bars["h"][i] >= rec["sl"]


def build(t0=1640995200, t1=1782950400):
    """Every breakout in the window, with quality features and both exits.

    The session/breakout/regime layer comes straight from the OR-width study so
    the two studies are talking about literally the same trades - that set was
    already matched against the EA's own fills (92.1% on day and side).
    """
    import r1_predictive as r1
    bars, ses, f, recs = r1.build()
    out = []
    for rec in recs:
        if not (t0 <= rec["t"] < t1):
            continue
        q = quality(bars, rec)
        if q is None:
            continue
        cost = rec["spread"] / rec["R"]
        d = dict(rec)
        d.update(q)
        d["cost_R"] = cost
        d["dead"] = died_in_fill_bar(bars, rec)
        d["year"] = 1970 + (rec["t"] // 86400) // 365.2425
        d["reg_ok"] = r1.regime_ok(f, rec["k"], rec["side"] > 0)
        d["orr"] = f["orr"][rec["k"]]

        # held to the normal exit, from the fill (what happens today)
        d["hold_trail"] = trail_from(bars, rec["side"], rec["entry"], rec["sl"],
                                     rec["R"], rec["i_fill"], rec["i_end"])[0] - cost
        d["hold_1R"] = bracket_from(bars, rec["side"], rec["entry"], rec["sl"],
                                    rec["R"], rec["i_fill"], rec["i_end"], 1.0)[0] - cost
        # dumped at the close of the fill bar (pays a second spread)
        i = rec["i_fill"]
        d["abort_R"] = (rec["side"] * (bars["c"][i] - rec["entry"]) / rec["R"]
                        - 2 * cost)
        if d["dead"]:
            d["abort_R"] = -1.0 - cost      # already stopped; no choice existed
        out.append(d)
    return bars, ses, out
