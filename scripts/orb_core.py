"""Shared primitives for the OR-width / ATR study on US100.cash.

Everything here is deliberately built from ONE source (the M5 bar file MT5
exported) and from integer epoch arithmetic. Two reasons:

  * pandas 3 turns datetime columns into datetime64[s]; a previous study in
    this line silently lost day buckets that way. Sessions are therefore keyed
    by an integer server-date ordinal computed from the epoch, never by a
    datetime dtype.
  * The EA reads the opening range off M1 bars aggregated over the same
    window. Max/min over M5 bars covering [start, start+N) is identical to the
    M1 aggregate, so the M5 file reproduces the EA's range exactly while also
    covering 2021-09 onward (the M1 export only reaches back to 2024-04).

Timestamps in the export are BROKER/SERVER time (the instrument's daily break
sits at 23:15-01:00 in them), which is the same clock the EA's hour/minute
inputs use. No timezone conversion is applied or wanted.
"""
from __future__ import annotations
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("BROKER_DATA_DIR", _os.path.join(_REPO, "data", "raw"))

import numpy as np

M5 = _os.path.join(_DATA, "bars_US100.cash_M5.csv")
POINT = 0.01            # US100.cash, 2 digits
DAY = 86400

# live configuration of the EA (report: ORB SCALPER Newest Report.html)
RANGE_START_H, RANGE_START_M = 16, 30
RANGE_MIN = 30
ENTRY_CUTOFF_MIN = 30       # after range end
FLAT_H, FLAT_M = 22, 30
BUFFER_PTS = 10.0


# ---------------------------------------------------------------- data ----
def load_m5(t0=None, t1=None):
    """(t, o, h, l, c, vol, spread) arrays, real 5-minute bars only."""
    raw = np.loadtxt(M5, delimiter=",", skiprows=1,
                     usecols=(0, 1, 2, 3, 4, 5, 6))
    t = raw[:, 0].astype(np.int64)
    # the export's first months are synthesised daily/hourly bars; real M5
    # history starts 2021-09-14 (see gap census in the study notes)
    keep = t >= 1631600000
    if t0 is not None:
        keep &= t >= t0
    if t1 is not None:
        keep &= t < t1
    raw, t = raw[keep], t[keep]
    return dict(t=t, o=raw[:, 1], h=raw[:, 2], l=raw[:, 3], c=raw[:, 4],
                v=raw[:, 5], spread=raw[:, 6])


def sec_of_day(t):
    return t % DAY


def day_ord(t):
    """Integer server-date ordinal (days since epoch)."""
    return t // DAY


# ------------------------------------------------------------ sessions ----
def sessions(bars):
    """One record per server date that has a complete opening range.

    Returns a dict of arrays keyed by session, plus index slices so the
    per-day path work can stay vectorised where it matters.
    """
    t, h, l = bars["t"], bars["h"], bars["l"]
    d = day_ord(t)
    sod = sec_of_day(t)

    r_start = RANGE_START_H * 3600 + RANGE_START_M * 60
    r_end = r_start + RANGE_MIN * 60

    days = np.unique(d)
    # index bounds of every day
    starts = np.searchsorted(d, days, "left")
    ends = np.searchsorted(d, days, "right")

    out = []
    for k, day in enumerate(days):
        a, b = starts[k], ends[k]
        s = sod[a:b]
        m = (s >= r_start) & (s < r_end)
        n = int(m.sum())
        if n < RANGE_MIN // 5:          # incomplete range window -> no trade
            continue
        hi = h[a:b][m].max()
        lo = l[a:b][m].min()
        out.append((day, a, b, hi, lo, n))

    if not out:
        raise SystemExit("no sessions built")
    arr = np.array(out, dtype=np.float64)
    return dict(day=arr[:, 0].astype(np.int64), i0=arr[:, 1].astype(np.int64),
                i1=arr[:, 2].astype(np.int64), or_hi=arr[:, 3],
                or_lo=arr[:, 4], nbars=arr[:, 5].astype(np.int64))


def daily_ohlc(bars, ses):
    """Daily O/H/L/C per session day, built from the same M5 file."""
    n = len(ses["day"])
    o = np.empty(n); h = np.empty(n); l = np.empty(n); c = np.empty(n)
    for k in range(n):
        a, b = ses["i0"][k], ses["i1"][k]
        o[k] = bars["o"][a]
        h[k] = bars["h"][a:b].max()
        l[k] = bars["l"][a:b].min()
        c[k] = bars["c"][b - 1]
    return o, h, l, c


def wilder_atr(high, low, close, period=14):
    """Wilder ATR on daily bars. atr[k] uses bars 0..k; shift it yourself."""
    n = len(high)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    pc = close[:-1]
    tr[1:] = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - pc), np.abs(low[1:] - pc)))
    atr = np.full(n, np.nan)
    if n <= period:
        return atr
    atr[period] = tr[1:period + 1].mean()
    for k in range(period + 1, n):
        atr[k] = (atr[k - 1] * (period - 1) + tr[k]) / period
    return atr


def wilder_adx(high, low, close, period=14):
    """Wilder ADX on daily bars, same convention as MT5's iADX."""
    n = len(high)
    adx = np.full(n, np.nan)
    if n < 2 * period + 2:
        return adx
    up = high[1:] - high[:-1]
    dn = low[:-1] - low[1:]
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:] - close[:-1])))

    def smooth(x):
        s = np.full(len(x), np.nan)
        s[period - 1] = x[:period].sum()
        for k in range(period, len(x)):
            s[k] = s[k - 1] - s[k - 1] / period + x[k]
        return s

    str_, spdm, sndm = smooth(tr), smooth(pdm), smooth(ndm)
    with np.errstate(invalid="ignore", divide="ignore"):
        pdi = 100 * spdm / str_
        ndi = 100 * sndm / str_
        dx = 100 * np.abs(pdi - ndi) / (pdi + ndi)
    # ADX = Wilder average of DX
    a = np.full(len(dx), np.nan)
    first = period - 1 + period
    if first < len(dx):
        a[first] = np.nanmean(dx[period - 1:first + 1])
        for k in range(first + 1, len(dx)):
            a[k] = (a[k - 1] * (period - 1) + dx[k]) / period
    adx[1:] = a
    return adx


def trailing_rank(x, window, min_obs):
    """Percentile rank of x[k] inside x[k-window:k] (STRICTLY prior values).

    This is the only ranking an EA can actually compute live, so every
    deployable statement in the study uses it rather than a full-sample
    quantile.
    """
    n = len(x)
    out = np.full(n, np.nan)
    for k in range(n):
        a = max(0, k - window)
        prior = x[a:k]
        prior = prior[np.isfinite(prior)]
        if len(prior) < min_obs or not np.isfinite(x[k]):
            continue
        out[k] = 100.0 * (prior < x[k]).sum() / len(prior)
    return out


# ------------------------------------------------------------ breakout ----
def breakouts(bars, ses, allow_long=True, allow_short=True):
    """First OR breakout per session, with the EA's order mechanics.

    Buy stop at OR high + buffer, sell stop at OR low - buffer, both live from
    range end until range end + ENTRY_CUTOFF_MIN. SL is the opposite edge
    (SL_OPPOSITE_SIDE, the live setting). Path stats are then tracked to the
    flat time.

    Within-bar ambiguity is always resolved AGAINST the trade: if one bar
    touches both the stop and a target, the stop is taken first.
    """
    t, o, h, l, c, sp = (bars["t"], bars["o"], bars["h"], bars["l"],
                         bars["c"], bars["spread"])
    sod = sec_of_day(t)
    r_start = RANGE_START_H * 3600 + RANGE_START_M * 60
    r_end = r_start + RANGE_MIN * 60
    cutoff = r_end + ENTRY_CUTOFF_MIN * 60
    flat = FLAT_H * 3600 + FLAT_M * 60
    buf = BUFFER_PTS * POINT

    recs = []
    for k in range(len(ses["day"])):
        a, b = ses["i0"][k], ses["i1"][k]
        s = sod[a:b]
        hi, lo = ses["or_hi"][k], ses["or_lo"][k]
        buy_px, sell_px = hi + buf, lo - buf

        win = np.where((s >= r_end) & (s < cutoff))[0]
        if len(win) == 0:
            continue
        side = 0
        fill_i = -1
        for j in win:
            bh, bl, bo = h[a + j], l[a + j], o[a + j]
            hit_l = allow_long and bh >= buy_px
            hit_s = allow_short and bl <= sell_px
            if hit_l and hit_s:
                # both edges in one bar: give it to the edge the bar opened
                # nearer to, and flag it
                side = 1 if abs(bo - buy_px) <= abs(bo - sell_px) else -1
                fill_i = j
                break
            if hit_l:
                side, fill_i = 1, j
                break
            if hit_s:
                side, fill_i = -1, j
                break
        if side == 0:
            continue

        entry = buy_px if side > 0 else sell_px
        sl = (lo - buf) if side > 0 else (hi + buf)
        R = abs(entry - sl)
        if R <= 0:
            continue

        # path from the fill bar (inclusive) to the flat time
        path = np.where((s >= s[fill_i]) & (s < flat))[0]
        ph, pl = h[a + path], l[a + path]
        if side > 0:
            mfe = (ph.max() - entry) / R
            mae = (entry - pl.min()) / R
        else:
            mfe = (entry - pl.min()) / R
            mae = (ph.max() - entry) / R

        recs.append(dict(
            k=k, day=int(ses["day"][k]), t=int(t[a + fill_i]), side=side,
            entry=entry, sl=sl, R=R, or_hi=hi, or_lo=lo,
            or_w=hi - lo, mfe=mfe, mae=mae,
            spread=sp[a + fill_i] * POINT,
            i_fill=int(a + fill_i), i_end=int(a + path[-1]) + 1,
        ))
    return recs


def bracket_outcome(bars, rec, tp_R, sl_R=1.0):
    """Resolve a fixed bracket on the M5 path. Ties inside a bar go to the SL.

    Returns (outcome, bars_held): outcome +tp_R, -sl_R, or the marked-to-market
    result at the flat time. Timeouts are NEVER scored as zero -- a previous
    study in this line faked a positive edge that way.
    """
    side, entry, R = rec["side"], rec["entry"], rec["R"]
    tp = entry + side * tp_R * R
    st = entry - side * sl_R * R
    h, l, c = bars["h"], bars["l"], bars["c"]
    for i in range(rec["i_fill"], rec["i_end"]):
        if side > 0:
            if l[i] <= st:
                return -sl_R, i - rec["i_fill"]
            if h[i] >= tp:
                return tp_R, i - rec["i_fill"]
        else:
            if h[i] >= st:
                return -sl_R, i - rec["i_fill"]
            if l[i] <= tp:
                return tp_R, i - rec["i_fill"]
    last = c[rec["i_end"] - 1]
    return side * (last - entry) / R, rec["i_end"] - 1 - rec["i_fill"]
