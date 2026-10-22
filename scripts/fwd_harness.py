"""Forward-return harness for US100 ORB hypotheses.

ONE instrument, ONE metric, so results are comparable across hypotheses.

    forward return = side x (Close[t + h] - entry) / ATR_prior_day

Signed by the trade's direction, so positive always means "the breakout went
the way it was taken". Measured in ATR rather than in R because R is the
opening-range width, and normalising by it cancels any signal that is itself a
function of range width - that mistake cost a whole study on this desk.

Net of the actual per-bar spread at the fill. No stops, no targets, no trailing
- forward return is deliberately exit-agnostic, so a hypothesis lives or dies
on whether price went somewhere, not on how a particular bracket sliced it.

EVERY feature is computable strictly before the entry it grades. Each one
carries a note saying when it is known.

Each hypothesis gets the same five tests:
    quintiles   monotone structure across the feature's range
    rank corr   permutation p-value on the continuous version
    per-year    how many of 5 years share the pooled sign
    IS/OOS      2022-01..2024-07 vs 2024-07..2026-07
    placebo     feature shuffled across sessions
A hypothesis that only passes the first two is not evidence.
"""
from __future__ import annotations
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("BROKER_DATA_DIR", _os.path.join(_REPO, "data", "raw"))
import sys
import numpy as np

sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import orb_core as oc            # noqa: E402
import stats_util as su          # noqa: E402

T0, T1 = 1640995200, 1782950400
SPLIT = 1719792000
HOR = (15, 30, 60, 120)
CUTOFF_MIN = 60


def sessions_and_breakouts():
    """Every NY-session opening-range breakout with its forward returns."""
    oc.ENTRY_CUTOFF_MIN = CUTOFF_MIN
    bars = oc.load_m5()
    ses = oc.sessions(bars)
    o, h, l, c = oc.daily_ohlc(bars, ses)
    atr = oc.wilder_atr(h, l, c, 14)
    atr_pre = np.roll(atr, 1); atr_pre[0] = np.nan
    ph = np.roll(h, 1); ph[0] = np.nan          # prior day's high / low / close
    pl = np.roll(l, 1); pl[0] = np.nan
    pc = np.roll(c, 1); pc[0] = np.nan
    po = np.roll(o, 1); po[0] = np.nan

    sod = oc.sec_of_day(bars["t"])
    r_start = oc.RANGE_START_H * 3600 + oc.RANGE_START_M * 60
    r_end = r_start + oc.RANGE_MIN * 60
    cutoff = r_end + CUTOFF_MIN * 60
    flat = oc.FLAT_H * 3600 + oc.FLAT_M * 60
    buf = oc.BUFFER_PTS * oc.POINT

    rows = []
    for k in range(len(ses["day"])):
        a, z = ses["i0"][k], ses["i1"][k]
        A = atr_pre[k]
        if not np.isfinite(A) or A <= 0:
            continue
        s = sod[a:z]
        hi, lo = ses["or_hi"][k], ses["or_lo"][k]
        win = np.where((s >= r_end) & (s < cutoff))[0]
        fin = np.where(s < flat)[0]
        if len(win) == 0 or len(fin) == 0:
            continue
        i_end = a + fin[-1] + 1

        side, i_fill = 0, -1
        for j in win:
            up, dn = h[0] if False else bars["h"][a + j] >= hi + buf, bars["l"][a + j] <= lo - buf
            if up and dn:
                side = 1 if abs(bars["o"][a + j] - (hi + buf)) <= \
                            abs(bars["o"][a + j] - (lo - buf)) else -1
            elif up:
                side = 1
            elif dn:
                side = -1
            else:
                continue
            i_fill = a + j
            break
        if side == 0 or not (T0 <= int(bars["t"][i_fill]) < T1):
            continue

        entry = (hi + buf) if side > 0 else (lo - buf)
        d = dict(k=k, i0=a, i1=z, i_fill=i_fill, i_end=i_end, side=side,
                 entry=entry, atr=A, or_hi=hi, or_lo=lo, or_w=hi - lo,
                 t=int(bars["t"][i_fill]),
                 day_h=h[k], day_l=l[k], day_o=o[k],
                 pd_h=ph[k], pd_l=pl[k], pd_c=pc[k], pd_o=po[k],
                 cost=bars["spread"][i_fill] * oc.POINT / A,
                 year=1970 + (int(bars["t"][i_fill]) // 86400) // 365.2425)
        for m in HOR:
            tgt = bars["t"][i_fill] + m * 60
            j = np.searchsorted(bars["t"][:i_end], tgt, "right") - 1
            j = min(max(j, i_fill), i_end - 1)
            d[f"fwd{m}"] = side * (bars["c"][j] - entry) / A
        d["fwd_flat"] = side * (bars["c"][i_end - 1] - entry) / A
        rows.append(d)
    return bars, ses, rows


def col(rs, k):
    return np.array([r.get(k, np.nan) for r in rs], float)


def fwd(rs, h=60):
    return col(rs, f"fwd{h}") - col(rs, "cost")


# ------------------------------------------------------------------ tests --
def evaluate(rs, key, label, horizon=60, nb=5, placebo_iters=2000):
    """The same five tests for every hypothesis."""
    v = col(rs, key)
    m = np.isfinite(v)
    s = [r for r, ok in zip(rs, m) if ok]
    v = v[m]
    if len(s) < 100:
        print(f"\n### {label}\n  too few observations ({len(s)})")
        return None
    y = fwd(s, horizon)

    b, edges = su.bucket(v, nb)
    print(f"\n### {label}")
    print(f"  n={len(s)}   feature known: see note   horizon fwd{horizon}")
    print(f"  {'Q':<4}{'n':>6}{'median':>11}" +
          "".join(f"{'fwd'+str(x):>9}" for x in HOR) + f"{'flat':>9}{'t':>7}")
    qmeans = []
    for q in range(nb):
        g = [s[i] for i in np.where(b == q)[0]]
        if len(g) < 15:
            qmeans.append(np.nan); continue
        vy = fwd(g, horizon)
        st = su.summary(vy[np.isfinite(vy)])
        qmeans.append(st["avg"])
        print(f"  Q{q+1:<3}{len(g):>6}{np.median(col(g,key)):>+11.3f}" +
              "".join(f"{np.nanmean(fwd(g,x)):>+9.4f}" for x in HOR) +
              f"{np.nanmean(col(g,'fwd_flat'))-np.nanmean(col(g,'cost')):>+9.4f}"
              f"{st['t']:>7.2f}")

    hi_ = y[b == nb - 1]; lo_ = y[b == 0]
    hi_, lo_ = hi_[np.isfinite(hi_)], lo_[np.isfinite(lo_)]
    p, ci = su.boot_diff(hi_, lo_)
    spread = hi_.mean() - lo_.mean()
    rho, prho = su.spearman(v, y)

    yr = np.array([int(r["year"]) for r in s])
    signs = []
    for yy in sorted(set(yr)):
        mm = yr == yy
        if mm.sum() < 30:
            continue
        a_ = y[mm & (b == nb - 1)]; z_ = y[mm & (b == 0)]
        if len(a_) > 4 and len(z_) > 4:
            signs.append(np.sign(a_.mean() - z_.mean()))
    agree = int(sum(1 for x in signs if x == np.sign(spread)))

    isv, oosv = [], []
    ts = np.array([r["t"] for r in s])
    for cond, out in ((ts < SPLIT, isv), (ts >= SPLIT, oosv)):
        a_ = y[cond & (b == nb - 1)]; z_ = y[cond & (b == 0)]
        out.append(a_.mean() - z_.mean() if len(a_) > 10 and len(z_) > 10 else np.nan)

    null = np.empty(placebo_iters)
    for i in range(placebo_iters):
        bb, _ = su.bucket(su.RNG.permutation(v), nb)
        null[i] = y[bb == nb - 1].mean() - y[bb == 0].mean()
    pl = float((np.abs(null) >= abs(spread)).mean())

    print(f"  Q{nb}-Q1 {spread:+.4f} ATR  95% CI [{ci[0]:+.4f},{ci[1]:+.4f}]"
          f"  boot P={p:.3f}")
    print(f"  spearman {rho:+.4f} (perm p={prho:.3f})   placebo P={pl:.3f}"
          f"   years agreeing {agree}/{len(signs)}"
          f"   IS {isv[0]:+.4f}  OOS {oosv[0]:+.4f}")
    verdict = ("STRONG" if (pl < 0.05 and agree >= len(signs) - 1
                            and np.sign(isv[0]) == np.sign(oosv[0]) == np.sign(spread))
               else "weak" if pl < 0.15 else "none")
    print(f"  --> {verdict}")
    return dict(label=label, n=len(s), spread=spread, boot_p=p, rho=rho,
                rho_p=prho, placebo=pl, years=f"{agree}/{len(signs)}",
                IS=isv[0], OOS=oosv[0], verdict=verdict)


def summarise(results):
    rs = [r for r in results if r]
    rs.sort(key=lambda r: r["placebo"])
    print("\n" + "=" * 104)
    print("SUMMARY - ranked by placebo p (the hardest of the five tests)")
    print("=" * 104)
    print(f"  {'hypothesis':<44}{'n':>6}{'Q5-Q1':>9}{'placebo':>9}"
          f"{'rho p':>8}{'years':>8}{'IS':>9}{'OOS':>9}  verdict")
    print("  " + "-" * 100)
    for r in rs:
        print(f"  {r['label'][:43]:<44}{r['n']:>6}{r['spread']:>+9.4f}"
              f"{r['placebo']:>9.3f}{r['rho_p']:>8.3f}{r['years']:>8}"
              f"{r['IS']:>+9.4f}{r['OOS']:>+9.4f}  {r['verdict']}")
