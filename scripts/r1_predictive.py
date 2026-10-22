#!/usr/bin/env python3
"""R1 - how much predictive value is in ORRatio = OR width / pre-OR ATR?

Measured on the SAME instrument, window and mechanics the EA trades:
US100.cash, NY open 16:30 server, 30-minute opening range, stop orders at the
edges with a 10-point buffer, SL at the opposite edge, entry window 30 min
after the range, flat 22:30.

Per ORRatio quintile it reports WR, PF, avg R, MAE, MFE and the probability of
reaching a target before the stop. Everything is net of the actual per-bar
spread at the fill (a median spread understated cost 3x in an earlier study on
this desk), ties inside a bar are resolved against the trade, and unresolved
trades are marked to market at the flat time rather than scored as zero.
"""
from __future__ import annotations
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import orb_core as oc
import stats_util as su

T0, T1 = 1640995200, 1782950400          # 2022-01-01 .. 2026-07-01 (server)
OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def build():
    bars = oc.load_m5()                   # full history: ATR needs warm-up
    ses = oc.sessions(bars)
    o, h, l, c = oc.daily_ohlc(bars, ses)

    atr = oc.wilder_atr(h, l, c, 14)      # atr[k] includes day k
    atr_pre = np.roll(atr, 1); atr_pre[0] = np.nan          # strictly prior
    adx = oc.wilder_adx(h, l, c, 14)
    adx_pre = np.roll(adx, 1); adx_pre[0] = np.nan

    # EA's vol-regime percentile: rank of the newest CLOSED daily ATR in its
    # own prior 100 values
    volpct = oc.trailing_rank(atr_pre, 100, 10)

    orw = ses["or_hi"] - ses["or_lo"]
    with np.errstate(invalid="ignore", divide="ignore"):
        orr = orw / atr_pre

    recs = oc.breakouts(bars, ses)
    return bars, ses, dict(orw=orw, orr=orr, atr_pre=atr_pre, adx_pre=adx_pre,
                           volpct=volpct, dh=h, dl=l, dc=c), recs


def regime_ok(f, k, is_long):
    """The live report's permission sets, ADX source, D1 timeframe."""
    vp, adx = f["volpct"][k], f["adx_pre"][k]
    if not np.isfinite(vp) or not np.isfinite(adx):
        return False                       # EA cannot classify -> no trade
    vol = 0 if vp <= 30.0 else (2 if vp >= 70.0 else 1)
    trd = 2 if adx >= 25.0 else (1 if adx >= 18.0 else 0)
    if is_long:
        if vol == 0 or trd == 0:           # LongAllowVolLow/Choppy = false
            return False
        return True
    if vol != 0:                           # short allowed only in LOW vol
        return False
    return True


def rows(bars, recs, f, tp_list=(0.5, 1.0, 1.5)):
    out = []
    for r in recs:
        k = r["k"]
        if not (T0 <= r["t"] < T1):
            continue
        if not np.isfinite(f["orr"][k]):
            continue
        cost_R = r["spread"] / r["R"]      # one spread, round trip on a CFD
        d = dict(k=k, t=r["t"], side=r["side"], orr=f["orr"][k], entry=r["entry"],
                 orw=r["or_w"], atr=f["atr_pre"][k], R=r["R"],
                 mfe=r["mfe"], mae=r["mae"], cost_R=cost_R,
                 reg_ok=regime_ok(f, k, r["side"] > 0),
                 year=1970 + (r["t"] // 86400) // 365.2425)
        for tp in tp_list:
            o, held = oc.bracket_outcome(bars, r, tp, 1.0)
            d[f"r{tp}"] = o - cost_R
            d[f"hit{tp}"] = 1.0 if o >= tp - 1e-9 else 0.0
            d[f"held{tp}"] = held
        out.append(d)
    return out


def col(rs, name):
    return np.array([r[name] for r in rs], float)


def table(rs, key, label, tps=(0.5, 1.0)):
    orr = col(rs, "orr")
    b, edges = su.bucket(orr, 5)
    print(f"\n=== {label}  (n={len(rs)}) ===")
    print("quintile edges of OR/ATR: " + ", ".join(f"{e:.3f}" for e in edges))
    hdr = (f"{'Q':<4}{'n':>5}{'ORR med':>9}{'ORw':>7}" +
           "".join(f"{'WR'+str(t):>7}{'avgR'+str(t):>9}{'PF'+str(t):>7}"
                   for t in tps) +
           f"{'MFE':>7}{'MAE':>7}{'cost%R':>8}")
    print(hdr); print("-" * len(hdr))
    per_q = {}
    for q in range(5):
        m = b == q
        s = [rs[i] for i in np.where(m)[0]]
        line = f"Q{q+1:<3}{m.sum():>5}{np.median(orr[m]):>9.2f}{np.median(col(s,'orw')):>7.0f}"
        for t in tps:
            r = col(s, f"r{t}")
            st = su.summary(r)
            line += f"{st['wr']:>7.1f}{st['avg']:>9.3f}{st['pf']:>7.2f}"
        line += (f"{np.median(col(s,'mfe')):>7.2f}{np.median(col(s,'mae')):>7.2f}"
                 f"{100*np.mean(col(s,'cost_R')):>8.2f}")
        print(line)
        per_q[q] = s
    # TP-before-SL probabilities
    print(f"\n{'Q':<4}" + "".join(f"{'P(TP'+str(t)+' first)':>16}" for t in tps)
          + f"{'mean MFE':>10}{'mean MAE':>10}{'P(MFE>1R)':>11}")
    for q in range(5):
        s = per_q[q]
        line = f"Q{q+1:<3}"
        for t in tps:
            line += f"{100*col(s,f'hit{t}').mean():>16.1f}"
        line += (f"{col(s,'mfe').mean():>10.2f}{col(s,'mae').mean():>10.2f}"
                 f"{100*(col(s,'mfe')>=1).mean():>11.1f}")
        print(line)
    return per_q, edges


def significance(per_q, tps=(0.5, 1.0)):
    print("\n--- Q1 (narrowest OR) vs Q5 (widest) ---")
    for t in tps:
        a, bq = col(per_q[0], f"r{t}"), col(per_q[4], f"r{t}")
        p, ci = su.boot_diff(a, bq)
        print(f"  TP={t}R  mean Q1 {a.mean():+.3f}R  Q5 {bq.mean():+.3f}R  "
              f"diff {a.mean()-bq.mean():+.3f}R  95% CI [{ci[0]:+.3f},{ci[1]:+.3f}]  "
              f"P(diff<=0)={p:.3f}")
    print("\n--- rank correlation, continuous ORRatio vs outcome ---")
    allr = [r for q in per_q.values() for r in q]
    for t in tps:
        rho, p = su.spearman(col(allr, "orr"), col(allr, f"r{t}"))
        print(f"  TP={t}R  spearman {rho:+.4f}  perm p={p:.3f}")
    rho, p = su.spearman(col(allr, "orr"), col(allr, "mfe"))
    print(f"  MFE   spearman {rho:+.4f}  perm p={p:.3f}")
    rho, p = su.spearman(col(allr, "orr"), col(allr, "mae"))
    print(f"  MAE   spearman {rho:+.4f}  perm p={p:.3f}")


def per_year(rs, tp=1.0):
    """A pooled lift above every per-year lift is the tell for regime
    confounding, so the yearly table is not optional."""
    orr = col(rs, "orr")
    b, _ = su.bucket(orr, 5)
    yr = np.array([1970 + (r["t"] // 86400) / 365.2425 for r in rs], int)
    print(f"\n--- per-year avg R (TP={tp}R) by quintile ---")
    print(f"{'year':<6}" + "".join(f"{'Q'+str(q+1):>9}" for q in range(5))
          + f"{'Q1-Q5':>9}{'n':>6}")
    for y in sorted(set(yr)):
        line = f"{y:<6}"
        vals = []
        for q in range(5):
            m = (yr == y) & (b == q)
            v = col([rs[i] for i in np.where(m)[0]], f"r{tp}").mean() if m.sum() else np.nan
            vals.append(v)
            line += f"{v:>9.3f}"
        line += f"{vals[0]-vals[4]:>9.3f}{(yr==y).sum():>6}"
        print(line)


def placebo(rs, tp=1.0, iters=2000):
    """Shuffle ORRatio across sessions. Any quintile spread that survives is
    an artefact of the bucketing, not of the OR."""
    orr = col(rs, "orr"); r = col(rs, f"r{tp}")
    b, _ = su.bucket(orr, 5)
    real = r[b == 0].mean() - r[b == 4].mean()
    null = np.empty(iters)
    for i in range(iters):
        p = su.RNG.permutation(orr)
        bb, _ = su.bucket(p, 5)
        null[i] = r[bb == 0].mean() - r[bb == 4].mean()
    print(f"\n--- placebo (ORRatio shuffled across days), TP={tp}R ---")
    print(f"  real Q1-Q5 = {real:+.3f}R;  shuffled mean {null.mean():+.3f}R "
          f"sd {null.std():.3f};  P(|shuffled| >= |real|) = "
          f"{(np.abs(null) >= abs(real)).mean():.3f}")


if __name__ == "__main__":
    bars, ses, f, recs = build()
    rs = rows(bars, recs, f)
    print(f"sessions {len(ses['day'])}, breakouts in window {len(rs)}")
    n_reg = sum(1 for r in rs if r["reg_ok"])
    print(f"of those, allowed by the live regime filter: {n_reg}")

    per_q, edges = table(rs, "all", "ALL breakouts (regime filter OFF)")
    significance(per_q)
    per_year(rs)
    placebo(rs)

    rsf = [r for r in rs if r["reg_ok"]]
    per_qf, edges_f = table(rsf, "reg", "REGIME-FILTERED breakouts (what the EA actually trades)")
    significance(per_qf)
    per_year(rsf)

    json.dump(dict(edges_all=list(edges), edges_reg=list(edges_f),
                   n_all=len(rs), n_reg=len(rsf)),
              open(os.path.join(OUT, "r1_edges.json"), "w"), indent=1)
