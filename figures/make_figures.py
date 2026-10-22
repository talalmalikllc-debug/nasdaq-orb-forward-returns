#!/usr/bin/env python3
"""Build every figure in figures/ from results/breakout_ledger.csv.

Runs without the broker export: the ledger holds one row per breakout with
forward returns already normalised by ATR and net of the spread at the fill.

    python figures/make_figures.py
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a3a29c"
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
ITERS = 100_000
RNG = np.random.default_rng(20260813)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "legend.frameon": False,
})


def boot_ci(v, iters=10000):
    m = RNG.choice(v, (iters, len(v)), replace=True).mean(1)
    return np.percentile(m, [2.5, 97.5])


def tstat(v):
    return v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))


def best_single(y, dow):
    return max(y[dow == d].mean() - y[dow != d].mean() for d in range(1, 6))


def fig_weekday(df):
    y, dow = df["fwd_flat"].to_numpy(), df["dow"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for d in range(1, 6):
        v = y[dow == d]
        lo, hi = boot_ci(v)
        c = BLUE if d == 5 else GRAY
        ax.bar(d, v.mean(), width=0.6, color=c, zorder=2)
        ax.errorbar(d, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]],
                    color=INK, capsize=4, lw=1.2, zorder=3)
        ax.annotate(f"t = {tstat(v):+.2f}\nn = {len(v)}", (d, max(hi, 0) + 0.012),
                    ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.axhline(y.mean(), color=ORANGE, lw=1.5, ls="--", zorder=1)
    ax.text(5.45, y.mean(), f"all days\n{y.mean():+.3f}", color=INK2, fontsize=8.5, va="center")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(1, 6), DAYS)
    ax.set_xlim(0.4, 6.0)
    ax.set_ylabel("mean forward return to session flat (ATR, net of spread)")
    ax.set_ylim(top=0.25)
    ax.set_title("Breakout follow-through by weekday (95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_weekday_effect.png", dpi=180)
    plt.close(fig)


def fig_null(df):
    y, dow = df["fwd_flat"].to_numpy(), df["dow"].to_numpy()
    fri = dow == 5
    real = y[fri].mean() - y[~fri].mean()
    naive = np.empty(ITERS)
    sel = np.empty(ITERS)
    for i in range(ITERS):
        s = RNG.permutation(dow)
        naive[i] = y[s == 5].mean() - y[s != 5].mean()
        sel[i] = best_single(y, s)
    p_naive = (naive >= real).mean()
    fmt = lambda p: f"P < {1 / ITERS:.4f}" if p == 0 else f"P = {p:.4f}"
    p_sel = (sel >= real).mean()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bins = np.linspace(min(naive.min(), sel.min()), max(sel.max(), real) * 1.05, 70)
    ax.hist(naive, bins=bins, color=GRAY, alpha=0.75, label=f"naive null: Friday fixed in advance ({fmt(p_naive)})")
    ax.hist(sel, bins=bins, color=BLUE, alpha=0.65,
            label=f"selection-aware null: best of 5 days re-searched ({fmt(p_sel)})")
    ax.axvline(real, color=ORANGE, lw=2)
    ax.text(real, ax.get_ylim()[1] * 0.92, f"  observed Friday\n  {real:+.4f}", color=INK, fontsize=9, va="top")
    ax.set_xlabel("best-day minus rest, fwd_flat (ATR)")
    ax.set_ylabel(f"count of {ITERS:,} label permutations")
    ax.set_title("The search is part of the null: Friday still clears it")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_selection_aware_null.png", dpi=180)
    plt.close(fig)


def fig_stability(df):
    df = df.assign(year=pd.to_datetime(df["date"]).dt.year,
                   nth=(pd.to_datetime(df["date"]).dt.day - 1) // 7 + 1)
    fri = df[df.dow == 5]
    rest = df[df.dow != 5]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.0), gridspec_kw={"width_ratios": [1.25, 1]})
    yrs = sorted(df.year.unique())
    x = np.arange(len(yrs))
    a.bar(x - 0.18, [fri[fri.year == Y].fwd_flat.mean() for Y in yrs], 0.34, color=BLUE, label="Friday", zorder=2)
    a.bar(x + 0.18, [rest[rest.year == Y].fwd_flat.mean() for Y in yrs], 0.34, color=GRAY, label="Mon–Thu", zorder=2)
    a.axhline(0, color=INK2, lw=0.8)
    a.set_xticks(x, [f"{Y}\n(n={len(fri[fri.year == Y])})" for Y in yrs])
    a.set_ylabel("mean fwd_flat (ATR, net)")
    a.set_title("Same sign in every year")
    a.legend(loc="upper right")
    ks = [1, 2, 3, 4, 5]
    vals = [fri[fri.nth == k].fwd_flat for k in ks]
    b.bar(ks, [v.mean() for v in vals], 0.6, color=[ORANGE if k == 3 else BLUE for k in ks], zorder=2)
    for k, v in zip(ks, vals):
        b.annotate(f"t={tstat(v):+.2f}\nn={len(v)}", (k, v.mean() + 0.008), ha="center", fontsize=8, color=INK2)
    b.axhline(0, color=INK2, lw=0.8)
    b.set_xticks(ks, ["1st", "2nd", "3rd\n(OpEx)", "4th", "5th"])
    b.set_ylim(top=0.21)
    b.set_title("Not an option-expiry artefact")
    b.set_ylabel("mean fwd_flat, Fridays only (ATR, net)")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_stability.png", dpi=180)
    plt.close(fig)


def fig_horizons(df):
    hs = ["fwd_15", "fwd_30", "fwd_60", "fwd_120", "fwd_flat"]
    lab = ["15m", "30m", "60m", "120m", "flat"]
    fri = df.dow == 5
    d, lo, hi = [], [], []
    for h in hs:
        a, b = df.loc[fri, h].to_numpy(), df.loc[~fri, h].to_numpy()
        da = RNG.choice(a, (10000, len(a))).mean(1) - RNG.choice(b, (10000, len(b))).mean(1)
        d.append(a.mean() - b.mean())
        lo.append(np.percentile(da, 2.5))
        hi.append(np.percentile(da, 97.5))
    x = np.arange(len(hs))
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.fill_between(x, lo, hi, color=BLUE, alpha=0.18, lw=0)
    ax.plot(x, d, color=BLUE, lw=2, marker="o", ms=7)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, lab)
    ax.set_xlabel("holding horizon after the breakout fill")
    ax.set_ylabel("Friday minus Mon–Thu (ATR)")
    ax.set_title("The Friday gap builds with horizon (95% bootstrap band)")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_horizon_profile.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    df = pd.read_csv(ROOT / "results" / "breakout_ledger.csv")
    fig_weekday(df)
    fig_null(df)
    fig_stability(df)
    fig_horizons(df)
    print("wrote", sorted(p.name for p in FIG.glob("*.png")))
