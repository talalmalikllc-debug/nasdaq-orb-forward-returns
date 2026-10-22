# US100 ORB — forward-return predictive value

**Scope as requested:** US100.cash only. Forward return only — no strategy simulation, no
stops, no targets, no trailing, no EA. 1082 NY-session opening-range breakouts,
2022-01-01 → 2026-07-01, M5, signed by the trade's direction, in ATR units, net of the
per-bar spread at the fill.

**Baseline:** fwd60 +0.0192 ATR (t=+2.51), fwd_flat +0.0461 (t=+3.32).

Every candidate is graded the same way: bucket/binary split, bootstrap CI, **placebo**
(feature shuffled across sessions), **per-year sign agreement**, and **IS/OOS**
(2022-01→2024-07 vs 2024-07→2026-07). A candidate passes only if the placebo is under
0.05, at least 4 of 5 years agree, and both halves share the sign.

---

## VALIDATED

### 1. Friday — the strongest signal found

| day | n | fwd_flat | t |
|---|---|---|---|
| Monday | 201 | −0.0144 | −0.57 |
| Tuesday | 225 | +0.0717 | +2.63 |
| Wednesday | 219 | −0.0060 | −0.18 |
| Thursday | 220 | +0.0336 | +0.93 |
| **Friday** | 217 | **+0.1409** | **+4.58** |

Friday's forward drift is roughly **3× the all-day baseline**. It builds coherently with
horizon (fwd15 +0.019 → fwd30 +0.041 → fwd60 +0.041 → fwd120 +0.054 → flat +0.101 for
the Tue+Fri group), and it is stable across every year:

`2022 +0.1776 · 2023 +0.1115 · 2024 +0.1169 · 2025 +0.1569 · 2026 +0.1381`

**It survives the selection-aware placebo, which is the test that matters.** Picking the
best day out of five is a search, so the null must include the search: shuffle the weekday
labels, re-run the selection, repeat 100,000 times. Best-single-day from pure noise averages
+0.0451 (p95 +0.0806); the real +0.1185 gives **P = 0.0017** (Monte-Carlo s.e. 0.0001;
`scripts/p8_friday.py`). An earlier 5,000-draw run printed 0.0006, which was sampling noise
in the permutation count. The best *pair* (Tue+Fri) gives P = 0.0016 (5,000 draws, `p7`).

**It is not an option-expiry artefact** — the obvious confound:

| Friday of month | n | fwd_flat | t |
|---|---|---|---|
| 1st | 51 | +0.1692 | +2.80 |
| 2nd | 51 | +0.1376 | +1.73 |
| 3rd (monthly OpEx) | 47 | +0.1586 | +2.67 |
| 4th | 51 | +0.1275 | +2.23 |
| **excluding the 3rd** | **170** | **+0.1360** | **+3.81** |

Every Friday of the month works. Removing expiry Fridays barely moves it.

*Caveat worth keeping: there is no mechanism here, only a robust statistic. Weekday
effects are a well-known graveyard, and this was found by searching five days. It passed
the correct null, which is much better than most such findings manage — but a mechanism
would raise confidence a lot, and I do not have one.*

### 2. Pre-session drift (5-minute return to 16:30, aligned with the trade)

Confirmed here in pure forward-return terms, and the horizon profile is the finding:

| horizon | diff | placebo | years | IS | OOS |
|---|---|---|---|---|---|
| fwd15 | +0.0075 | 0.417 | 4/5 | +0.0098 | +0.0047 |
| fwd30 | +0.0151 | 0.208 | 3/5 | +0.0154 | +0.0145 |
| fwd60 | +0.0175 | 0.249 | 4/5 | +0.0165 | +0.0184 |
| **fwd120** | **+0.0474** | **0.010** | **5/5** | +0.0448 | +0.0505 |
| **fwd_flat** | **+0.0681** | **0.016** | **5/5** | +0.0773 | +0.0570 |

Monotone in horizon, significant only at long horizons, 5/5 years, IS ≈ OOS. That is the
signature of a slow drift, and it means **judging this signal at 60 minutes understates
it** — an earlier pass of mine did exactly that and made it look dead.

### 3. The interaction — the drift filter matters only on non-Fridays

| cell | n | fwd_flat | t |
|---|---|---|---|
| Friday, drift aligned | 114 | +0.1275 | +2.99 |
| Friday, drift against | 103 | +0.1557 | +3.50 |
| non-Friday, drift aligned | 449 | +0.0664 | +3.03 |
| **non-Friday, drift against** | **416** | **−0.0252** | **−1.17** |

Friday is good regardless of the drift. The drift filter earns its keep entirely on the
other four days, where it separates +0.0664 from −0.0252. The single worst group in the
sample — **38% of all breakouts** — is a non-Friday breakout taken against the
pre-session drift, and it is the only negative cell.

The two signals are independent: 52.5% of Fridays are drift-aligned vs 51.9% of
non-Fridays.

---

## NOT VALIDATED

Ten new features, none of which cleared the placebo:

| hypothesis | Q5−Q1 fwd60 | placebo | years |
|---|---|---|---|
| Relative volume in the opening range | +0.0230 | 0.348 | 2/5 |
| Overnight gap, aligned | −0.0214 | 0.371 | 2/5 |
| Prior day's return, aligned | −0.0055 | 0.830 | 3/5 |
| Prior day's close location, aligned | −0.0148 | 0.548 | 4/5 |
| Breakout beyond the prior day's high/low | +0.0204 | 0.400 | 3/5 |
| Overnight range / ATR | −0.0306 | 0.206 | 4/5 |
| Opening range / overnight range | +0.0103 | 0.659 | 3/5 |
| OR position in the overnight range | −0.0118 | 0.636 | 2/5 |
| Bars touching the edge before the break | — | n.s. | — |
| Monday (as a standalone claim) | −0.0313 | 0.111 | 5/5 |

Two near-misses worth naming:

- **Later break** (price still inside the range at 17:00): +0.0237 at fwd60 and the whole
  baseline edge sits in that group (t=+2.99 vs +0.79). But the placebo never drops below
  0.07 and **OOS collapses to −0.0024** against IS +0.0466. In-sample only.
- **Monday negative**: 5/5 years, both halves negative, and it has real literature behind
  it (the weekend effect) — but selection-aware **P = 0.077**. It fails its own search.

---

## Two bugs I hit, recorded so they don't recur

1. **Quantile bucketing breaks on discrete features.** Trigger delay and edge-touch counts
   have heavy ties, so `Q5−Q1` came back NaN and the placebo degenerated to a fake
   `P = 0.000`. Discrete features need explicit groups.
2. **`spearman` does not correct for ties.** On two binary flags it returned +0.355 where
   the true association is ~0. Any rank correlation on tied/binary data from this codebase
   is unreliable; use contingency counts instead.

---

## Where this leaves things

Only two signals in this instrument have survived proper testing: **Friday**, and the
**pre-session drift at long horizons**. They are independent, and they compose into one
clean statement about which breakouts are worth having — with the strongest practical
content being the *negative* cell, not the positive one.

Everything else tested across this project — OR width/ATR, VWAP in three forms, late ORB,
fading failed breakouts, the daily regime filter, breakout candle quality, and the
Gao et al. intraday-momentum paper — is falsified or was lookahead.
