# Results — Study 545 (IPO-Birthday): the anniversary 'birthday effect' as an event study

*Generated from [`ipo_birthday/`](../ipo_birthday/) over this study's cached yfinance tape: daily
adjusted close for a curated **30-name US IPO basket** + SPY (prices fingerprint `19fee811b30d`,
1997-01-02 → 2026-06-26), the curated **IPO-date table** (fingerprint `d7888035240a`), and the
resulting **firm-year event CARs** (fingerprint `73d3051eb6f4`). Event window `[-5,+5]` trading days
around each IPO-date anniversary; abnormal return = name − SPY (beta-1 market model). As-of
**2026-06-30**.*

## The verdict, earned — Signal `NONE` · Tradability `MIRAGE`

The folklore: a listed company gets an **attention bump** around the anniversary of its IPO each
year, so its stock should show a small positive cumulative abnormal return (CAR) in a window
straddling that date. We test it as a classic **event study**: for every firm-year, take the IPO
anniversary, cumulate market-adjusted returns over the `[-5,+5]` trading-day window, and ask whether
the mean CAR across all events is positive and significant.

**It is not.** Across **312 firm-year events (29 names)** the mean anniversary CAR is **+62.75 bps**
with a one-sample **_t_ = 1.36** — below the *t* ≥ 2 bar. The **median** CAR is a trivial **+12.0
bps** and only **50.6%** of events are positive (a coin flip). Decisively, the **random-anchor
placebo** puts the null mean CAR at **+86.2 bps** (sd 68.0) — *higher* than the observed anniversary
CAR — so the placebo two-sided **_p_ = 0.648**: a random 11-day window on these same names does at
least as well as the "birthday" window. The small positive CAR is not a calendar effect at all; it
is the generic outperformance of a **survivor basket of high-flying growth IPOs** showing up in *any*
window. `NONE` on the signal axis; `MIRAGE` on tradability.

## Data stamp

- **Prices**: 30 curated US IPOs + SPY, daily adjusted close, 1997-01-02 → 2026-06-26, fingerprint
  `19fee811b30d`. (LNKD returns no yfinance history and contributes no events → 29 active names.)
- **IPO-date table**: 30 names with IPO (first-trade) dates from public listing records, fingerprint
  `d7888035240a`
- **Event CARs** (`[-5,+5]`, 312 firm-year events): fingerprint `73d3051eb6f4`

## The headline event study — the birthday CAR is not there

| | value |
|---|---|
| Events (firm-years) | **312** across **29** names |
| Mean anniversary CAR `[-5,+5]` | **+62.75 bps** |
| One-sample *t* (H0: mean CAR = 0) | **+1.36** |
| Median CAR | **+12.0 bps** |
| Share of events positive | **50.6%** |

A positive-but-insignificant mean, a near-zero median, and a coin-flip hit rate — the classic
signature of *no* effect plus a right-skewed survivor basket.

## The random-anchor placebo — the effect is just being one of these names

| | value |
|---|---|
| Observed anniversary CAR | **+62.75 bps** |
| Placebo null mean CAR (random anchors) | **+86.23 bps** |
| Placebo null sd | **68.0 bps** |
| Two-sided placebo *p* | **0.648** |

The placebo re-runs the identical window machinery on **random calendar dates that are not IPO
anniversaries**, matched to each name's event count. Its mean is *higher* than the anniversary CAR:
these growth-stock survivors beat SPY on average over 1997-2026, so *any* 11-day window shows a
positive market-adjusted return. The anniversary window is not special — it is *below* the random
null. `p = 0.648` says the birthday effect is indistinguishable from window noise on a rising basket.

## Robustness — the window sweep

| Event window (trading days) | Events | Mean CAR (bps) | *t* |
|---|---|---|---|
| `[-1,+1]` | 313 | **+3.7** | +0.15 |
| `[-3,+3]` | 313 | **+66.0** | +1.65 |
| `[-5,+5]` (headline) | 312 | **+62.7** | +1.36 |
| `[-10,+10]` | 312 | **+129.0** | +1.99 |

The CAR *grows with window width* — the tell that it is generic drift, not a spike on the
anniversary. The tightest window that actually brackets the date (`[-1,+1]`) is **+3.7 bps, _t_
0.15** — nothing. Only the widest 21-day window nudges *t* to +1.99, and its placebo (`[-3,+3]`
placebo *p* = 0.42) confirms random windows do the same. No window delivers a *t* ≥ 2 that survives
the placebo.

## Costs

| | value |
|---|---|
| Gross mean CAR `[-5,+5]` | **+62.75 bps** |
| Net (round trip, 5 bps/one-way × 2) | **+52.75 bps** |

Costs are a footnote: the effect is not statistically distinguishable from zero (or from a random
window) before you pay anything. The trade is long-only (no borrow); a short-the-anniversary variant
would additionally pay borrow and is not worth pricing.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted `bump_bps` (total over window) | Mean CAR (bps, 25 seeds) | Mean *t* (25 seeds) |
|---|---|---|
| 0 (null) | **−2.4** | **−0.04** |
| 40 | +37.6 | +0.57 |
| 120 | +117.6 | +1.78 |
| 200 | **+197.6** | **+2.99** |

At the null the mean CAR ≈ 0 and *t* ≈ 0 — no false positive. Plant a genuine anniversary bump and
the engine recovers it almost exactly in bps and drives *t* past 2 as it grows. So the detector
works; the flat real-tape result is the **tape talking, not a broken engine**. (Control only; never
cited for the real-tape stamp.)

## Why there is no birthday effect here

1. **No mechanism at the daily-return level.** Even if IPO anniversaries draw press retrospectives,
   there is no reason a *predictable, calendar-fixed* date should carry abnormal return in an even
   modestly efficient market — the attention is anticipated and already priced.
2. **Survivor basket, right-skewed.** The basket is names *still trading in 2026* — winners like
   AMZN, NVDA-adjacent growth, MA/V. They beat SPY on average, so *every* window (including random
   placebo anchors) shows a positive CAR. Naming a window "the birthday" does not make its return
   special: the placebo null mean (+86 bps) exceeds the observed anniversary CAR (+63 bps).
3. **The CAR scales with window width**, the fingerprint of generic drift rather than an
   anniversary spike; the tight `[-1,+1]` window is +3.7 bps (*t* 0.15).

## The honest takeaway

The IPO 'birthday effect' does not appear on this tape. The mean `[-5,+5]` anniversary CAR is
+62.75 bps at *t* 1.36 — below the bar — the median is +12 bps, half the events are negative, and a
random-anchor placebo (null mean +86 bps, *p* 0.648) shows the small positive number is nothing but
the generic outperformance of a survivor basket of growth IPOs, visible in any window. `NONE` ×
`MIRAGE`. The synthetic control confirms the engine would catch a real anniversary bump (recovering
a planted 200 bps at *t* 2.99), so this is a genuine absence, not a detector failure.
