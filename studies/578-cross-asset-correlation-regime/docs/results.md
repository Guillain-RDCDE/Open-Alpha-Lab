# Results — Study 578 (Cross-Asset-Correlation-Regime): is rising co-movement a fragility gauge?

*Generated from [`cross_asset_correlation_regime/`](../cross_asset_correlation_regime/) over this
study's cached yfinance tape: daily adjusted-close returns for a **14-ETF cross-asset panel**
(SPY, QQQ, IWM, EFA, EEM, TLT, IEF, LQD, HYG, GLD, DBC, USO, VNQ, SLV), **2007-01-04 → 2026-06-26**,
returns fingerprint `9dbfa917d6ba`. The correlation index is the trailing-63-day mean off-diagonal
pairwise correlation; the regime is HIGH when it exceeds its expanding 70th percentile (past data
only). Forward returns/vol are the risk asset SPY's next 21 trading days. Aligned regime/forward
frame fingerprint `0b73f61632f6`. As-of **2026-06-30** (the final ~21 days carry no complete
forward window and are dropped by construction).*

## The verdict, earned — Signal `NONE` · Tradability `MIRAGE` · "Fragility gauge?" `COINCIDENT`

The folklore: *when everything starts moving together, the market is about to break* — a high /
rising **average cross-asset correlation** is read as a fragility indicator that should **predict**
lower forward returns and higher forward volatility. We build the mean pairwise correlation across a
14-ETF cross-asset panel, split the tape into HIGH- and LOW-correlation regimes, and test SPY's
forward return and forward vol across the two.

The vol half of the claim holds; the **return half is backwards, robustly.** In the HIGH-correlation
regime SPY's forward 21-day return averaged **+1.65%** versus **+0.67%** in the LOW regime — a
HIGH−LOW spread of **+0.98%** (two-sample *t* **+6.80**, block-shuffle placebo *p* **0.048**), the
**opposite** of "high correlation → lower returns." Forward volatility *does* run higher in the HIGH
regime (**17.2%** vs **15.6%**, spread **+1.62%**, *t* +4.62): correlation flags coming turbulence,
just not lower returns. The sign of the return spread is **positive in every** window/horizon/quantile
we tried. So `NONE` on the signal axis (the fragility-predicts-drawdowns claim is the wrong sign),
`MIRAGE` on tradability (a risk-off overlay that sits out the HIGH regime **underperforms** buy-and-hold
net: Sharpe **+0.42** vs **+0.62**, CAGR **+5.55%** vs **+12.20%**), and `COINCIDENT` on the myth
axis: high correlation marks the stress you are **already in**, not the crash ahead.

## Data stamp

- **Returns panel**: 14 cross-asset ETFs, daily, 2007-01-04 → 2026-06-26, fingerprint `9dbfa917d6ba`
- **Aligned regime/forward frame** (63-day corr, 70th-pct regime, 21-day forward): fingerprint `0b73f61632f6`
- **Correlation index**: mean off-diagonal pairwise correlation ≈ **0.275** (min 0.046, median 0.266,
  max **0.598** on 2022-12-06)

## The regime forward test — the return sign is BACKWARDS

| Forward 21d on SPY | HIGH-corr regime (n=1792) | LOW-corr regime (n=2774) |
|---|---|---|
| Mean forward **return** | **+1.65%** | **+0.67%** |
| Mean forward **vol** (ann.) | **17.2%** | **15.6%** |

| Spread (HIGH − LOW) | value | two-sample *t* | claim predicts | reads as |
|---|---|---|---|---|
| Forward **return** | **+0.98%** | **+6.80** | *negative* (fragility) | **wrong sign** |
| Forward **vol** | **+1.62%** | **+4.62** | *positive* (turbulence) | right sign |

Block-shuffle placebo (21-day blocks, 2000 perms) on the return spread: *p* = **0.048** — the effect
is real, it is just the *opposite* of the fragility story. The overlapping forward windows inflate the
raw *t*; the placebo is the honest significance read, and the sign is what matters.

## Why it's backwards — correlation spikes are BOTTOMS, not tops

| | value |
|---|---|
| Mean *contemporaneous* SPY drawdown, HIGH regime | **−11.7%** |
| Mean *contemporaneous* SPY drawdown, LOW regime | **−5.3%** |

The HIGH-correlation days are exactly the days SPY is **already deep in a drawdown** (2009: 91% of
days HIGH; 2020: 63%; 2022: 62%). Correlations peak *during* the crash — a **coincident** stress
signal. But by the time the panel is maximally correlated the market is near a *bottom*, and the
forward window catches the V-shaped rebound. So "everything moving together" flagged the *buy*, not
the *sell*: the folklore mistakes a coincident stress gauge for a leading crash predictor.

## Robustness — the wrong sign is stable

| window | horizon | q | return spread (HIGH−LOW) | *t* | vol spread | *t* |
|---|---|---|---|---|---|---|
| 63 | 21 | 0.70 | **+0.98%** | +6.80 | +1.62% | +4.62 |
| 63 | 21 | 0.80 | +1.03% | +6.62 | +1.03% | +3.01 |
| 63 | 63 | 0.70 | +3.37% | +14.81 | −0.06% | −0.18 |
| 126 | 21 | 0.70 | +1.50% | +10.64 | +0.51% | +1.47 |
| 126 | 63 | 0.70 | +3.76% | +16.75 | −1.03% | −3.34 |
| 42 | 21 | 0.70 | +0.72% | +4.94 | +1.57% | +4.52 |
| 63 | 5 | 0.70 | +0.18% | +2.36 | +2.77% | +6.63 |

The forward-return spread is **positive in all seven specs** — the fragility claim's sign never
appears. The forward-vol spread is positive at short horizons (turbulence *is* flagged) and fades to
zero/negative by 63 days.

## Tradability — the risk-off overlay LOSES to buy-and-hold

| Book (2007-2026) | CAGR | vol | Sharpe | max drawdown |
|---|---|---|---|---|
| Overlay gross (flat in HIGH regime, 1-day lag) | +5.81% | 13.3% | +0.44 | −37.5% |
| Overlay **net** (5 bps/switch, 98 switches) | **+5.55%** | 13.3% | **+0.42** | −37.5% |
| Buy-and-hold SPY | **+12.20%** | 19.6% | **+0.62** | −55.2% |

Sitting out the HIGH-correlation regime *did* cut the drawdown (−37.5% vs −55.2%) — but it also cut
the return in half and *lowered* the risk-adjusted Sharpe, because the regime it dodged contained the
best forward returns (the rebounds). Net of a trivial 5 bps/switch cost the overlay is a strictly
worse portfolio: **`MIRAGE`**.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted `fragility` | Mean return spread (HIGH−LOW) | Mean *t* (25 seeds) | |
|---|---|---|---|
| 0.00 (null) | +0.05% | **+0.37** | flat — no false signal |
| 0.50 | −0.26% | −1.48 | effect emerging |
| 0.90 | −0.50% | **−2.57** | clears the bar |
| 1.40 | −0.78% | **−3.44** | strong |

In the synthetic world where fragility is *planted* (high correlation genuinely precedes worse
returns), the engine recovers a **negative** HIGH−LOW spread past *t* = −2 as the effect grows, and
stays flat (*t* ≈ 0) at the null. So the detector works and has the right sign convention — the
positive real-tape spread is the **tape talking**, not a broken engine. (Control only; never cited
for the real-tape stamp.)

## Data-availability & honesty notes

1. **Sample span, not survivorship.** The panel is broad-index ETFs (no single-name survivorship),
   but liquid multi-asset ETF history only reaches ~2007 — one GFC and one COVID stress, not a
   century of crises. Named on the SIGNAL axis: a longer tape (or index-level proxies back to the
   1970s) could in principle contain a leading-crash episode this sample lacks.
2. **Coincident vs leading.** The regime label uses only trailing data and the forward window is
   strictly *after* day *t* (one-day execution lag on the overlay), so there is no look-ahead — the
   "wrong sign" is a genuine forward result, not leakage.
3. **Overlapping windows.** The 21-day forward windows overlap, so the raw two-sample *t* overstates
   precision; the block-shuffle placebo (*p* = 0.048) is the significance we lean on, and the stable
   *sign* across seven specs is the real evidence.

## The honest takeaway

Rising cross-asset correlation is a **real and coincident** stress gauge — it spikes precisely when
markets are already in a drawdown, and it *does* forecast higher near-term volatility. But it is
**not** a leading crash predictor: on 2007-2026, the HIGH-correlation regime was followed by
*higher* SPY returns (+0.98%/month, *t* +6.80, placebo *p* 0.048, sign stable across every spec),
because correlation peaks near bottoms and the rebound follows. A risk-off overlay built on it
underperforms buy-and-hold net (Sharpe +0.42 vs +0.62). `NONE` × `MIRAGE`, with the fragility gauge
itself `COINCIDENT` — it marks the storm you are in, not the one ahead.
