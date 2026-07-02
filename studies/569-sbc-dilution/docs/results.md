# Results — Study 569 (SBC-Dilution): the hidden-cost anomaly on a survivor basket

*Generated from [`sbc_dilution/`](../sbc_dilution/) over this study's cached yfinance tape:
year-end adjusted closes, split-adjusted shares outstanding, and a per-name **SBC/revenue**
snapshot for a fixed **31-name large-cap survivor basket**. Prices fingerprint `db57cbb8d955`,
shares `9629ac4ff009`, SBC-intensity `75f937c63b8e`, joint panel `bbec9e324ac1`. Formation
year-ends 2016 → 2024, each held the **next** calendar year (one execution lag). As-of
**2026-06-30**; the partial 2026 year is dropped.*

## The verdict, earned — Signal `NONE` · Tradability `MIRAGE` · "Hidden-cost anomaly replicates here?" `BUSTED`

The claim (an accounting hidden-cost anomaly, cousin of net-share-issuance): stock-based
compensation is a real economic cost that dilutes shareholders while barely touching GAAP
earnings, so the heaviest-SBC / fastest-diluting firms should earn *lower* forward returns. The
tradable expression is **long the low-dilution (lean) names, short the high-dilution (heavy-SBC)
names**; the claim predicts a *positive* long-short spread.

On a **31-name large-cap survivor** basket over **9** annual rebalances (formation 2016 → 2024),
the long-short spread is **−15.2%/yr at *t* = −2.01** — the **wrong sign** and, if anything,
significant *against* the claim. A **label-shuffle placebo** beats the real sort **99.8%** of the
time (the real sort sits deep in the *losing* tail, placebo *p* = 0.998), and the win-rate is only
**22%** (2 of 9 years positive). The result is stable the wrong way across every quantile width
(*t* −1.9 to −2.0) and holds in each leg alone (SBC-only −41%, dilution-only −8%). A faithful
25-seed synthetic control confirms a *genuine* hidden-cost edge would print a clearly **positive**
*t* — so this is a real **inversion on this tape**, not a broken engine.

So `NONE` on the signal axis (no positive *t* ≥ 2 for the claim — the sign is reversed), `MIRAGE`
on tradability (a survivor basket, annual rebalance, with the short leg being the expensive-to-borrow
mega-cap tech), and `BUSTED` on replication: the anomaly that lives in the full survivorship-free
cross-section does not survive a 31-name survivor sort over an AI-driven melt-up that rewarded
exactly the heavy-diluting growth names.

## Data stamp

- **Prices**: 31 large-cap survivors, year-end adjusted close, fingerprint `db57cbb8d955`
- **Shares**: split-adjusted shares outstanding (so a 4-for-1 split is not 300% dilution), year-end,
  fingerprint `9629ac4ff009`
- **SBC intensity**: `Stock Based Compensation` (cash-flow) / `Total Revenue` (income statement),
  fingerprint `75f937c63b8e` — a **shallow ~4-year snapshot** (yfinance exposes only a shallow
  statement history), forward-filled onto the year grid; named on the SIGNAL axis
- **Joint panel** (formation 2016 → 2024): fingerprint `bbec9e324ac1`

## The long-short — the anomaly is the WRONG WAY ROUND here

| | value |
|---|---|
| Formation years | **9** (2016 → 2024, held *t → t+1*) |
| Long (low-dilution) mean | **+17.4%/yr** |
| Short (high-dilution) mean | **+32.6%/yr** |
| **Long-short spread (long low − short high)** | **−15.2%/yr** |
| One-sample *t* (vs 0) | **−2.01** |
| Win-rate (positive years) | **22%** (2/9) |
| Annual Sharpe | **−0.67** |
| Label-shuffle placebo *p* (P[placebo ≥ real]) | **0.998** |

The claim predicts long-low / short-high > 0 (the diluters lag). The tape delivers the opposite:
the **high-dilution** bucket — the SBC-heavy AI mega-caps (2024 formation short leg: NVDA, META,
AMZN, MSFT, CRM, AMD, ORCL, QCOM, INTC) — *out*-earned the lean value names by 15pts/yr. The
placebo *p* = 0.998 says the real sort is worse than 998 of every 1000 random relabellings: this
is a genuine inversion on this window, not noise.

## Each leg alone — the sign is consistently reversed

| Signal | Years | LS mean | *t* |
|---|---|---|---|
| **Composite** z(SBC/rev) + z(share growth) | 9 | **−15.2%** | **−2.01** |
| **SBC-only** (intensity) | 3 | **−41.3%** | −1.92 |
| **Dilution-only** (share growth) | 9 | **−8.2%** | −1.37 |

Both legs point the *wrong* way; the SBC leg (only 3 snapshot years) is the most extreme. A hidden
cost that were being under-priced would show a positive spread in at least one leg — none does.

## Robustness — stable, but stably inverted

| Quantile width | Years | LS mean | *t* | Sharpe |
|---|---|---|---|---|
| 0.2 | 9 | −13.8% | −1.91 | −0.64 |
| 0.3 (headline) | 9 | −15.2% | −2.01 | −0.67 |
| 0.4 | 9 | −16.5% | −1.99 | −0.66 |

The wrong sign is not an artefact of the tail width — it is the same, near *t* −2, at every cut.

## Costs

| | value |
|---|---|
| Gross LS mean (headline) | **−15.2%/yr** |
| Net (10 bps/leg × 2 legs, full turnover + 50 bps/yr short borrow) | **−15.9%/yr** |
| Net *t* | **−2.10** |

Costs are a footnote: the trade is the wrong sign before you pay for it, and the heavy-dilution
short leg (mega-cap tech) is exactly the expensive-to-borrow tail.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds, 6-year panels)

| Planted `edge` | Mean *t* (25 seeds) | LS mean | |
|---|---|---|---|
| 0.00 (null) | **+0.06** | −0.3% | flat — no false signal |
| 0.10 | +1.00 | +4.8% | edge emerging |
| 0.20 | +1.81 | +9.9% | nearly there |
| 0.30 | **+2.42** | +15.0% | clears the bar |

At the null the *t* is ≈ 0; planting a genuine hidden-cost edge (heavy-diluters underperform)
drives the long-short *t* **positive** and past +2 as it grows. The detector works and points the
*right* way when the effect is real — so the real-tape *negative* *t* is a statement about **this
survivor basket on this window**, not a broken or sign-flipped engine. (Control only; never cited
for the real-tape stamp.)

## Why the anomaly doesn't certify here

1. **Survivorship, the wrong way.** The basket is names *still trading in 2026*. The real
   dilution/SBC penalty is driven by firms that over-issued into oblivion (secondary-raise
   death spirals, wipe-outs) — absent by construction here. Strip them out and you keep the
   *survivors* of heavy dilution, biasing the tape against the anomaly.
2. **An AI/growth melt-up window.** 2020-24 rewarded exactly the heavy-SBC growth mega-caps
   (NVDA, META, AMZN, CRM): the "diluters" *led*. The anomaly earns its keep over full cycles and
   in the full cross-section, not a growth-led bull run on 31 blue chips.
3. **SBC snapshot, not point-in-time.** yfinance exposes only a shallow ~4-year statement history,
   so the SBC-intensity leg is a recent snapshot forward-filled onto the grid rather than a true
   year-*t* panel — adequate for an illustration, but a real replication needs Compustat-style
   point-in-time SBC and a survivorship-free universe.

## The honest takeaway

The stock-based-comp / dilution anomaly is a real, documented cost-hiding effect in the full US
cross-section — but on a 31-name survivor basket over 2016-24 it **inverts**: long-lean /
short-diluters loses **−15.2%/yr at *t* −2.01**, the placebo beats it 998 times in 1000, every leg
and width points the wrong way, and the synthetic control confirms a genuine edge would have
printed a *positive* *t*. `NONE` × `MIRAGE`, replication `BUSTED` — a textbook survivorship /
small-sample illustration, not a tradable edge. The tape is talking, not the code.
