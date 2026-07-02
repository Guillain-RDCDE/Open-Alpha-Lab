# Results — Study 544 (Oyster-R-Months): the folk 'eat oysters only in R months' calendar on the tape

*Generated from [`oyster_r_months/`](../oyster_r_months/) over this study's cached yfinance tape:
monthly total returns of the **S&P 500** (`^GSPC`, 1928-01 → 2026-06, **1182** complete months,
fingerprint `e89dca918e03`) and the **SPDR Consumer-Staples ETF** (`XLP`, 1999-01 → 2026-06,
**330** complete months, fingerprint `15bb025c5ac1`). The R/non-R label is a calendar fact known
in advance (no estimation, no look-ahead). Trailing partial month dropped. As-of **2026-06-30**.*

## The verdict, earned — Signal `NONE` · Tradability `MIRAGE` · "R-month rule ≠ dressed-up sell-in-May?" `MYTH`

The folk rule says eat oysters only in months with the letter **R** — September through April —
and skip the R-less months of **May, June, July, August**. Transposed to markets, this is almost
exactly **sell-in-May / Halloween** (Bouman & Jacobsen 2002), one month wider on the winter side.
We test whether R-months out-earn R-less months on the broad market and on consumer staples.

**They do not.** On the **market**, R-months earned *less* than R-less months — a monthly gap of
**−24.5 bp** (R-months +7.1%/yr vs R-less +10.3%/yr), Welch *t* **−0.72**, label-shuffle placebo
*p* **0.457**: the R-month rule is the **wrong sign** and insignificant. On **staples** the gap is
the *right* sign but tiny and insignificant: **+17.9 bp/mo** (R +8.2%/yr vs non-R +5.9%/yr), Welch
*t* **0.45**, placebo *p* **0.670**. Neither instrument clears *t* ≥ 2 → `NONE`. The tradable
"in during R-months, cash in R-less months" rule compounds to a **fraction** of buy-and-hold
(market: **28x** vs **425x** over 98 years) → `MIRAGE`. And the myth axis: the R-month split is a
*worse* sell-in-May, because it adds **September** — historically the **single worst month**
(−111.7 bp/mo over 98 years) — to the "hold" side. So the oyster rule is not sell-in-May's equal;
it is sell-in-May with its best-known weakness (September) bolted on → `MYTH`.

## Data stamp

- **Market**: S&P 500 (`^GSPC`) monthly total return, 1928-01 → 2026-06, 1182 months, fp `e89dca918e03`
- **Staples**: SPDR Consumer-Staples ETF (`XLP`) monthly total return, 1999-01 → 2026-06, 330 months, fp `15bb025c5ac1`

## The R-month split — wrong sign on the market

| Instrument | R-months (Sep–Apr) | R-less (May–Aug) | Gap (R − non-R) | Welch *t* | Placebo *p* |
|---|---|---|---|---|---|
| **Market** (^GSPC, 1928–2026) | +0.57%/mo (**+7.1%/yr**) | +0.82%/mo (**+10.3%/yr**) | **−24.5 bp/mo** | **−0.72** | 0.457 |
| **Staples** (XLP, 1999–2026) | +0.66%/mo (**+8.2%/yr**) | +0.48%/mo (**+5.9%/yr**) | **+17.9 bp/mo** | **+0.45** | 0.670 |

The oyster rule predicts R-months win. On the market they *lose*; on staples they win by a
statistically invisible margin. No *t* ≥ 2 anywhere → `NONE`.

## The sell-in-May cousin — the R-month rule is a WORSE version of it

| Split | Market gap (in − out) | Market Welch *t* |
|---|---|---|
| **R-months** (Sep–Apr) vs (May–Aug) | **−24.5 bp/mo** | **−0.72** |
| **Halloween** (Nov–Apr) vs (May–Oct) | **+40.6 bp/mo** | **+1.31** |

The Halloween split is a *positive*, if soft, seasonal (the [Study 55 Summer-Lull](../../55-summer-lull/)
result). The R-month split flips it to *negative* — because it moves **September** from the "out"
(skip) side to the "in" (hold) side, and September is the market's worst month:

| Month | Mean return (1928–2026, 98 obs) |
|---|---|
| **September** | **−1.12%/mo (−111.7 bp)** — the single worst calendar month |

That one month is why the oyster rule underperforms its sell-in-May cousin.

## Tradability — hold in R-months, cash in R-less months

| | Buy & hold | R-month rule (net, 5 bps/switch) |
|---|---|---|
| **Market** CAGR | **6.3%** | **3.4%** |
| **Market** Sharpe | **0.43** | **0.31** |
| **Market** wealth multiple (98 yr) | **425x** | **28x** |
| **Staples** CAGR | **6.6%** | **4.7%** |
| **Staples** Sharpe | **0.58** | **0.48** |
| **Staples** wealth multiple (27 yr) | **5.8x** | **3.5x** |

Sitting in cash for four months a year — months that on the market *out-earn* the R-months —
compounds to a fraction of buy-and-hold at a lower Sharpe. There is nothing to trade → `MIRAGE`.
(The literal long-R/short-R-less book is worse still: gross **−0.5%/yr** on the market, **−0.8%**
net of a 50 bps borrow on the short leg.)

## Sub-period stability — no decay because there was never a signal

| Instrument | Pre-2000 gap-*t* | Post-2000 gap-*t* |
|---|---|---|
| Market | −0.79 (n=864) | +0.01 (n=318) |
| Staples | — (n=12, pre-ETF) | +0.58 (n=318) |

The market gap-*t* is negative early and ~0 late; there is no era in which the R-month rule was a
real, tradable edge.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted `r_premium_bp` | Mean gap-*t* (25 seeds) | |
|---|---|---|
| 0 (null) | **−0.05** | flat — no false signal |
| +20 | +0.63 | emerging |
| +40 | +1.30 | visible |
| +60 | +1.98 | at the bar |
| +80 | **+2.65** | clears the bar |

At the null the gap-*t* is ≈ 0; planting a genuine R-month premium drives the Welch *t* up and past
+2. The detector works — so the flat/wrong-sign real-tape result is the **tape talking**, not a
broken engine. (Control only; never cited for the real-tape stamp.)

## The honest takeaway

The "eat oysters only in R-months" rule is folk food-safety wisdom, not a market seasonal. On the
market the R-months earned *less* (*t* −0.72, placebo *p* 0.46); on staples slightly more but
insignificantly (*t* 0.45). The rule is a *dressed-up* sell-in-May — and a **worse** one, because
it hands September (the market's worst month) to the hold side. `NONE` × `MIRAGE`, myth axis `MYTH`.
The synthetic control confirms the engine would catch a real R-month premium — so this is a genuine
null, not a bug.
