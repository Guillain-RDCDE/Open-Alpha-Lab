# Results — Study 748 (CEO-Age-Effect): do young-CEO firms out- or under-perform old-CEO firms?

*Generated from [`ceo_age_effect/`](../ceo_age_effect/) over this study's cached yfinance tape:
daily **total-return** (dividend-adjusted) closes for a **hand-curated 40-name large-cap CEO
table** + SPY, price fingerprint `4c394346b808`, 2018-01-02 → 2026-06-29, resampled to monthly.
Each CEO is bucketed **young** (age < 55 at the scoring date) or **old** from a public birth year;
the equal-weight long-young / short-old book is the tradable claim. Scored as-of **2024-12-31**
(age split), monthly return tape **2018-02 → 2026-06** (101 months, partial trailing month dropped).
L/S panel fingerprint `bcef5ab08c9a`. As-of **2026-06-30**.*

## The verdict, earned — Signal `NONE` · Tradability `MIRAGE` · "Young-CEO edge?" `MISATTRIBUTED`

The claim, with real academic backing: young CEOs run aggressive, high-risk, growth-hungry firms
(Yim 2013), old CEOs play it safe (Serfling 2014) — so a long-young / short-old equity book should
pay. We build exactly that book on 40 large-cap names split by their CEO's age.

It does earn a raw spread — **+7.97%/yr** — but that spread is **not statistically distinguishable
from noise and not an age effect at all**. The Newey-West HAC *t* on the mean monthly long/short is
**+0.92** (4 lags, n = 101), miles short of the *t* ≥ 2 bar; a label-shuffle placebo produces a |t|
this large **52% of the time**. Worse for the story, the young basket has a **lower Sharpe (0.90)
than the old basket (1.08)** — it earns more only by taking far more risk (32.4% vol vs 19.8%), so
risk-adjusted the *old* CEOs win. And the decisive control kills it: regress the L/S on the market
and the **CAPM alpha is +2.98%/yr at *t* = 0.36** while the **market beta is +0.35 (*t* = 2.7)** —
the entire "young premium" is growth/market beta, because the young bucket is a founder-led
growth-tech cohort (TSLA, META, COIN, HOOD, DASH, SHOP, SNAP…). The spread's sign even flips by
regime (+27%/yr in 2018–20, −34%/yr in the 2021–22 rate shock, +17%/yr in the 2023–26 AI melt-up)
and its *t* wanders with the arbitrary age cutoff. So `NONE` on signal (nothing survives, and a
curated confounded 40-name table can't certify `REAL` anyway), `MIRAGE` on tradability (the net
+7.19%/yr is beta you can rent cheaper), and the folklore's mechanism is `MISATTRIBUTED` — it's
sector/size/vintage, not the birthday.

## Data stamp

- **Prices**: 40 large-cap tickers + SPY, daily total-return closes, 2018-01-02 → 2026-06-29,
  fingerprint `4c394346b808`
- **Curated CEO → age table**: 40 names (public birth years → age at 2024-12-31), 14 young (< 55),
  26 old
- **L/S monthly panel** (2018-02 → 2026-06, 101 months): fingerprint `bcef5ab08c9a`

## The baskets — young earns more, but loses on risk-adjusted return

| | annual return | annual vol | Sharpe |
|---|---|---|---|
| Young-CEO basket (equal-weight) | **+29.24%** | 32.44% | **0.90** |
| Old-CEO basket (equal-weight) | **+21.27%** | 19.76% | **1.08** |
| Market (SPY, total return) | +14.46% | 16.53% | 0.87 |

The "aggressive youth" half of the claim is real — young-CEO firms are much more volatile — but
volatility is not reward: per unit of risk the **old** CEOs delivered more.

## H1 — the long/short premium (Newey-West HAC)

| | value |
|---|---|
| Long-young / short-old, annualised | **+7.97%/yr** (mean +0.664%/mo) |
| Newey-West HAC *t* (4 lags, n = 101) | **+0.92** |
| Label-shuffle placebo *p* (on \|t\|) | **0.523** |

A *t* of 0.92 is a coin flip; a random relabelling of the same CEOs beats it more than half the
time. **H1 not rejected.**

## H2 — alpha vs beta (CAPM with a HAC covariance) — the decisive control

| | value |
|---|---|
| CAPM alpha | **+2.98%/yr** |
| Alpha HAC *t* | **+0.36** |
| Market beta | **+0.35** |
| Beta HAC *t* | **+2.7** |
| R² | 0.07 |

The book has a real, significant market beta and an alpha indistinguishable from zero. **The entire
spread is market/growth beta.** H2 decisively not rejected.

## H3 — robustness: the cutoff sweep (the *t* wanders, never clears 2)

| Young cutoff | L/S %/yr | L/S HAC *t* | CAPM alpha %/yr | alpha *t* | beta |
|---|---|---|---|---|---|
| age < 50 | +1.1 | 0.11 | −6.1 | −0.66 | 0.50 |
| age < 55 (headline) | +8.0 | 0.92 | +3.0 | 0.36 | 0.35 |
| age < 60 | +11.0 | 1.90 | +7.6 | 1.46 | 0.24 |

The apparent "signal" depends entirely on where you draw the arbitrary age line, and never clears
significance at any cutoff.

## H3 — robustness: the sub-period sweep (sign flips by regime)

| Period | L/S %/yr | L/S HAC *t* | CAPM alpha *t* | beta |
|---|---|---|---|---|
| 2018–2020 (growth boom) | +26.6 | 1.96 | **+2.03** | 0.12 |
| 2021–2022 (rate shock) | −34.5 | −2.22 | **−2.35** | 0.18 |
| 2023–2026 (AI melt-up) | +16.8 | 1.64 | −0.04 | 0.81 |

A single characteristic-alpha does not reverse sign with interest rates. This is a growth-factor
timing exposure, full stop.

## Execution lag & costs

| | value |
|---|---|
| Lag 0 (contemporaneous, calendar-known) | +7.97%/yr, HAC *t* +0.92 |
| Lag 1 (conservative one-month formation) | +7.29%/yr, HAC *t* +0.83 |
| Gross annualised (Sharpe) | +7.97%/yr (0.38) |
| Net of 5 bps/leg × 0.30 turnover + 75 bps borrow (Sharpe) | **+7.19%/yr (0.34)** |

Membership is calendar-known so the lag is immaterial; turnover is negligible (a CEO's age barely
changes), so costs barely dent the number — which only underlines that there is nothing real to
charge them against.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted `age_alpha` | Mean CAPM alpha *t* (25 seeds) | Mean raw HAC *t* | |
|---|---|---|---|
| +0.0%/yr (null, beta tilt only) | **+0.01** | −0.00 | flat — no false alpha |
| +4.8%/yr | +2.12 | +1.51 | clears the bar |
| +9.6%/yr | +4.23 | +3.03 | unmistakable |
| +14.4%/yr | +6.35 | +4.54 | unmistakable |

At the null the CAPM alpha *t* sits at ~0 even though young firms carry a higher beta — the control
separates *beta* from *alpha* correctly. Plant a genuine premium and the alpha *t* sails past 2. The
detector works, so the flat real-tape alpha (*t* = 0.36) is a statement about the world, not a
broken engine. (Control only; never cited for the real-tape stamp.)

## Why this can never be REAL

1. **Confounded characteristic.** "Young CEO" ≈ "founder-led growth-tech that IPO'd recently." The
   raw spread loads the growth/market factor; the CAPM alpha — the part that could be an *age* edge
   — is zero.
2. **Curated, not survivorship-free, tiny.** A 40-name hand-picked table with a growth-tilted young
   bucket is neither powered nor bias-free; it exists to test (and bust) the *trade*, not to certify
   a premium.
3. **Regime-dependent sign.** An effect that reverses with interest rates is a factor exposure, not
   a firm-characteristic alpha.

## The honest takeaway

Young-CEO firms did earn more (+29%/yr vs +21%/yr) — but only by being far more volatile (Sharpe
0.90 vs 1.08, so risk-adjusted the *old* CEOs won), and the long/short spread is a coin flip (HAC
*t* 0.92, placebo *p* 0.52) that vanishes to zero alpha once you subtract the market (alpha *t*
0.36, beta *t* 2.7) and flips sign every regime. `NONE` × `MIRAGE`, mechanism `MISATTRIBUTED`. The
"aggressive young CEO" is real in the corporate-policy papers (Serfling, Yim); as a stock trade it
is just growth beta in a birthday costume.
