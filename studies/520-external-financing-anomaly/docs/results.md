# Results — Study 520 (External-Financing-Anomaly): net financing sorts on a large-cap basket, 2022–2024

*Generated from [`external_financing_anomaly/`](../external_financing_anomaly/) over the cached
yfinance tape: cash-flow / balance-sheet line items for a fixed **45-name large-cap survivor
basket** (fundamentals fingerprint `ca0833b2ddb8`, 223 name-years) and the matching daily
adjusted-close price panel (2018-01-02 → 2026-06-25, 2131 bars × 45 names, fingerprint
`4106bae4e4d7`). Signal years **2022–2025**; only **2022, 2023, 2024** survive the reporting lag +
12-month holding window, so the stamped long-short rests on **3 complete annual cross-sections**.
As-of **2026-06-26**.*

## The verdict, earned — Signal `NONE` · Tradability `MIRAGE` · "Do big raisers underperform here?" `BUSTED`

Bradshaw–Richardson–Sloan (2006) say firms raising a lot of external finance (debt **and** equity)
subsequently underperform. We build their cash-flow measure — net external financing scaled by
average total assets — sort the basket each year, and go long the *retirers* / short the *raisers*.
On this survivor basket over 2022–2024 the **sign is reversed**: the long-short loses **−8.5%/yr**
(the raisers *out*-earned the retirers), with a one-sample *t* of **−1.12** and a label-shuffle
placebo p-value of **0.86** (a random sort beats the real one 86% of the time). There is no signal
to bank — and there is barely any tape to bank it on: yfinance exposes only ~5 fiscal years of
cash-flow detail, leaving just three complete forward windows. So `NONE` on the signal,
`MIRAGE` on tradability, and `BUSTED` on the qualitative question of whether big raisers
underperform *here*.

## Data stamp

- **Basket**: 45 large-cap survivors (still trading 2026), cash-flow + balance-sheet from yfinance
- **Fundamentals**: 223 name-years, fingerprint `ca0833b2ddb8`
- **Prices**: 2131 daily bars × 45 names, 2018-01-02 → 2026-06-25, fingerprint `4106bae4e4d7`
- **Signal years**: 2022, 2023, 2024, 2025 (40–45 names each); **2025 dropped** (forward window
  runs past the price tape) and a partial 2026 row (5 names) dropped — no partial-year bar in a stamp
- **Execution**: signal = prior fiscal year's scaled XFIN, public after a **90-day** reporting lag;
  enter the close **one trading day after** the public date; hold **252 trading days**. One lag, no
  same-bar fill, no look-ahead.

## The long-short (long retirers, short raisers)

| Book | mean (ann) | Sharpe | one-sample *t* | HAC *t* | hit |
|---|---|---|---|---|---|
| Gross spread | **−8.5%/yr** | −0.65 | **−1.12** | −1.69 | 33% |
| Net (10 bps/leg + 50 bps/yr borrow) | −9.2%/yr | −0.70 | −1.21 | −1.83 | 33% |

Year by year (long retirers, short raisers):

| FY signal | long leg | short leg | spread | n names |
|---|---|---|---|---|
| 2022 | +20.3% | +25.0% | **−4.7%** | 40 |
| 2023 | +14.6% | +12.3% | **+2.3%** | 45 |
| 2024 | +14.0% | +37.2% | **−23.2%** | 45 |

The raisers won decisively in 2022 and 2024 (2024 especially — the AI-capex mega-caps that *raised*
the most also ran the hardest). Only 2023 prints the BRS sign, and weakly. Mean spread sits below
zero with a *t* nowhere near the bar.

## Placebo / label-shuffle null

Shuffling the financing labels within each year (2000 draws) and re-forming the long-short gives a
null mean of **+0.1%/yr** (std 8.0%). The real spread of **−8.5%/yr** has a placebo **p = 0.86** —
i.e. a *random* sort beats the real one 86% of the time. The real sort is, if anything, on the wrong
side of noise.

## Robustness — the cut doesn't rescue it

| Cut | mean (ann) | one-sample *t* | n |
|---|---|---|---|
| Median split | −10.0%/yr | −1.31 | 3 |
| Tercile (30%) | −8.5%/yr | −1.12 | 3 |
| Quartile (25%) | −6.7%/yr | −1.18 | 3 |

Every cut prints the same negative sign and an insignificant *t*. The result is robust only in the
sense that it is robustly *absent*.

## Synthetic positive control (faithful-engine / power check ONLY)

A deterministic synthetic world plants a true external-financing penalty (big raisers underperform).
Averaged over **20 RNG seeds**, the same long-short recovers a planted mean *t* of **+7.6**
(100% of seeds clear *t* ≥ 2), with the matched null (no planted penalty) sitting flat at *t* = −0.10
and a planted spread of **+9.1%/yr**. The engine **does** detect the anomaly when it is really there
— so the flat real-tape result is a property of the tape, not a broken sort. *This control is a power
proof only and is never cited toward the real-tape stamp.*

## The honest takeaway

The external-financing anomaly does not replicate on this basket — not because the engine can't see
it (the synthetic control recovers a planted penalty at *t* 7.6), but because (1) the **sign is
reversed** here: in a 2022–2024 mega-cap window the heaviest external-finance *raisers* (debt-funded
AI-capex names) were the *winners*, not the losers; (2) the tape is **desperately thin** — yfinance's
~5-fiscal-year fundamentals depth leaves only three complete annual cross-sections; and (3) the
basket is **survivorship-filtered** — the dead raisers the anomaly says should have cratered are
absent, which can only *flatter* the raiser leg. `NONE` × `MIRAGE`, the on-brand outcome for a
pointed academic factor replicated honestly on a small survivor basket with real costs.
