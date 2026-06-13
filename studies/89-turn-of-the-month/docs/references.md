# References & literature map — Study 89 (Turn-of-the-Month)

## The claim under test

The "turn-of-the-month" (TOM) effect is the observation that a disproportionate share of
the stock market's gains accrues in a narrow window straddling the month boundary — the
**last trading day of the old month plus the first three of the new one** (the canonical
`[-1, +3]` window). The strong, sold-at-full-strength version: *almost all of the market's
return happens in that ~4-day window, so be invested only then and you capture the market
with a fraction of the risk.*

## The primary literature

- **Ariel, R. (1987), "A Monthly Effect in Stock Returns", *Journal of Financial
  Economics* 18(1).** The foundational paper: cumulating returns separately for the first
  vs second half of each month, Ariel finds essentially all of the market's positive
  return occurs in the first half (and around the turn), with the rest of the month flat
  to negative over 1963–1981.
- **Lakonishok, J. & Smidt, S. (1988), "Are Seasonal Anomalies Real? A Ninety-Year
  Perspective", *Review of Financial Studies* 1(4).** Using ~90 years of the Dow, they
  sharpen the window to the four days `[-1, +3]` around the turn and document a large,
  persistent average return there — the definition this study adopts.
- **McConnell, J. & Xu, W. (2008), "Equity Returns at the Turn of the Month", *Financial
  Analysts Journal* 64(2).** Re-examines the effect into the 2000s and finds it persists
  and is not confined to small caps — the steelman that the effect did not simply vanish
  after publication.

## Why the steelman is almost coherent

- The window is **calendar-known** — defined purely by trading-day-of-month position, so
  there is no forecasting and no look-ahead, and (per house rules) **no execution lag**: a
  calendar rule is not a signal that must be acted on a day late.
- Proposed mechanisms are economically plausible: **month-end cash flows** (salary,
  pension and 401(k) contributions, fund inflows, dividend reinvestment) and
  **window-dressing / settlement** effects concentrate buying around the turn.
- On deep history the effect is genuinely large and statistically strong (see the
  price-only `^GSPC` 1950– result in [`results.md`](results.md): HAC *t* ≈ 5).

## Why it is likely to fail *as stated*

- **"Almost all the gains" overstates it.** On the total-return SPY tape the TOM window is
  ~19% of trading days and earns ~33% of the cumulative return — a real ~1.7x
  over-representation, not "almost all." The rest of the month still earns a positive
  equity premium you would forgo.
- **A TOM-only timer forgoes the equity premium ~81% of the time.** Being in cash four
  days a month and out the rest hands back most of the market's compounding; the headline
  "less risk" is mostly **less exposure** (lower beta), which a stock/cash blend
  reproduces without dozens of switches a year. (House rule: *normalise before you marvel*
  — the per-day-invested Sharpe is the only comparison on which the timer looks good.)
- **Post-publication / modern-sample fragility.** On the 1993– total-return SPY sample the
  daily TOM-minus-rest premium does **not** clear HAC *t* = 2; the literature certifies the
  effect, this tape alone cannot.

## Method lineage

- **Newey–West HAC standard errors** for the mean of an autocorrelated return series and
  for the dummy-regression slope (the TOM-minus-rest difference of means): Newey & West
  (1987), *A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation
  Consistent Covariance Matrix*, *Econometrica* 55(3).
- **Circular block bootstrap** for the decay test (the difference of the TOM premium
  between an early and late sub-period): resampling in blocks preserves the
  autocorrelation that i.i.d. resampling would destroy (Politis & Romano 1992;
  Künsch 1989).

## Data sources used

- **SPY**, daily, **total-return adjusted** (dividends folded in) via `quantlab.data`
  (Yahoo Finance), cached to parquet under `_cache/`. The headline tape — total return is
  the fair benchmark for a strategy that sits in cash part of the time.
- **^GSPC**, daily, **price-only / split-only** (no dividends) via `quantlab.data` — a
  longer 1950– sample, reported alongside and **explicitly labelled price-only** (its
  rest-of-month drift is understated, which flatters the TOM share; never called total
  return).
- Cash is assumed to earn **0%** — a stated, conservative choice that biases *against* the
  TOM-only timer.

## Related desk studies

- [Study 91 — Death-Cross](../../91-death-cross/) — the "less beta sold as alpha" teardown
  and the normalise-before-you-marvel discipline this study reuses.
- Other calendar-anomaly teardowns on the bench (sell-in-May, day-of-week) share the same
  "real concentration, no tradable edge" shape.
