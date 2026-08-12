# References & literature map — Study 876 (Industry-Relative MAX)

## The claim under test

- **The source paper.** Turan G. **Bali, Nusret Cakici & Robert F. Whitelaw**, *"Maxing Out:
  Stocks as Lotteries and the Cross-Section of Expected Returns"* (Journal of Financial
  Economics, 2011). Sorting the CRSP cross-section on **MAX** — a stock's single highest daily
  return over the prior month — they find the extreme high-MAX names go on to **under-earn**: a
  long low-MAX / short high-MAX portfolio earns a positive spread. The behavioural reading:
  investors with a taste for lottery-like, positively-skewed payoffs **over-pay** for names that
  just printed a big one-day pop, depressing their subsequent return.
- **The refinement tested here.** A name's raw MAX conflates two things: **sector-wide
  volatility** (a whole sector can be jumpy for macro reasons — a bad-CPI day slams every bank)
  and the **idiosyncratic** one-day pop that a lottery-demand story is actually about. We
  therefore sort on the **industry-relative MAX** — a name's MAX minus the **within-month median
  MAX of its GICS sector peers** — to strip the sector component and isolate idiosyncratic
  lottery demand, and ask whether the negative MAX→return relation **sharpens** or **dies**. This
  is in the spirit of industry-adjusted / characteristic-neutral sorts (e.g. Asness, Porter &
  Stevens 2000 on within-industry value & momentum; Novy-Marx 2013 on industry-adjusting factor
  signals), applied to the MAX lottery proxy.
- **The specific test here.** We take the self-contained monthly version on a liquid US
  cross-section: build both the raw and the industry-relative MAX each month, quintile-sort,
  measure the equal-weight long-Q1 / short-Q5 **next-month** return, and grade with a Newey-West
  *t*, a sign-flip placebo, a two-era robustness cut, a costed timer, and a seeded synthetic
  positive control. (Daily data on 50 mega-caps is a far smaller, more liquid slice than the
  paper's full CRSP universe, so the magnitudes are conservative and survivorship-tilted.)

## What we measure, and the honesty rails

- **MAX, no free model.** Each name's monthly MAX is the highest daily simple return in the
  calendar month (≥15 daily observations required). The industry-relative MAX subtracts the
  within-month **median MAX of the name's GICS sector** — a full-sector median (own name
  included; with 2–12 names per sector the leave-one-out shift is immaterial).
- **Point-in-time sort, one documented lag.** MAX is observed at the close of month `t`; the
  quintile book is held over month `t+1` (`fwd_ret = mret.shift(-1)`). Zero look-ahead.
- **Robust inference.** Newey-West (HAC, Bartlett, `floor(4(n/100)^{2/9})`-lag) *t* on the
  monthly long-short spread; a one-sample *t* cross-check; a **20,000-draw sign-flip placebo**
  (the small-sample workhorse) to confirm the spread is not a lucky alignment of the sort; a
  two-era cut to demand the result survive out of sub-sample.
- **Survivorship is named on the Signal axis.** The universe is a **current-membership** set of
  ~50 liquid mega-caps, pulled through the `quantlab.universe` survivorship guard
  (`allow_survivorship_bias=True`, an explicit opt-in). Delisted / de-rated names are absent, and
  the lottery effect is strongest in the small-cap names this panel omits — so the
  cross-sectional magnitudes are an **upper bound** and the venue is adverse to the claim.
- **The timer is graded separately.** Costs are 2 sides × one-way × NAV per monthly rebalance on
  the long-short book, and the short book pays borrow — the honest test of whether the spread
  survives friction.
- **The synthetic control never certifies the tape.** The seeded panel only proves the machinery
  is unbiased (null-silent) and that the industry adjustment *does* sharpen a genuinely-planted
  effect. The real-tape stamp is read from the real numbers alone.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the monthly spread).
- **Wilson, E. B. (1927)** — score interval for a binomial share.
- **Asness, C., Porter, R. B. & Stevens, R. (2000)** — within-industry vs across-industry
  decomposition of value and momentum, the intellectual template for industry-adjusting a
  cross-sectional signal.

## Data sources

- **yfinance daily OHLC** (`auto_adjust=True`, total-return), 50 liquid US large-caps, 2010-01-04
  → 2026-06-30, cached under `_cache/` through `quantlab.universe`.
- **GICS sector labels** — the S&P Dow Jones Indices / MSCI Global Industry Classification
  Standard, sector level; encoded statically in `max_industry/data.py` (`SECTORS`) so the study
  is fully offline. Visa / Mastercard / American Express are placed in Financials (their
  post-2023 GICS home); nothing here hinges on that single call.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [365-lottery-max-effect](../../365-lottery-max-effect/) — the **raw** MAX effect (a name's own
  maximum daily return), the parent this study refines. Study 876 sorts on the **industry-relative**
  MAX (MAX net of sector-peer median) and grades it **head-to-head** against the raw sort.
- [503-expected-idiosyncratic-skewness](../../503-expected-idiosyncratic-skewness/) — the
  **ex-ante / modelled** idiosyncratic skewness (Boyer-Mitton-Vorkink), a forecast of the whole
  skew, not the single extreme MAX order statistic net of sector.
- [806-prospect-theory-value](../../806-prospect-theory-value/) — the **prospect-theory value**
  of a name's historical return distribution (probability-weighted gains/losses), a full-distribution
  behavioural signal, not the industry-adjusted extreme tail.
- [538-industry-relative-reversal](../../538-industry-relative-reversal/) — an **industry-relative
  reversal** signal (past return net of industry), which shares the *industry-adjustment mechanic*
  but on the **reversal / past-return** characteristic, not the **MAX lottery** tail.

None of the siblings sort on a name's **maximum daily return net of its sector peers' median MAX**
— this study's own axis.
