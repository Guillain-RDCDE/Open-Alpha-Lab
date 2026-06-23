# References & literature map — Study 365 (Lottery / MAX effect)

## The claim under test

- **The anomaly (Bali, Cakici & Whitelaw).** Turan G. Bali, Nusret Cakici & Robert F.
  Whitelaw, *Maxing Out: Stocks as Lotteries and the Cross-Section of Expected Returns*
  (Journal of Financial Economics, 2011). They define **MAX** as a stock's *maximum daily
  return over the prior month* and show that, in the CRSP universe, the highest-MAX stocks
  *underperform* the lowest-MAX stocks by a large, significant margin — investors over-pay for
  "lottery-like" payoffs, so the flashy tail earns a low (even negative) future return. The
  trade is **long low-MAX, short high-MAX**, rebalanced monthly.
- **The behavioural mechanism.** The lottery-preference story rests on **probability
  weighting** and **skewness preference**: Tversky & Kahneman, *Advances in Prospect Theory*
  (1992); Barberis & Huang, *Stocks as Lotteries* (American Economic Review, 2008); Brunnermeier,
  Gollier & Parker, *Optimal Beliefs, Asset Prices, and the Preference for Skewed Returns*
  (2007). Investors over-weight small probabilities of large gains and so bid up positively
  skewed, lottery-like stocks past fair value.
- **Idiosyncratic-skewness cousins.** Boyer, Mitton & Vorkink, *Expected Idiosyncratic Skewness*
  (Review of Financial Studies, 2010); Kumar, *Who Gambles in the Stock Market?* (Journal of
  Finance, 2009) — gambling-motivated investors cluster in lottery stocks (low price, high
  idiosyncratic volatility and skew), and those stocks earn lower returns. MAX is the simplest,
  most transparent proxy for that lottery dimension.

## Why our tape can *invert* the published result — survivorship and universe

- **The published effect lives in the small-cap / illiquid tail.** Bali-Cakici-Whitelaw report
  the MAX premium is strongest among small, low-priced, high-idiosyncratic-vol names — exactly
  the segment a free large-cap feed excludes. We run the sort on a fixed **S&P-100-style
  large-cap basket** (yfinance adjusted closes), an explicit **proxy** that is the *least*
  lottery-prone slice of the market.
- **On survivors, high MAX flags the winners, not the losers.** A fixed surviving-large-cap
  basket carries **survivorship bias**, and here it points *against* the anomaly: a high MAX
  last month tags the high-beta growth names that subsequently led the 2009–2026 bull market.
  The desk's rule is to **name survivorship on the Signal axis** and reason about its direction
  explicitly (METHODOLOGY → *Survivorship is named on the Signal axis*). A current-membership /
  surviving panel can *invert* an anomaly outright — this study is a clean example, and it is
  why the Signal stamp is `NONE` (claimed edge absent), not `REAL`-of-the-opposite-sign.
- **MAX vs momentum/beta confound.** On large-caps, MAX correlates with short-horizon momentum
  and market beta; without thousands of small names to populate the genuinely "lottery" tail,
  the sort degenerates into a beta sort. Hou, Xue & Zhang, *Digesting Anomalies* (Review of
  Financial Studies, 2015), document that many cross-sectional anomalies weaken sharply once
  micro-caps are excluded — the same mechanism that flips MAX here.

## Why a significant *wrong-sign* spread is still `NONE` on Signal

- **The inference bar.** `REAL` is earned by an autocorrelation-robust statistic clearing
  **t = 2** *in the claimed direction* on the real tape (METHODOLOGY → *The inference bar*). A
  spread that is significant with the **opposite** sign does not certify the anomaly — it
  certifies its *failure* on this universe. We test the long-low / short-high mean with a
  **Newey-West (HAC) t-stat** (Newey & West, 1987, *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*) and a **sign-flip
  placebo** null (Fisher's randomization logic; Efron & Tibshirani, *An Introduction to the
  Bootstrap*, 1993).
- **Selection / multiple testing on a famous factor.** Harvey, Liu & Zhu (2016), *…and the
  Cross-Section of Expected Returns* (Review of Financial Studies), and McLean & Pontiff (2016),
  *Does Academic Research Destroy Stock Return Predictability?* — published anomalies decay
  (and sometimes reverse) out-of-sample and out-of-universe. MAX failing on large-cap survivors
  is consistent with both effects.

## Method lineage (the desk's shared engine)

- **Quintile sort + long-short.** [`strategy.quintile_returns`](../lottery_max_effect/strategy.py)
  ranks the cross-section by MAX each month and earns each quintile's next-month return (one
  execution lag, baked into the panel: month-*t* MAX pairs with month *t+1*'s return);
  [`strategy.long_short`](../lottery_max_effect/strategy.py) forms Q1 − Q5 net of one-way costs
  × turnover and short borrow.
- **Robust inference.** [`strategy.hac_tstat`](../lottery_max_effect/strategy.py) (Newey-West)
  and [`strategy.placebo_pvalue`](../lottery_max_effect/strategy.py) (sign-flip null) — the
  Signal-axis tests on the spread mean.
- **Deterministic synthetic control.**
  [`data.synthetic_panel`](../lottery_max_effect/data.py) plants a known lottery penalty
  (high-MAX names made to underperform next month); the offline core runs with no network. The
  control confirms the sort+inference recover a *real* low-minus-high edge when present and find
  **nothing** when the edge is zero — so the negative real-tape *t* is a genuine universe
  feature, not a coding artefact.

## Data sources used here

- **yfinance** daily adjusted closes for a fixed S&P-100-style 66-name large-cap basket,
  2005-01-03 → 2026-05-29, cached under `_cache/basket_prices.csv` (as-of 2026-05-31,
  fingerprint `5c0c1743c8d7`). All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 53 — Jackpot](../53-jackpot/)** and **[Study 43 — Free-Lunch](../43-free-lunch/)**:
  the high-beta / lottery tail winning in a bull-dominated large-cap sample is the same
  regime that lifts those studies' flashy legs.
- **[Study 330 — Low-Volatility-Anomaly](../330-low-volatility-anomaly/)**: the mirror
  question — calm beats wild *risk-adjusted* — and the same caveat that a 2011–2026 bull window
  flatters the high-beta leg.
- **[Study 238 — Betting-Against-Beta](../238-betting-against-beta/)**: the beta sort that the
  large-cap MAX sort collapses into once the genuine lottery tail (small, illiquid) is excluded.
