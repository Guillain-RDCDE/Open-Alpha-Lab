# References & literature map — Study 392 (Glassdoor-Sentiment)

## The claim under test

- **The academic anchor (Alex Edmans).** Edmans, A. (2011), *Does the stock market fully value
  intangibles? Employee satisfaction and equity prices*, **Journal of Financial Economics**
  101(3), 621–640. A value-weighted portfolio of the *"100 Best Companies to Work For in
  America"* (Fortune list) earned an abnormal return of ~2–4% per year over 1984–2009, robust
  to risk-factor and characteristic controls. Follow-up: Edmans, Li & Zhang (2014/2024),
  *Employee Satisfaction, Labor Market Flexibility, and Stock Returns Around the World* — the
  premium concentrates in flexible labour markets. This is the *steelman* of the folk claim.
- **The retail folklore.** The popular descendant — *"buy the companies with the best
  Glassdoor ratings and you'll beat the market"* — drops Edmans' careful list construction and
  risk adjustment for a raw star-rating sort. Glassdoor's own and third-party "best places to
  work" backtests circulate widely; whether the *retail* version survives costs, the right base
  rate, and out-of-sample testing is the open question.
- **The alt-data context.** Employee-review platforms (Glassdoor, Indeed, Comparably) are a
  canonical "alternative data" set: Green, Huang, Wen & Zhou (2019), *Crowdsourced employer
  reviews and stock returns*, **Journal of Financial Economics** 134(1), 236–251 — *changes* in
  Glassdoor ratings predict returns and earnings surprises, more than levels do.

## Why we cannot use the real ratings here — and what we do instead

- **Employer-review data is not free.** Glassdoor / Indeed ratings sit behind paid APIs and
  licensing; the historical panels used in the papers above are proprietary or hand-collected.
  The free yfinance endpoint serves per-ticker OHLCV only — no satisfaction data.
- **So we CONSTRUCT a transparent, clearly-labelled proxy.**
  [`data.constructed_sentiment`](../glassdoor_sentiment/data.py) assigns each name a fixed-seed
  1.0–5.0 "stars" score **independent of returns**. This is *not* a fabrication of real ratings:
  it is an explicit placeholder that lets us demonstrate the *method* (a leak-free happiness
  long-short with a placebo null) on an honest null, and it is named a proxy throughout. A real
  ratings feed can replace it wholesale. The genuine inputs (prices) are public adjusted closes.

## The method — a cross-sectional long-short and its honest tests

- **Quintile sort / long-short.** Sort names by the score, go long the happiest quintile and
  short the grumpiest; the spread is the factor return — the standard Fama-French style
  portfolio sort (Fama & French, 1992/1993). [`strategy.long_short_returns`](../glassdoor_sentiment/strategy.py).
- **Inference for a single factor.** A one-sample *t* of the mean monthly spread vs zero
  ([`strategy.t_stat`](../glassdoor_sentiment/strategy.py)); an i.i.d. **bootstrap** CI for the
  annualised Sharpe (Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993,
  [`strategy.bootstrap_sharpe_ci`](../glassdoor_sentiment/strategy.py)); and a **relabelling /
  placebo** null — shuffle which names are "happy" vs "grumpy" and ask how often a random
  labelling matches the real spread (Fisher's randomization logic,
  [`strategy.placebo_pvalue`](../glassdoor_sentiment/strategy.py)).
- **Multiple testing on a famous result.** A premium discovered ex-post and amplified by
  alt-data marketing needs a higher bar than a naive *t*: Harvey, Liu & Zhu (2016), *…and the
  Cross-Section of Expected Returns* (Review of Financial Studies); Bailey & López de Prado
  (2014), *The Deflated Sharpe Ratio*. The desk's bar is a robust *t* ≥ 2 on the **real** tape.

## Method lineage (the desk's shared engine)

- **One-period execution lag + one-way costs × turnover + short borrow.**
  [`strategy.net_of_costs`](../glassdoor_sentiment/strategy.py) — the Tradability axis.
- **Deterministic synthetic control with a planted edge.**
  [`data.synthetic_panel`](../glassdoor_sentiment/data.py) injects a known happiness→return
  tilt; with `edge = 0` the long-short must stay below *t* = 2, with a large `edge` it must light
  up. Proves the engine is faithful and the real-tape null is honest. The offline core runs with
  no network.

## Data sources used here

- **yfinance** daily adjusted closes for a fixed 40-name US large-cap basket, 2005-01-03 →
  2026-06-18, cached under `_cache/basket_prices.csv`. The sentiment input is the **constructed
  proxy** (no external feed). All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 257 — AAII-Sentiment](../257-aaii-sentiment/)**: investor (not employee) sentiment as
  a return predictor — the adjacent "does mood move prices?" question.
- **[Study 252 — Google-Trends](../252-google-trends/)**: another crowd-sourced alt-data signal
  tested for tradable content beyond the base rate.
- **[Study 335 — BUZZ-Sentiment-ETF](../335-buzz-sentiment-etf/)**: a packaged social-sentiment
  factor — the productised cousin of the happiness sort.
