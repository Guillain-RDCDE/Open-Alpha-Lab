# References & literature map — Study 367 (Closed-End Fund Discount)

## The claim under test

- **The folklore.** Closed-end funds (CEFs) issue a fixed number of shares and trade on an
  exchange like a stock, so their market **price** routinely diverges from the **net asset
  value (NAV)** of their holdings. A fund priced below NAV is "at a discount." The popular
  pitch — repeated by income-investing newsletters, CEF screeners and brokers — is that
  buying the **widest discounts** pays you to wait: the discount mean-reverts toward NAV, so
  you collect both the underlying return *and* the discount's narrowing.
- **The academic strand.** This is not pure folklore — there is a real literature. The
  **closed-end fund discount** is one of finance's classic puzzles, and several studies find
  that the *level* and *changes* of the discount carry predictive information for fund
  returns. We test the strongest, most tradable version: a monthly cross-sectional sort that
  buys wide-discount funds and (optionally) shorts narrow-discount / premium funds.

## Why true NAV is not on yfinance — and what we do instead

- **NAV data.** yfinance serves per-ticker market-price OHLCV only; it does **not** carry each
  CEF's daily reported NAV (the canonical inputs live on fund-sponsor sites, Morningstar, or
  CEF data vendors). We therefore build a **transparent NAV proxy**: each CEF is mapped to a
  published benchmark ETF for its stated mandate (`BENCH_MAP` in
  [`data.py`](../closed_end_fund_discount/data.py) — e.g. broad US-equity CEFs → SPY, a
  small-cap CEF → IWM, a utilities CEF → XLU, an energy CEF → XLE, a health CEF → XLV). The
  proxy discount is `log(price) − log(benchmark)`, demeaned per fund: a fund whose price has
  drifted *below* its mandate's path is "cheap" in proxy terms. This is a *narrower, noisier*
  stand-in for the reported price/NAV discount — labelled a proxy throughout — and it absorbs
  any genuine NAV-vs-benchmark tracking error, which we flag on the Signal axis.

## The closed-end fund discount literature

- **The puzzle.** Lee, Shleifer & Thaler (1991), *Investor Sentiment and the Closed-End Fund
  Puzzle* (Journal of Finance) — discounts co-move and are linked to retail sentiment; a
  foundational behavioural account. Earlier: Zweig (1973) and Malkiel (1977) on discounts as
  predictors. Pontiff (1996), *Costly Arbitrage and the Myth of Idiosyncratic Risk* (QJE) —
  why discounts persist (arbitrage is costly and risky).
- **Discount as a return predictor.** Pontiff (1995), *Closed-end fund premia and returns*
  (Journal of Financial Economics) finds the discount forecasts fund returns; this is exactly
  the "buy the discount" claim we sort on. Anderson, Born & Schnusenberg, *Closed-End Funds,
  Exchange-Traded Funds, and Hedge Funds* (Springer, 2010) surveys the discount-reversion
  evidence and its fragility once costs and structure are imposed.
- **Why it's hard to bank.** Cherkes, Sagi & Stanton (2009), *A Liquidity-Based Theory of
  Closed-End Funds* (Review of Financial Studies) — the discount partly compensates for
  illiquid underlying holdings, so it is not a free lunch. The short leg (narrow-discount /
  premium CEFs) is small and **hard to borrow**; the funds are thin; capacity is tiny.

## Why a positive in-sample *t* still needs caveats — the statistics

- **Survivorship on a fixed basket.** A current-membership CEF panel keeps only funds that
  *did not liquidate*. Funds that stayed deeply, permanently cheap and were wound down are
  absent — which biases a discount-**reversion** test *toward* finding reversion. Per the
  desk's house rule, survivorship is named on the **Signal** axis, not buried in Tradability:
  it can support *existence* but inflates *magnitude* (Brown, Goetzmann, Ibbotson & Ross,
  1992, *Survivorship Bias in Performance Studies*, RFS).
- **Robust inference + persistence.** The monthly long-short series is autocorrelated, so the
  null is a **circular block bootstrap** (6-month blocks) rather than i.i.d. resampling
  (Politis & Romano, 1994). We also run a **placebo** that shuffles the discount labels across
  funds each month — a real signal must die under the shuffle (it does). Welch (1947) for the
  unequal-variance *t*.
- **Decay / selection on a famous anomaly.** Anomalies decay after publication (McLean &
  Pontiff, 2016, *Does Academic Research Destroy Stock Return Predictability?*, JF) — which is
  exactly what our 1995–2010 vs 2011–2026 split shows: *t* = 3.4 → *t* = 1.56.

## Method lineage (the desk's shared engine)

- **Discount sort + Welch t + block-bootstrap null.**
  [`strategy.long_short_returns`](../closed_end_fund_discount/strategy.py),
  [`strategy.welch_t_vs_zero`](../closed_end_fund_discount/strategy.py) and
  [`strategy.block_bootstrap_p`](../closed_end_fund_discount/strategy.py).
- **Deterministic synthetic control.**
  [`data.synthetic_panel`](../closed_end_fund_discount/data.py) builds a mean-reverting
  AR(1) discount panel with a **planted edge** knob; the offline core runs with no network.
  The control confirms the harness banks a planted edge and finds nothing when the discount
  carries no return information.
- **Execution lag + costs.** One-month lag (discount at close of *t* earns *t+1*) in
  [`strategy.monthly_panels`](../closed_end_fund_discount/strategy.py); one-way costs ×
  turnover in [`strategy.net_of_costs`](../closed_end_fund_discount/strategy.py).

## Data sources used here

- **yfinance** daily adjusted closes for 18 long-listed US equity CEFs + benchmark ETFs
  (SPY, IWM, QQQ, XLU, XLE, XLV), 1995-01-03 → 2026-06-18, cached under
  `_cache/cef_prices.csv`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).
