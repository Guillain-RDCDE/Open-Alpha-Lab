# References & literature map — Study 384 (ISM-PMI-Regime)

## The claim under test

- **The 50 line as a regime switch.** Market commentary routinely treats the ISM
  Manufacturing PMI's **50 level** as a binary switch for equities: a reading above 50
  signals manufacturing *expansion* (own stocks), below 50 signals *contraction* (raise
  cash). The diffusion-index construction makes 50 the natural pivot — it is the share of
  purchasing managers reporting improving conditions, so 50 is "as many up as down." The
  folklore extrapolates from "PMI leads the business cycle" to "PMI times the stock market."
- **What the survey actually is.** The **ISM Report On Business** (Institute for Supply
  Management; the PMI succeeded the older NAPM index) is a monthly *diffusion* of purchasing
  managers across new orders, production, employment, supplier deliveries and inventories.
  It is a **proprietary** product — ISM sells the data — which is why it is not on free
  feeds. The Federal Reserve and NBER treat it as a *coincident-to-slightly-leading*
  indicator of **real activity**, not as a predictor of asset returns.

## Why the true PMI is not available here — and what we do instead

- **Proprietary survey + blocked fallback.** The ISM PMI itself is paywalled. The standard
  free proxy is FRED (historical `NAPM`, or regional-Fed manufacturing diffusion indices
  such as the Empire State / Philadelphia / Dallas Fed surveys, or the Chicago Fed National
  Activity Index `CFNAI`). **FRED is unreachable from this execution environment** (every
  `fredgraph.csv` request times out), so we cannot pull even the proxy series. We therefore
  **construct a transparent PMI proxy from price data**: a monthly diffusion index across a
  fixed industrial/manufacturing basket — the share of names whose trailing 3-month return
  is positive, on a 0–100 PMI-like axis. This mirrors the survey's object (diffusion of
  improving "new orders" across manufacturers) using only public adjusted closes. It is a
  *narrower, noisier, market-priced* stand-in — named a proxy on the Signal axis — not the
  survey, and not a fabrication.
- **Diffusion indices generally.** Geoffrey H. Moore and the NBER tradition on diffusion
  indexes and coincident/leading indicators; the Conference Board's Leading Economic Index
  (which *includes* ISM new orders as a component) — the lineage that treats breadth of
  improvement as a business-cycle gauge.

## Does the PMI predict stock returns? — the evidence

- **PMI as a coincident growth gauge, not a return signal.** Koenig (2002, *Dallas Fed
  Economic & Financial Review*), *Using the Purchasing Managers' Index to Assess the
  Economy's Strength and the Likely Direction of Monetary Policy*, documents the PMI's tight
  link to GDP growth and Fed policy — its job is nowcasting *activity*, not forecasting
  *returns*. Equity markets are forward-looking and largely price the cycle in advance, so a
  coincident activity gauge has little left to forecast.
- **Macro state and the equity premium.** Cooper & Priestley (2009, *Review of Financial
  Studies*), *Time-Varying Risk Premiums and the Output Gap*, find macro/output-gap variables
  carry *some* equity-premium predictability in-sample — but the effect is weak,
  regime-dependent, and fragile out of sample. Welch & Goyal (2008, *RFS*), *A Comprehensive
  Look at the Empirical Performance of Equity Premium Prediction*, is the canonical caution:
  the vast majority of macro/valuation predictors fail to beat the historical mean out of
  sample. A binary PMI>50 switch is exactly the kind of predictor that looks compelling
  in-sample and adds nothing out of sample.
- **The 50 line is a coin-toss for returns.** Because equities drift up in *both* expansion
  and (most) contraction months, sorting forward returns on the PMI regime mostly recovers
  the unconditional mean. Any apparent gain from "own stocks only above 50" is dominated by
  the base rate (the market rises most months) and by the asymmetry that PMI is above 50 the
  majority of the time — so the rule is close to always-invested.

## Why a regime split needs careful inference — the statistics

- **Few, long regimes ⇒ inflated naive significance.** PMI regimes are *persistent*: the
  index spends long stretches above or below 50, so the effective number of independent
  regime observations is far smaller than the month count. We test the above-vs-below mean
  with a **Welch two-sample t** (Welch, 1947, *The generalization of "Student's" problem*)
  and, because the labels are autocorrelated, with a **block bootstrap / placebo** null
  (Künsch, 1989, *The jackknife and the bootstrap for general stationary observations*,
  *Annals of Statistics*; Politis & Romano, 1994, *The stationary bootstrap*) that resamples
  the regime labels in blocks instead of i.i.d.
- **Base rates and the win-rate illusion.** US equities rise most months *unconditionally*,
  so a high above-50 win-rate is *expected under the null*. The right comparison is the
  **spread** over the other regime / the base rate, not the raw win-rate — the classic
  base-rate fallacy (Kahneman & Tversky, 1973, *On the psychology of prediction*).
- **Less beta is not alpha.** A timing rule that sits in cash part of the time lowers
  exposure; comparing its raw return or Sharpe to fully-invested buy-and-hold without
  risk-matching is a category error. We label gross/net, charge one-way costs per switch,
  and read the Sharpe race and terminal wealth side by side.

## Method lineage (the desk's shared engine)

- **Welch t + block-placebo p-value.** [`strategy.welch_t`](../ism_pmi_regime/strategy.py)
  and [`strategy.block_bootstrap_pvalue`](../ism_pmi_regime/strategy.py) — the Signal-axis
  tests: above-50 vs below-50 monthly returns, and a 20,000-draw block-resampled
  regime-label null.
- **Deterministic synthetic control.**
  [`data.synthetic_regime`](../ism_pmi_regime/data.py) plants a *known* above-50-only return
  edge; the offline core runs with no network. The control confirms the engine is faithful
  *and* powered: edge=0 stays below t=2, a large edge lights it up.
- **Deployable timing rule with execution lag + costs.**
  [`strategy.timing_strategy`](../ism_pmi_regime/strategy.py) acts on the prior month's PMI
  (one-month lag, no look-ahead), charges one-way cost per regime switch, and races the rule
  against buy-and-hold on Sharpe, drawdown and terminal wealth.

## Data sources used here

- **yfinance** daily adjusted closes for SPY + a fixed 37-name industrial/manufacturing
  basket, 1995-01 → 2026-06, cached under `_cache/manuf_prices.csv`. All headline numbers
  are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 118 — Fed-Model](../../118-fed-model/)**: another "macro level predicts equity
  returns" claim (earnings-yield-minus-rate). Same lesson, different gauge — a macro level
  that *feels* like it should time the market mostly recovers the base rate.
- **[Study 120 — Excess-CAPE-Yield](../../120-excess-cape-yield/)**: a valuation/real-rate
  predictor that carries *some* long-horizon signal — the contrast that shows what a macro
  predictor with a (fragile) real edge looks like, versus a coincident gauge that carries none.
