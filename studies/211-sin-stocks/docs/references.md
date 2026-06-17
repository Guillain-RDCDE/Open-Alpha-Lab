# References & literature map — Study 211 (Sin-Stocks)

## The claim under test

"Sin stocks" — equities in tobacco, alcohol, gambling, weapons/defense, and sometimes
adult entertainment — are systematically excluded or underweighted by institutional
investors bound by ESG/SRI mandates or public-pressure policies.  The neglect premium
hypothesis holds that this exclusion depresses valuations relative to fundamentals,
mechanically raising expected returns for investors indifferent to the moral cost.  The
flip-side is that "virtue" (ESG) portfolios should underperform as the premium accrues
to the neglected segment.

## Why the claim is plausible — the theoretical foundation

- **Neglect premium / investor exclusion.** Hong & Kacperczyk (2009), *The price of sin:
  the effects of social norms on markets* (Journal of Financial Economics), is the
  foundational empirical paper.  Using US stocks from 1965–2006, they show that sin stocks
  trade at lower multiples and earn excess returns of approximately 2.5% per year after
  controlling for common risk factors (Fama-French + momentum).  They attribute this to
  limited arbitrage: social norms prevent institutions from holding sin stocks, and the
  resulting neglect creates a persistent valuation discount.

- **Sin stocks as low-beta defensives.** Tobacco, beer, and defense companies share a
  "defensive" demand profile — cigarettes, beer, and weapons contracts are largely
  recession-proof.  The CAPM would predict lower expected returns for low-beta assets; the
  neglect premium hypothesis predicts a countervailing return boost.  In the original
  Hong-Kacperczyk sample, the neglect effect dominates.

- **Sin as value in disguise.** Blitz & Fabozzi (2017), *Sin stocks revisited: resolving
  the sin stock anomaly* (Journal of Portfolio Management), argue that the sin stock premium
  disappears once standard value and profitability exposures are properly controlled.  This
  reframing shifts the question: the "premium" may simply be compensation for value/quality
  exposure, not for social-norm neglect.

- **Underinvestment hypothesis.** Derwall, Koedijk & Ter Horst (2011), *A tale of values-
  driven and profit-seeking social investors* (Journal of Banking and Finance), propose that
  the ESG market has expanded sufficiently since 2000 that the original neglect premium has
  been largely arbitraged away by capital flowing into ESG exclusion strategies.

## Why the backtest may disappoint

- **Institutional ownership normalised.** Since Hong-Kacperczyk published in 2009, ESG
  AUM has grown from ~$5tn to >$40tn globally, dramatically increasing the share of
  capital excluding sin stocks.  McLean & Pontiff (2016), *Does academic research destroy
  stock return predictability?* (Journal of Finance), document that anomalies decay by
  ~32% after publication — the neglect premium, if real, would be expected to compress.

- **Basket heterogeneity.** The six-stock equal-weight basket (MO, PM, STZ, LVS, LMT, RTX)
  is internally heterogeneous.  LVS (Las Vegas Sands) is a high-vol speculative casino play
  that fell −98.3% peak-to-trough; tobacco and defense stocks are stable income compounders.
  Bundling them as "sin" is conceptually coherent but empirically noisy.

- **Regime-dependence.** The basket's excess return is strongly regime-dependent: it
  outperformed in value-led markets (2009–2017) and dramatically underperformed in
  growth-led markets (2018–2021, 2023).  The HAC t-stat on the full period is −0.46,
  reflecting the near-cancellation of two opposing regimes.

- **No sector-neutral framing.** The sin basket is heavily tilted toward consumer staples
  (tobacco), consumer discretionary (alcohol, gambling), and industrials (defense).
  Any period in which technology mega-caps dominate the S&P 500 will mechanically penalise
  this basket regardless of "sinfulness."

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Sharpe bootstrap CI.** Politis & Romano (1994), *The Stationary Bootstrap* (JASA) —
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).
- **OLS alpha/beta.** Standard OLS decomposition —
  [`quantlab.stats.beta_decomposition`](../../../quantlab/stats.py).

## Data sources used here

- **Yahoo Finance (via yfinance)** — daily adjusted total-return closes for MO, PM, STZ,
  LVS, LMT, RTX (sin basket), SPY (market benchmark), and DSI (iShares MSCI KLD 400
  Social ETF, the ESG counter-portfolio).  PM's IPO date (2008-03-17) determines the start
  of the common sample.  `auto_adjust=True` folds in dividends — essential for tobacco and
  defense stocks, which have historically returned large fractions of earnings as dividends.

## Related desk studies

- **[Study 88 — Dogs-of-the-Dow](../../88-dogs-of-the-dow/)**: high-dividend-yield selection
  — tobacco and defense stocks frequently appear here; a natural overlap.
- **[Study 206 — Dividend-Aristocrats](../../206-dividend-aristocrats/)**: quality/dividend-
  growth ETF (NOBL) — similar value/defensive tilt, similar underperformance vs SPY.
- **[Study 201 — Dividend-Growth](../../201-dividend-growth/)**: a factor approach to
  dividend-growth stocks — tobacco stocks are canonical dividend-growers.
- **[Study 144 — Permanent-Portfolio](../../144-permanent-portfolio/)**: another defensive
  multi-asset allocation that succeeded where this sector tilt has not.
