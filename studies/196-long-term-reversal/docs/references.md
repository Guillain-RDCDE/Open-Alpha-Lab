# References & literature map -- Study 196 (Long-Term-Reversal)

## The canonical claim

- **De Bondt, W. F. M. & Thaler, R. H. (1985).** *Does the Stock Market Overreact?*
  Journal of Finance, 40(3), 793--805.
  The founding paper: stocks ranked in the bottom decile by 3--5 year prior returns
  (the "loser portfolio") subsequently beat the top decile ("winners") by ~25
  percentage points over the next 3 years on the NYSE. Attributed to investor
  overreaction to long sequences of good/bad news, leading to systematic price
  overshooting that later corrects.

- **De Bondt, W. F. M. & Thaler, R. H. (1987).** *Further Evidence on Investor
  Overreaction and Stock Market Seasonality.* Journal of Finance, 42(3), 557--581.
  Confirms the 1985 result with extended sample; documents January seasonality in
  the reversal (most of the loser-portfolio return occurs in January).

## Why the steelman is almost right -- the real economics

- **Fama, E. F. & French, K. R. (1996).** *Multifactor Explanations of Asset Pricing
  Anomalies.* Journal of Finance, 51(1), 55--84.
  Shows that long-horizon reversal is largely explained by the three-factor model:
  loser stocks load more heavily on the HML (value) and SMB (size) factors. Much of
  the "overreaction" premium is a disguised small-cap/value bet, not a distinct
  behavioural alpha. This is why our beta decomposition finds loser-portfolio beta = 1.31
  and near-zero alpha after adjusting for the market alone.

- **Chopra, N., Lakonishok, J. & Ritter, J. R. (1992).** *Measuring Abnormal Performance:
  Do Stocks Overreact?* Journal of Financial Economics, 31(2), 235--268.
  Confirms the overreaction result but finds the abnormal returns are concentrated in
  January and in smaller-cap stocks; after accounting for size and January the effect
  weakens materially.

## Post-publication decay and risk explanations

- **McLean, R. D. & Pontiff, J. (2016).** *Does Academic Research Destroy Stock Return
  Predictability?* Journal of Finance, 71(1), 5--32.
  Documents that anomaly returns decay on average ~58% after publication as arbitrageurs
  trade against them. The LTR effect, published in 1985, is a canonical example: it was
  economically large in the original sample (1926--1985) and has become substantially
  weaker or absent in post-publication periods (our sub-period analysis: t = +2.33 in
  1993--2005 vs t = +0.76 in 2016--2026).

- **Conrad, J. & Kaul, G. (1993).** *Long-Term Market Overreaction or Biases in
  Computed Returns?* Journal of Finance, 48(1), 39--63.
  Argues that much of the apparent long-horizon reversal in the De Bondt-Thaler
  framework is a data artefact: cumulating single-period returns with bid-ask bounce
  creates a mechanical negative autocorrelation in multi-period returns. A significant
  fraction of the "loser-minus-winner" spread may be spurious.

- **Ball, R. & Kothari, S. P. (1989).** *Nonstationary Expected Returns: Implications
  for Tests of Market Efficiency and Serial Correlation in Returns.* Journal of
  Financial Economics, 25(1), 51--74.
  Demonstrates that beta changes over the sample period: losers become riskier (higher
  beta) and winners become safer after their return extreme. The apparent reversal
  premium is partly compensation for this time-varying risk, not alpha.

## The survivorship-bias dimension

- **Shumway, T. (1997).** *The Delisting Bias in CRSP Data.* Journal of Finance,
  52(1), 327--340.
  Delisted stocks (failed, acquired, or force-removed) earn extreme negative returns
  but are often excluded from standard data vendors. The true "loser" portfolio in
  the wild contains many firms on the way to delisting; their returns are missing in
  survivorship-biased panels (like the current S&P 500 universe we use), biasing
  the loser-portfolio return upward.

## Related desk studies

- **[Study 24 -- Stampede](../../24-stampede/)**: 12-1 month momentum, the
  short-horizon counterpart. Momentum and long-term reversal co-exist: momentum
  is positive at 1--12 months and negative (reversal) at 36--60 months. Same
  cross-sectional engine, opposite sign, different horizon.
- **[Study 33 -- Slingshot](../../33-slingshot/)**: short-term reversal (1-month),
  the extreme short end of the return-autocorrelation horizon spectrum.
- **[Study 121 -- Magic-Formula](../../121-magic-formula/)**: quality + cheapness
  factor, a related EDGAR-fundamentals-based screen using the same
  cross-sectional quintile machinery.
- **[Study 65 -- Scorecard](../../65-scorecard/)** and
  **[Study 153 -- Net-Operating-Assets](../../153-net-operating-assets/)**: other
  EDGAR-based cross-sectional signals sharing the quintile-return infrastructure.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance
  Matrix* (Econometrica) -- [`strategy.summarize`](../long_term_reversal/strategy.py)
  and [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Block-bootstrap Sharpe CI.** Politis & Romano (1994), *The Stationary Bootstrap*
  (JASA) -- [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).
- **Survivorship notation.** Shumway (1997) -- named on the Signal axis.
