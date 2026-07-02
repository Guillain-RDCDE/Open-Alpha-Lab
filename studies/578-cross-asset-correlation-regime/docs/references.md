# References & literature map — Study 578 (Cross-Asset-Correlation-Regime)

## The claim, at full strength

- **The practitioner folklore / "correlation goes to one in a crisis."** The widely repeated desk
  maxim that in a stress event *all correlations go to one* — diversification vanishes exactly when
  you need it. Popularised across risk-management and multi-asset commentary; the fragility reading
  adds that a *rising* average correlation is a **leading** warning that a break is near.
- **Longin & Solnik (2001)**, *"Extreme Correlation of International Equity Markets."* *Journal of
  Finance* 56(2). The canonical evidence that cross-market correlation rises in *bear* markets (and
  not symmetrically in bull markets) — correlations spike with downside stress. The empirical seed
  of "correlations go to one in a crash."
- **Ang & Chen (2002)**, *"Asymmetric Correlations of Equity Portfolios."* *Journal of Financial
  Economics* 63(3). Downside correlations exceed upside correlations — co-movement clusters in
  falling markets, the mechanism behind the fragility gauge.
- **Pollet & Wilson (2010)**, *"Average Correlation and Stock Market Returns."* *Journal of
  Financial Economics* 96(3). The closest academic test of the *predictive* claim: average
  correlation among stocks forecasts market returns and variance. Their result is a key contrast for
  this study — the *sign* and the *asset scope* (cross-equity vs cross-asset) matter.
- **Driessen, Maenhout & Vilkov (2009)** / the **implied-correlation** literature. Option-implied
  average correlation as a risk factor and a stress gauge — the forward-looking cousin of the
  realised measure we build.

## The measure we build

- **Average pairwise correlation.** Over a trailing 63-day window we compute the full pairwise
  correlation matrix of a 14-ETF cross-asset panel's daily returns and take the mean of its
  off-diagonal entries — one *average cross-asset correlation* per day. The HIGH/LOW regime is an
  expanding-quantile split (past data only, no look-ahead). This is the realised-correlation
  fragility index in its simplest cross-asset form.
- **The regime forward test.** A two-sample (Welch) *t* on the risk asset's forward return and
  forward realised volatility across the HIGH vs LOW regimes; the *sign* of the return spread is the
  claim (fragility predicts a *negative* HIGH−LOW forward return).

## Neighbours on this bench (the dedup map)

- **[Study 245 — Oil-Equity-Correlation](../../245-oil-equity-correlation/)** — the *single-pair*
  rolling correlation of oil vs equities as a signal. Study 578 is the **panel-average** correlation
  across many asset classes read as a *regime/fragility* gauge, not a single pair.
- **[Study 502 — Betting-Against-Correlation](../../502-betting-against-correlation/)** — a
  *cross-sectional* stock sort on each name's correlation-to-market (the AFGP low-risk decomposition).
  Study 578 is a **time-series regime** from the panel-wide average correlation, tested as a forward
  drawdown/vol predictor, not a cross-sectional sort.
- **[Study 349 — Regime-Dependence](../../349-regime-dependence/)** /
  **[Study 384 — ISM-PMI-Regime](../../384-ism-pmi-regime/)** — other regime-conditioning studies.
  578 shares the regime-split machinery but the conditioning variable (average cross-asset
  correlation) and the fragility claim are its own.
- **[Study 330 — Low-Volatility-Anomaly](../../330-low-volatility-anomaly/)** — a low-risk ETF race.
  Related in spirit (risk regimes) but a different signal.

## Shared method

- **Welch (1947)** — the unequal-variance two-sample *t* used for the HIGH vs LOW forward-return and
  forward-vol spreads.
- **Block bootstrap / block permutation** (Künsch 1989; Politis & Romano 1994) — the placebo null:
  the 21-day forward windows overlap and are serially correlated, so we shuffle the regime labels in
  21-day blocks (preserving the forward series' autocorrelation) and read the spread's tail
  probability, rather than trusting the overlap-inflated *t*.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a robust *t* ≥ 2
  on the *real* tape for `REAL`, plus a placebo null and a seed-robust synthetic control), the
  data-span caveat named on the SIGNAL axis, one documented execution lag, and costs one-way × NAV.
