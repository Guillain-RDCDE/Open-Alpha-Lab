# References & literature map — Study 85 (Dr-Copper)

## The claim under test

- **"Dr. Copper has a PhD in economics."** The copper/gold price ratio is widely cited in
  macro commentary as a real-time indicator of economic growth expectations: copper is an
  industrial metal (demand rises with growth) while gold is a safe-haven asset (demand
  rises in risk-off periods). The ratio therefore reflects the market's aggregate growth
  vs risk-off stance. The claim, popularised by Jeffrey Gundlach (DoubleLine, 2017–2018
  CNBC appearances) and embedded in Bloomberg macro terminals, extends further: that the
  ratio *predicts* (leads) equity returns and 10-year Treasury yields, not merely moves
  with them. We take the strongest version — a rolling regression of *lagged* ratio change
  on *forward* equity and yield returns — and test it honestly.

## Why the steelman is coherent — the real economics behind the claim

- **Copper as a growth proxy.** Copper is consumed in construction, manufacturing, and
  electrical infrastructure; futures prices aggregate global demand expectations. Ye, Nix &
  Macaulay (2019), *Commodity Prices and Economic Activity: A Review* (IMF Working Paper),
  document the leading-indicator properties of industrial metals for OECD activity. The
  copper/gold *ratio* controls for the general commodity price level and the inflation/risk
  premium embedded in gold, sharpening the growth signal.
- **Bond-yield expectations channel.** A rising ratio signals faster growth → higher
  inflation expectations → higher nominal bond yields. Gundlach's original chart
  (2017) showed the Cu/Au ratio closely tracking the 10y Treasury yield level, and the
  relationship is visually compelling over 2002–2020. Haubrich, Pennacchi & Ritchken
  (2012), *Inflation Expectations, Real Rates, and Risk Premia: Evidence from Inflation
  Swaps* (RFS), document how real rates embed growth forecasts.
- **Equities and the growth-rate of growth.** If the ratio anticipates earnings growth, it
  should also lead equity returns — especially via the discount-rate channel (higher yields
  → lower multiples, so a rising-ratio environment is ambiguous for equities). This is the
  key tension the study exploits: the contemporaneous link is real, the directional
  *predictive* link is empirically absent.

## Why the predictive claim fails — known evidence

- **Contemporaneous vs predictive: the Goyal-Welch critique.** Goyal & Welch (2008),
  *A Comprehensive Look at the Empirical Performance of Equity Premium Prediction*
  (Review of Financial Studies) — the canonical reference for showing that macro variables
  (earnings yield, term spread, etc.) that look predictive in-sample regularly fail
  out-of-sample. The OOS R² framework used in this study is their exact methodology.
  Our OOS R² of −0.3% weekly confirms the standard Goyal-Welch finding: the historical
  mean beats the model out of sample.
- **The coincident vs leading problem.** Gorton & Rouwenhorst (2006), *Facts and Fantasies
  about Commodity Futures* (FAJ), show that commodities are procyclical but their
  predictive horizon is very short. Most of the "predictability" found in rolling-window
  IS regressions is driven by the contemporaneous growth-cycle, not genuine forecasting
  lead time. Our study's decomposition (contemporaneous R² = 12.2% vs predictive R² =
  0.45%) quantifies this precisely.
- **Data-snooping and narrative construction.** Harvey, Liu & Zhu (2016), *... and the
  Cross-Section of Expected Returns* (RFS), caution that visually compelling macro
  indicator stories that survive in-sample often fail once properly cross-validated.
  The copper/gold ratio's visual fit with 10y yields over 2010–2020 is heavily influenced
  by two extreme cycles (GFC recovery, COVID) — our 25-year OOS test corrects for this.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.regression_is`](../dr_copper/strategy.py) uses inline Bartlett-kernel HAC
  to correct for weekly return autocorrelation.
- **OOS R² (Goyal-Welch).** Goyal & Welch (2008), *A Comprehensive Look at the Empirical
  Performance of Equity Premium Prediction* (Review of Financial Studies) — the expanding-
  window OOS R² in [`strategy.oos_r2`](../dr_copper/strategy.py) follows their exact
  construction.
- **Diebold-Mariano test.** Diebold & Mariano (1995), *Comparing Predictive Accuracy*
  (JBES); Clark & West (2007), *Approximately Normal Tests for Equal Predictive Accuracy
  in Nested Models* (JE) — the DM t-stat on forecast-error differences provides
  formal inference on the OOS comparison.
- **Reproducibility stamp.** [`quantlab/repro.py`](../../../quantlab/repro.py) — the as-of
  freeze and content fingerprint each headline run carries.

## Data sources used here

- **Yahoo! Finance daily closes** (via `yfinance`): HG=F (copper front-month), GC=F (gold
  front-month), ^GSPC (S&P 500 index), ^TNX (10y US Treasury yield). Daily history from
  2000-08-30 to 2026-06-12 (~25 years, 1,344 weekly periods). Every headline is pinned
  with an as-of date and a content fingerprint (see [`docs/results.md`](results.md)). The
  test-suite runs entirely on the deterministic [`data.synthetic_daily`](../dr_copper/data.py)
  generator, never the network.

## Related desk studies

- **[Study 67 — Fed-Drift](../../67-fed-drift/)**: macro event-window study — the same
  "macro signal → equity return" family, with FOMC timing as the event.
- **[Study 70 — Digital-Gold](../../70-digital-gold/)**: gold vs Bitcoin as safe-haven
  assets — the gold side of the Cu/Au ratio, tested separately.
- **[Study 42 — Last-Call](../../42-last-call/)** and **[Study 48 — Groundhog](../../48-groundhog/)**:
  calendar and seasonality predictors — the same Goyal-Welch OOS framework applied to
  time-calendar signals.
- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: the golden cross on daily bars —
  another technically-motivated macro-timing rule that fails the OOS test.
