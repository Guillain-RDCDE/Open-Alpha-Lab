# References & literature map — Study 94 (Level-Pegging)

## The claim under test

The folk version, stated at full strength: *"equal-weighting **always** beats cap-weighting.
The equal-weight S&P (RSP) outperforms the cap-weight S&P (SPY) because it tilts toward
smaller names and harvests a mechanical **rebalancing premium** — sell what went up, buy what
went down. Just buy RSP and forget cap-weighting."*

- Index marketing: S&P Dow Jones Indices, *S&P 500 Equal Weight Index* methodology and fact
  sheets, which foreground the long-run outperformance of the EWI over the cap-weighted S&P
  500: <https://www.spglobal.com/spdji/en/indices/equity/sp-500-equal-weight-index/>
- Invesco's **RSP** (S&P 500 Equal Weight ETF, inception 2003-04-24) product page sells the
  same story of diversification away from mega-cap concentration.

## Why the steelman is almost coherent

- **Plyakha, Uppal & Vilkov, *Equal or Value Weighting? Implications for Asset-Pricing
  Tests*** (2012 working paper; 2014/2015 versions): an equal-weighted portfolio earns a
  higher *total* and risk-adjusted return than its value- and price-weighted counterparts over
  1962-2012, and the authors decompose the excess into a systematic part (higher exposure to
  market, size and value factors) and an **alpha** they attribute to the contrarian
  *rebalancing* back to equal weights.
- **The size and value premia are real and documented** (Fama & French, *The Cross-Section of
  Expected Stock Returns*, JF 1992; *Common Risk Factors*, JFE 1993). Equal-weighting is a
  mechanical small-cap-and-value tilt, so it is not pure superstition — it is a factor bet.
- **Rebalancing a diversified basket back to fixed weights** does harvest a small premium from
  mean-reverting idiosyncratic moves (Willenbrock, *Diversification Return, Portfolio
  Rebalancing, and the Commodity Return Puzzle*, FAJ 2011).

## Why it is likely to fail *as stated* ("always beats")

- The tilt is a **bet on market breadth**. When leadership is broad and small caps lead, the
  size/value tilt pays; when a handful of mega-caps drive the index — as in 2015-2024 — the
  cap-weight index lets its winners compound while equal-weight keeps *selling* them at each
  rebalance. "Always" is the word that cannot survive a regime split.
- The published EW alpha is computed on **price-weighted CRSP indices, gross of costs**. The
  investable vehicle (RSP) carries a higher expense ratio than SPY and rebalances quarterly,
  so its **turnover** and fee drag eat into a thin gross edge.
- Sub-period claims ("EW outperformed since 1990") are vulnerable to **start-date and regime
  selection**; a difference that flips sign across a justified split is not "always".

## Method lineage

- **Newey–West HAC standard errors** for the mean of an autocorrelated return difference and
  for the alpha of the EW-on-CW regression: Newey & West (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*,
  Econometrica.
- **Circular block bootstrap** for the difference-of-differences across the regime split
  (preserving the volatility clustering that i.i.d. resampling destroys): Politis & Romano
  (1992/1994), *The Stationary Bootstrap*, JASA; Künsch (1989), *The Jackknife and the
  Bootstrap for General Stationary Observations*.

## Data sources used

- **RSP** (Invesco S&P 500 Equal Weight ETF, inception 2003-04-24) and **SPY** (SPDR S&P 500
  ETF), daily, **total-return adjusted** (dividends folded in) via `quantlab.data` (Yahoo
  Finance), cached to parquet under `_cache/`. Total return is the only fair basis for an
  EW-vs-CW race. The window is **RSP-bounded from 2003** — there is no investable equal-weight
  S&P tape before RSP. A 0% cash rate nets out of the return *difference*, so it does not move
  the relative verdict.

## Related desk studies

- [Study 18 — Dull-Roar](../../18-dull-roar/) — the low-volatility anomaly as a defensive
  *tilt* rather than free alpha; the same "is it a factor bet or skill?" question.
- [Study 20 — Freight-Train](../../20-freight-train/) — a real but thin, regime-dependent
  premium and the bar a tilt has to clear.
- [Study 91 — Death-Cross](../../91-death-cross/) — another "always beats buy-and-hold" claim
  that turns out to be exposure, not skill.
