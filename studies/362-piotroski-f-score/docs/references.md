# References & literature map — Study 362 (Piotroski F-Score)

## The claim under test

- **The rule (Joseph Piotroski).** Piotroski, J. (2000), *Value Investing: The Use of
  Historical Financial Statement Information to Separate Winners from Losers*, **Journal of
  Accounting Research** 38 (Supplement), 1–41. Piotroski defines a **9-point F-Score** — four
  profitability signals (ROA > 0, CFO > 0, ΔROA > 0, accruals: CFO > net income), three
  leverage/liquidity/funding signals (ΔLeverage < 0, ΔCurrentRatio > 0, no new shares), and two
  operating-efficiency signals (ΔGrossMargin > 0, ΔAssetTurnover > 0). He shows that **within the
  universe of high book-to-market (value) firms**, going long the high-F-score names (8–9) and
  short the low-score names (0–1) earned a large annual spread (~23%/yr in his 1976–1996 sample),
  concentrated in **small, thinly-traded, low-analyst-coverage** firms.
- **The folklore.** The F-Score escaped its original "within deep-value, small-cap" boundary and
  is now repeated across screeners and financial media as a general-purpose "winners vs losers"
  health check for *any* stock — the framing this study tests on a large-cap basket.

## Why the boundary conditions matter — and what we do

- **Piotroski's edge is a value/small-cap/illiquidity effect, not a large-cap one.** The original
  result lives among high book-to-market microcaps with low analyst coverage — precisely where
  fundamental information is slow to be priced. Subsequent work confirms the screen adds most
  *inside* a value sleeve and *among small caps*: Mohanram (2005), *Separating Winners from Losers
  among Low Book-to-Market Stocks using Financial Statement Analysis*, **Review of Accounting
  Studies** 10 (the growth-side analogue — our [Study 232](../../232-mohanram-g-score/)); and broad
  factor-zoo evidence that quality premia are strongest in small, illiquid names. We deliberately
  run the F-Score on a **large-cap survivor basket** to show how much of the headline survives
  outside its native habitat — and name the universe mismatch on the Signal axis.
- **Survivorship.** Our basket is fixed on *current* large/mid-cap tickers projected backwards, so
  every name survived to 2026; the dead firms a low F-score would have flagged are absent. A
  surviving-names panel tilts results **bullish** and shrinks the low-score tail. This is named on
  the Signal axis (cf. the desk's survivorship rule in [`METHODOLOGY.md`](../../../METHODOLOGY.md)),
  and it points *against* finding a clean low-score "loser" leg — exactly what we observe.

## Why a positive-but-insignificant spread is not an edge — the statistics

- **Small-sample / short-panel inference.** With ~15 annual observations the standard error of a
  long-short mean is large; a few-point spread cannot be told from luck. We test the spread with a
  **Newey-West (HAC) t-stat** (Newey & West, 1987, *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, **Econometrica** 55) —
  appropriate for short, autocorrelated annual series — and, because the cross-section is tiny, a
  **placebo / randomization test**: random same-size portfolios in place of the F-score legs
  (Fisher's randomization logic; Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993).
- **Monotonicity is the honest test of "separates winners from losers."** A real ordering factor
  produces returns that rise across the whole 0..9 ladder. A spread that is positive only because
  one tail is weak — while the top bucket *underperforms* the middle — is a screen that avoids
  losers, not one that picks winners. We report the full bucket ladder, not just the extremes.
- **Selection / multiple testing on a famous rule.** Published factors are selected on their
  in-sample record; Harvey, Liu & Zhu (2016), *…and the Cross-Section of Expected Returns*
  (**Review of Financial Studies**) and Bailey & López de Prado (2014), *The Deflated Sharpe
  Ratio*, formalise why a single celebrated screen discovered ex-post needs a far higher bar than
  a naive *t*.

## Method lineage (the desk's shared engine)

- **F-score construction.** [`strategy.compute_fscore`](../piotroski_f_score/strategy.py) builds
  all nine binary points from the raw EDGAR concepts; year-over-year points compare a fiscal year
  to the prior one on the same ticker (no look-ahead).
- **Long-short + HAC t + placebo.** [`strategy.long_short`](../piotroski_f_score/strategy.py),
  [`strategy.hac_tstat`](../piotroski_f_score/strategy.py) and
  [`strategy.placebo_pvalue`](../piotroski_f_score/strategy.py) — the Signal-axis tests:
  high-minus-low spread, a Newey-West *t*, and a 20,000-draw random-portfolio null.
- **Deterministic synthetic control.**
  [`data.synthetic_panel`](../piotroski_f_score/data.py) plants a known F-score → forward-return
  edge; the offline core runs with no network. The control confirms the detector is faithful *and*
  that the panel cannot reach significance unless the planted edge is real.
- **Execution lag + costs.** Fiscal year *y* signals are matched to calendar-year *y+1* returns (a
  conservative reporting lag); costs in [`strategy.net_of_costs`](../piotroski_f_score/strategy.py)
  charge one-way × turnover on both legs plus borrow on the short.

## Data sources used here

- **EDGAR companyfacts** (`data.sec.gov`, public, no key) — annual 10-K figures for ten us-gaap
  concepts across a 40-name large/mid-cap basket, cached under `_cache/edgar_fundamentals.parquet`.
- **yfinance** daily adjusted closes → calendar-year total returns, cached under
  `_cache/yearly_returns.parquet`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 121 — Magic-Formula](../../121-magic-formula/)**: Greenblatt's quality + cheapness rank on
  the same EDGAR machinery — a sibling fundamentals screen, same long-short / placebo apparatus.
- **[Study 232 — Mohanram G-Score](../../232-mohanram-g-score/)**: the growth-side analogue Piotroski
  inspired (low book-to-market firms) — the other half of the financial-statement-analysis pair.
- **[Study 122 — Gross-Profitability](../../122-gross-profitability/)** and
  **[Study 123 — Altman-Z](../../123-altman-z/)**: adjacent single-metric fundamental screens — how
  much does the nine-point composite add over its parts?
