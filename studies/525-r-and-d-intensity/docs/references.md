# References & literature map — Study 525 (R-And-D-Intensity)

## The claim under test

- **The CLS R&D premium (believers' / academic version).** Firms with high
  **R&D-to-market-capitalisation** (R&D / ME) earn higher subsequent returns, because the market
  is slow to price the intangible capital those expensed R&D dollars build. A portfolio **long the
  high-R&D/ME names and short the low-R&D/ME names** should harvest a persistent premium.
- **The academic backbone.** Louis K. C. Chan, Josef Lakonishok & Theodore Sougiannis (2001), *The
  Stock Market Valuation of Research and Development Expenditures*, **Journal of Finance** 56(6),
  2431–2456. The decisive finding: the return predictability is concentrated in **R&D scaled by
  market value** (R&D/ME) — a *value / mispricing* signal — and is **far weaker when R&D is scaled
  by sales** (a pure spending-intensity signal). High R&D/ME means a firm is doing a lot of
  intangible building *relative to how cheaply the market prices it*; CLS argue that is the cheap,
  under-priced corner, not the "spends a lot on R&D" corner. This study replicates that exact
  **denominator contrast** (market-cap vs sales).

## Why the denominator is the whole game — and how this differs from study 400

- **Study 400 (Patent-Intensity)** sorts on **R&D / revenue** (R&D *spending intensity*) and finds
  a weak, style-confounded, insignificant spread. That is precisely the *sales*-scaled signal CLS
  say should be weak. **This study (525)** sorts on **R&D / market-cap** — the CLS mispricing
  signal — and races it head-to-head against the R&D/sales signal on the *same* basket. The third
  axis is: *does scaling by price (CLS) actually beat scaling by sales, as the paper claims?*
- **R&D/ME is a value cousin.** Because the denominator is market value, R&D/ME mechanically tilts
  toward *cheap* stocks (low price → high ratio), so it overlaps the book-to-market value factor —
  Fama & French (1992, 1993), *Common Risk Factors in the Returns on Stocks and Bonds*, **JFE** 33.
  Any "R&D premium" must be shown to be more than a relabelled value / growth-vs-value tilt.
- **Intangibles and accounting.** Baruch Lev (2001), *Intangibles: Management, Measurement, and
  Reporting* (Brookings); Lev & Sougiannis (1996), *The capitalization, amortization, and
  value-relevance of R&D*, **Journal of Accounting and Economics** 21 — expensed R&D distorts book
  values and creates the mismeasurement the R&D/ME signal rides on.
- **Innovative efficiency.** David Hirshleifer, Po-Hsuan Hsu & Dongmei Li (2013), *Innovative
  Efficiency and Stock Returns*, **JFE** 107(3): the premium that most robustly survives is
  patents/citations *per R&D dollar* (output), not gross input intensity — a reason to expect even
  R&D/ME to be a weak, confounded signal on a modern, large-cap tape.

## Inference & honesty (the desk's shared method)

- **HAC (Newey-West) t-stat.** [`strategy.hac_tstat`](../r_and_d_intensity/strategy.py) — the
  Signal-axis test on the monthly long-short and long-minus-SPY spreads. Newey & West (1987), *A
  Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance
  Matrix*, **Econometrica** 55. `REAL` requires `t ≥ 2` on the real tape **and** survival of a
  label-shuffle placebo; a sub-2 *t* with literature support reads `WEAK`.
- **Label-shuffle placebo.** [`strategy.placebo_null`](../r_and_d_intensity/strategy.py) permutes
  the cross-sectional signal labels across names and rebuilds the long-short; the real spread must
  sit in the tail. This kills "any split of a heterogeneous field would do it."
- **Multiple testing on a famous factor.** Harvey, Liu & Zhu (2016), *…and the Cross-Section of
  Expected Returns*, **RFS** 29; McLean & Pontiff (2016), *Does Academic Research Destroy Stock
  Return Predictability?*, **Journal of Finance** 71 — the R&D-valuation premium has been public
  since 2001 and tends to shrink out of sample.
- **Deterministic synthetic control.**
  [`data.synthetic_panel`](../r_and_d_intensity/data.py) plants a *known* annual R&D/ME premium via
  the `edge` knob; at `edge = 0` the long-short must stay insignificant (no false positive), and a
  large planted edge must light up.
- **One reporting lag + one execution lag, costs, borrow.**
  [`strategy.signal_books`](../r_and_d_intensity/strategy.py) forms each month's book from
  fiscal-year-(Y-1) R&D and the contemporaneous price (no look-ahead), enters the *next* month (one
  execution lag), charges one-way turnover × NAV at a stated bps, and **charges borrow on the short
  leg** (a long/short pays to be short).

## Data sources used here

- **SEC EDGAR companyfacts** — annual `ResearchAndDevelopmentExpense`, revenue and
  shares-outstanding for 40 large-caps (10-K full-fiscal-year facts only), 2007–2026 fiscal years,
  cached under `_cache/{rd,rev,shares}.parquet`.
- **yfinance** — monthly total returns and auto-adjusted closes (for the market-cap denominator)
  for the 40 names + SPY, 2005-02 → 2026-05, cached under `_cache/{returns,prices}.parquet`. All
  headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 400 — Patent-Intensity](../400-patent-intensity/)**: the **R&D / revenue** (sales-scaled)
  cousin — the *weak* leg of the CLS contrast. This study's third axis tests whether scaling R&D by
  *market value* (CLS's preferred denominator) actually does better.
- **[Study 330 — Low-Volatility-Anomaly](../330-low-volatility-anomaly/)** and
  **[Study 238 — Betting-Against-Beta](../238-betting-against-beta/)**: sibling cross-sectional
  long-short teardowns on the same survivor-basket machinery (HAC t, placebo null, costs + borrow).
