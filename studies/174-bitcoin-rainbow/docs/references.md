# References & literature map — Study 174 (Bitcoin-Rainbow)

## The claim under test

- **The Bitcoin Rainbow Chart.** Originally created by user "azop" on bitcointalk.org
  (~2014) and popularised by Blockchaincenter.net (https://www.blockchaincenter.net/en/
  bitcoin-rainbow-chart/). The chart fits log(price) ~ log(days since genesis) on the
  full available history and places 9 coloured bands ("Fire sale" to "Maximum bubble
  territory") at fixed sigma offsets. The recipe: buy in the cold bands, sell in the hot
  bands. The chart is widely cited in retail crypto media as a "long-term valuation
  indicator." There is no peer-reviewed paper behind it; the only theoretical justification
  offered is that "Bitcoin has historically followed a power-law growth curve."

## The core statistical problems this study tests

### Look-ahead bias / in-sample curve fitting

- **Campbell & Thompson (2008).** "Predicting Excess Stock Returns Out of Sample: Can
  Anything Beat the Historical Average?" *Review of Financial Studies* 21(4), 1509–1531.
  The canonical reference for the gap between in-sample and out-of-sample predictability.
  A model that fits its own training data well is not evidence of an edge; the OOS test is
  the only one that counts. The Rainbow Chart never passes this test: the bands are drawn
  after the fact.
- **Harvey, Liu & Zhu (2016).** "...and the Cross-Section of Expected Returns." *Review
  of Financial Studies* 29(1), 5–68. On the multiple-comparisons problem and the need for
  higher t-stat hurdles when a model is fitted to the same data it is evaluated on. The
  rainbow's in-sample t = +4.53 looks impressive; under this lens it is worthless.

### Spurious regression — two trending series

- **Granger & Newbold (1974).** "Spurious Regressions in Econometrics." *Journal of
  Econometrics* 2(2), 111–120. The foundational result: regressing one non-stationary
  I(1) series on another produces high R² and significant t-stats even when the series
  are generated independently. Both log(BTC price) and log(time) are strongly upward
  trending — the textbook setup for a spurious regression.
- **Phillips (1986).** "Understanding Spurious Regressions in Econometrics." *Journal of
  Econometrics* 33(3), 311–340. Formal derivation: OLS t-stats diverge to infinity as
  sample size grows when two independent I(1) processes are regressed on each other.

### Power-law claims for Bitcoin

- **Peterson (2019).** "Metcalfe's Law as a Model for Bitcoin's Value." *Alternative
  Investment Analyst Review* Q2 2019. An early attempt to provide a theoretical basis for
  power-law BTC pricing based on network effects. Contested as circular (the model uses
  Bitcoin price data to calibrate Bitcoin price predictions).
- **Hiesboeck (2020).** "The Bitcoin Power Law Theory." Various crypto media. The retail
  popularisation of the power-law narrative that underpins the rainbow chart's log-time
  regressor. No peer-reviewed publication; the predictive claims have not survived
  systematic OOS testing.
- **PlanB (2019).** "Modeling Bitcoin Value with Scarcity." *Medium.* The Stock-to-Flow
  model, studied rigorously in Study 84 (Moon-Math) — another log-regression of BTC
  price, with the same spurious-regression diagnosis. The rainbow chart's log-time
  regressor is mathematically equivalent to a simplified version of log(S2F) after
  noting that log(S2F) ≈ log(time) between halvings.

### Walk-forward testing as the remedy

- **Lo & MacKinlay (1990).** "Data Snooping Biases in Tests of Financial Asset Pricing
  Models." *Review of Financial Studies* 3(3), 431–467. In-sample parameter search
  followed by in-sample evaluation guarantees overfit; the honest test uses new data
  (or an expanding window that withholds it).
- **Pardo (2008).** *The Evaluation and Optimization of Trading Strategies*. Wiley.
  The practitioner's reference for walk-forward analysis as the antidote to curve-fitting:
  the model is re-fitted at each step using only data available up to that point, and
  evaluated on the next period's unseen data.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *Econometrica* 55(3), 703–708 —
  `strategy.summarize` inline implementation; handles serial correlation in daily returns.
- **Reproducibility stamp.** Content fingerprint on the BTC close series, as-of date,
  and walk-forward parameters pinned in `docs/results.md`.

## Data sources used here

- **Yahoo Finance daily BTC-USD** (via `yfinance`), full available history from
  2014-09-17. Shared cache from Study 84 (Moon-Math) re-used when present. The log-time
  regressor is constructed deterministically from the genesis date (2009-01-03) — no
  network access needed for the regressor itself.

## Related desk studies

- **[Study 84 — Moon-Math](../../84-moon-math/)**: the BTC Stock-to-Flow model —
  same spurious-regression family, same diagnosis. log(S2F) is nearly identical to
  log(time) between halvings, so the two charts are closely related.
- **[Study 117 — Pi-Cycle-Top](../../117-pi-cycle-top/)**: another BTC chart indicator
  (the 111-day MA vs 2x 350-day MA crossover), tested with the same honest protocol.
- **[Study 81 — Four-Year-Itch](../../81-four-year-itch/)**: the BTC halving cycle
  narrative tested rigorously — small effective n, multiple comparisons.
