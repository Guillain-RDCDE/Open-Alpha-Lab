# References & literature map — Study 387 (Economic-Surprise-Index)

## The claim under test

- **Citi's Economic Surprise Index (CESI).** Citigroup's quantitative team (Cesa-Bianchi,
  Pesaran, et al. lineage; the index popularised by Citi FX strategy ~2003) defines a
  rolling, weighted aggregate of how far each U.S. macro release lands **above or below the
  Bloomberg consensus forecast**, decaying old surprises and normalising by historical
  surprise volatility. Positive CESI ⇒ data is beating expectations. It is the standard
  "macro momentum" gauge on every trading desk.
- **The folklore.** "Buy stocks when the data keeps beating." The narrative: when releases
  systematically surprise to the upside, the economy has unpriced positive momentum, so
  forward equity returns should be higher. CESI is widely charted against the S&P 500 and
  10-year yields as a leading/coincident risk-on signal.

## Why we construct a proxy — and how

- **CESI is proprietary.** Citi's exact constituent weights, decay, and — crucially — the
  *consensus forecasts* it differences against are commercial data (Bloomberg/Citi). They are
  not freely reproducible. There is **no public history of analyst expectations** for the
  monthly releases.
- **The transparent proxy.** We build a surprise index from six public, monthly U.S.
  real-activity series on **FRED** — nonfarm payrolls (`PAYEMS`), industrial production
  (`INDPRO`), advance retail sales (`RSAFS`), U. Michigan consumer sentiment (`UMCSENT`),
  housing starts (`HOUST`), durable-goods new orders (`DGORDER`). For each, the *surprise* is
  the month's change **minus a trailing-12-month average** of that change — a naive, unbiased,
  clearly-labelled stand-in for the Street's consensus (released analysts' forecasts are
  themselves close to a slow average of recent prints; Coibion & Gorodnichenko, 2015, show
  forecasters under-react to news, so a trailing mean is a defensible — if imperfect — proxy).
  Each surprise is standardised by a trailing rolling std and averaged. This is an explicit
  **proxy** for CESI, named as such throughout, and it is the methodological knife the study
  turns on: a surprise measured against a *trailing average* is mechanically an
  **autocorrelation / mean-reversion** signal, not a clean forecast error.

## Why a high surprise index need not predict returns — the economics & statistics

- **Macro is priced fast.** Under semi-strong efficiency the market impounds a release within
  minutes; the *surprise* moves prices on the announcement, not over the following months
  (Andersen, Bollerslev, Diebold & Vega, 2003, *Micro effects of macro announcements*, AER).
  A monthly-sampled surprise index therefore captures mostly *already-reflected* information.
- **The recovery-drift confound.** Surprise indices run high coming out of recessions (data
  beats a depressed trailing average) exactly when equities stage their largest recovery
  rallies — so a naive "buy when ESI>0" inherits the market's own post-recession drift rather
  than a distinct nowcast. This is a base-rate / common-driver confound, not a directional
  edge.
- **Small-sample / base-rate inference.** US equities rise in most rolling windows, so a high
  post-signal win-rate is expected under the null; the right object is the **excess** over the
  unconditional base rate (Kahneman & Tversky, 1973). We test it with a **Welch two-sample t**
  (Welch, 1947) and a **placebo / randomization** null (Fisher's logic; Efron & Tibshirani,
  *An Introduction to the Bootstrap*, 1993).
- **Selection on a famous gauge.** A commercial index charted ex-post against the index it is
  supposed to lead is selected on its best-looking episodes; Harvey, Liu & Zhu (2016),
  *…and the Cross-Section of Expected Returns* (RFS), is the multiple-testing caution.

## Method lineage (the desk's shared engine)

- **Welch t + placebo p-value.** [`strategy.welch_t`](../economic_surprise_index/strategy.py)
  and [`strategy.placebo_pvalue`](../economic_surprise_index/strategy.py) — conditional vs
  unconditional forward returns and a 20,000-draw randomization null sized to the event count.
- **Timing backtest, net of costs.**
  [`strategy.timing_backtest`](../economic_surprise_index/strategy.py) — long/flat (or
  long/short) SPY held when ESI>0, one-month execution lag, one-way cost per turn, raced
  against buy-and-hold on a Sharpe basis (price-only, labelled).
- **Deterministic synthetic control.**
  [`data.synthetic`](../economic_surprise_index/data.py) plants a known forward edge tied to
  the surprise; the offline core runs with no network. The control confirms the inference
  recovers a planted edge **and** does not manufacture significance when the true edge is zero.

## Data sources used here

- **FRED** monthly: `PAYEMS`, `INDPRO`, `RSAFS`, `UMCSENT`, `HOUST`, `DGORDER`, cached under
  `_cache/fred_macro.csv`. **yfinance** daily adjusted SPY closes, cached under
  `_cache/spy.csv`, resampled to month-end. Window 1993-01 → 2026-05 (33.3 years). All
  headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **Macro-nowcasting & regime gauges** on this bench — the Sahm rule, the misery index, the
  Fed model and real-rate regimes — share the lesson: a real macro relationship rarely
  survives as a *tradable* monthly timing rule once you charge it the opportunity cost of
  sitting in cash through an up-drifting market.
