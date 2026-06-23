# References & literature map — Study 396 (Reshoring-Basket)

## The claim under test

- **The reshoring / nearshoring thesis.** A wave of sell-side and macro commentary argues that
  deglobalisation — the 2018 US–China tariff war, the 2020 COVID supply-chain breakdown, the
  2022 *Inflation Reduction Act* and *CHIPS and Science Act* subsidies — is structurally
  pulling manufacturing back to North America, and that a basket of **US industrials**,
  **factory-automation / robotics** names and **Mexican equities** (the nearshoring
  beneficiary) should therefore *durably out-earn* a plain global index. The trade has been
  packaged into thematic ETFs and "nearshoring" research notes (e.g. Morgan Stanley and BofA
  reshoring baskets, 2022–2024; the Tema *American Reshoring* ETF `RSHO`, 2023-).
- **The believers' framing.** "Reshoring is a multi-decade capex super-cycle; own US factories
  and Mexico and you own the trend." The strong version is a *structural alpha* claim — not
  "industrials are high beta" but "this theme out-earns *beyond* its market exposure."

## Why this is a beta question before it is an alpha question

- **Sector beta ≠ alpha.** A basket of cyclical industrials plus an EM single-country fund is
  **high beta** (β ≈ 1.04 vs SPY here). In a rising market a high-beta book out-*returns* the
  index for free; that is the *risk premium you were always paid for* (Sharpe, 1964, *Capital
  Asset Prices*; Lintner, 1965). The honest test of a thematic "edge" is the **CAPM/Jensen
  alpha** — the regression intercept of the basket's excess-of-cash return on the benchmark's
  excess-of-cash return (Jensen, 1968, *The Performance of Mutual Funds in the Period
  1945–1964*, Journal of Finance). Raw outperformance that disappears once you regress out the
  market is not alpha.
- **Benchmark choice moves the verdict.** Measured against the US `SPY` the basket looks
  better than against the global `ACWI`; a US-vs-global tilt is itself a *factor bet*, not
  manufacturing alpha. We report both and lead with the harder (global) yardstick.

## Why a thematic basket's track record is usually a mirage

- **Theme/ETF launch timing & backward selection.** Thematic baskets are constructed *after* a
  narrative is hot, on names selected on their realised winners — classic selection bias.
  Ben-David, Franzoni, Kim & Moussawi (2023), *Competition for Attention in the ETF Space*
  (Review of Financial Studies) document that *specialised/thematic* ETFs launch near a theme's
  peak and **underperform** afterwards; this is the empirical signature we test for with a
  pre-announced pre/post-narrative split.
- **Survivorship at the single-name level.** A "robotics/automation" sleeve represented by one
  long-listed name that *compounded* (`ROK`) is breadth-of-the-survivor: the names that failed
  to deliver the theme are not in the basket. Survivorship is named on the Signal axis (per the
  desk's house rule).
- **In-sample storytelling / multiple testing.** A narrative discovered ex-post needs a higher
  bar than a naïve *t* (Harvey, Liu & Zhu, 2016, *…and the Cross-Section of Expected Returns*,
  Review of Financial Studies; Bailey & López de Prado, 2014, *The Deflated Sharpe Ratio*).

## Method lineage (the desk's shared engine)

- **Plain + HAC *t* of the daily excess.** [`strategy.plain_t`](../reshoring_basket/strategy.py)
  and [`strategy.hac_t`](../reshoring_basket/strategy.py) — the iid and Newey-West (1987)
  autocorrelation-robust tests of mean excess return vs zero; the HAC *t* is the desk's `REAL`
  bar.
- **CAPM/Jensen alpha with iid + HAC inference.**
  [`strategy.capm_alpha`](../reshoring_basket/strategy.py) regresses (basket − cash) on
  (benchmark − cash) and reports the intercept's iid and HAC *t* — the beta-adjusted edge.
- **Excess-of-cash Sharpe race + sign-flip placebo + costs.**
  [`strategy.sharpe_race`](../reshoring_basket/strategy.py),
  [`strategy.placebo_pvalue`](../reshoring_basket/strategy.py) (Rademacher sign-flip on the
  monthly excess — Fisher's randomization logic; Efron & Tibshirani, 1993, *An Introduction to
  the Bootstrap*), and [`strategy.net_of_costs`](../reshoring_basket/strategy.py) (one-way cost
  × turnover on a long-basket / short-benchmark book).
- **Deterministic synthetic control with a planted-alpha knob.**
  [`data.synthetic_reshoring`](../reshoring_basket/data.py) builds basket = β·market + planted
  alpha + noise; the offline core runs with no network. The control confirms the harness is
  unbiased (zero planted alpha ⇒ no significant alpha despite high beta) and that a large
  planted alpha lights up.

## Data sources used here

- **yfinance** daily adjusted closes for `XLI`, `EWW`, `ROK`, `SPY`, `ACWI`, 1999-01-05 →
  2026-06-18, cached under `_cache/reshoring_prices.csv`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 394 — Defense-Basket](../394-defense-basket/)**: the same thematic-basket family,
  tested as an event study around geopolitical shocks; "war stocks rally on war news" is busted
  the way "reshoring structurally out-earns" is busted — a narrative wrapped around beta and a
  handful of memorable names.
- **[Study 356 — GLP-1-Basket](../356-glp1-basket/)** and
  **[Study 393 — AI-Datacenter-Basket](../393-ai-datacenter-basket/)**: sibling theme-basket
  teardowns — does owning the narrative deliver anything beyond the sector beta you could have
  bought directly?
