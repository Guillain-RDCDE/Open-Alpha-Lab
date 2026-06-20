# References & literature map — Study 306 (Crack-Spread)

## The claim under test

- **The "fat crack spread → buy refiners" trade.** A staple of energy-sector commentary
  and retail trading blogs: when refining margins (the crack spread) are wide or rising,
  refiner equities (Valero, Marathon Petroleum, Phillips 66) should outperform, so the
  crack can be used to *time* a refiner basket. Stated at full strength, this is a
  predictive claim — *today's* crack tells you something about *tomorrow's* refiner
  returns. We test that literally with a lagged predictive regression and two crack-timers,
  and contrast it with the merely *coincident* relation the trade is usually built on.

## What a crack spread is, and how it relates to refiner profits

- **The 3-2-1 crack.** CME Group, *Crack Spread Handbook* / *Trading the 3:2:1 Crack
  Spread* — the industry-standard refining-margin proxy: 3 barrels of crude in → 2 of
  gasoline + 1 of distillate out. The U.S. EIA publishes crack-spread context in *Today in
  Energy* and the *Petroleum Marketing Monthly*. The crack is, by construction, an
  *accounting* margin: it is the refiner's gross profit per barrel expressed in market
  prices, so it co-moves with refiner earnings **contemporaneously**.
- **Why coincidence is not prediction.** Because the crack is the revenue-minus-input line
  of the business, a *same-day* correlation between the crack and refiner stocks is
  expected and uninformative for timing — the stock has already discounted today's margin
  by today's close. A tradable signal requires the crack to **lead** the equity, which is
  the hypothesis this study isolates (Granger 1969, *Investigating Causal Relations by
  Econometric Models and Cross-Spectral Methods*, Econometrica — the lead-lag framing).

## Commodity ↔ producer-equity predictability (the broader evidence)

- **Commodity prices and related equities.** The general finding in the literature is that
  related-commodity moves are largely *contemporaneous* with producer/refiner equity and
  offer little out-of-sample timing power once you lag the signal — consistent with
  semi-strong efficiency in liquid, heavily-followed names. See e.g. Boyer & Filion
  (2007), *Common and fundamental factors in stock returns of Canadian oil and gas
  companies* (Energy Economics), and the oil-equity links surveyed across the energy-
  finance literature. The desk's own commodity studies repeatedly find the same pattern.
- **Refining margins and cracking economics.** Industry treatments (e.g. the EIA's
  refining-margin notes; CME crack-spread educational material) describe the crack as a
  *risk-management* and *hedging* instrument for refiners — not a forecast of refiner
  equity. Its usefulness is in locking in a margin, not in predicting the share price.

## The headline number — why a high coincident R² is the wrong metric

- **Coincident vs predictive regression.** The decisive statistic is the slope on the
  *lagged* crack change (Δcrack\_{t−1} → ret\_t) and its HAC *t*; the same-day slope is a
  foil. A study that quotes the contemporaneous fit ("refiners track the crack!") and calls
  it a signal is committing the classic look-ahead-of-zero-lag error.
- **Cash-drag and the Sharpe illusion.** A part-time-in-cash timer can post a *higher*
  Sharpe than always-long buy-and-hold while adding **zero** active return — lower
  volatility from sitting out, not alpha. The honest race is excess-of-cash vs
  excess-of-cash, i.e. the active return (timer − B&H) and its HAC *t* (see METHODOLOGY →
  *excess-vs-excess Sharpe races*).

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.hac_tstat`](../crack_spread/strategy.py) and the regression slope *t*.
- **Circular block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap* (JASA),
  and the moving-block bootstrap (Künsch 1989) — [`strategy.block_bootstrap_ci`](../crack_spread/strategy.py),
  21-day blocks to preserve daily autocorrelation.
- **One execution lag.** Signal known at the close of *t* earns the return of *t+1*, one
  ``shift`` applied once in [`strategy.book_returns`](../crack_spread/strategy.py); costs
  one-way × NAV on each position change.

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`), auto-adjusted. Crack legs RB=F (RBOB
  gasoline), HO=F (heating oil), CL=F (WTI crude); refiner basket VLO/MPC/PSX, equal-weight
  total-return (PSX listed 2012, which sets the common start). All headline numbers are
  pinned with an as-of date and content fingerprint (see [`docs/results.md`](results.md)).
  The offline reproducible core and test-suite run on the deterministic
  [`data.synthetic_tape`](../crack_spread/data.py) generator, never the network.

## Related desk studies

- **[Study 226 — Crude-Seasonality](../../226-crude-seasonality/)**: another commodity
  (WTI) timing claim — same energy complex, same "the futures tell you when" framing.
- **[Study 132 — Yield-Curve-Steepener](../../132-yield-curve-steepener/)**: a curve/carry
  signal raced honestly against its benchmark — the same excess-vs-excess discipline.
- **[Study 105 — Coppock-Curve](../../105-coppock-curve/)**: a level/momentum timing rule
  graded by the same cash-drag-aware comparison.
