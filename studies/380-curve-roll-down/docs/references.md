# References & literature map — Study 380 (Curve-Roll-Down)

## The claim under test

- **"Riding the yield curve" / roll-down.** The classic practitioner pitch: on an
  upward-sloping (steep) curve, buy a bond at a longer maturity, hold it, and as it *ages*
  it "rolls down" the curve to a lower-yield, higher-price point. If the curve is unchanged
  you earn the bond's **carry** (its yield) *plus* a **roll-down** capital gain — said to
  beat holding cash reliably. The canonical statements are in fixed-income textbooks:
  Frank J. Fabozzi, *Bond Markets, Analysis, and Strategies* (ride-the-yield-curve and
  rolling-yield); Bruce Tuckman & Angel Serrat, *Fixed Income Securities* (carry, roll-down,
  and the term-structure decomposition of expected return).
- **Carry/roll as a documented return source.** Koijen, Moskowitz, Pedersen & Vrugt (2018),
  *Carry*, Journal of Financial Economics — bond carry (yield + roll-down) is a positive,
  priced premium across markets *on average*, but with large drawdowns concentrated in
  rate-rising regimes. Ilmanen, *Expected Returns* (2011) decomposes bond expected return
  into yield, roll-down, and the rate-change term, and is explicit that roll-down is a
  *static-curve* accounting identity, not a forecast.

## Why roll-down is a static-curve concept (the crux)

- **The realized return identity.** For a duration-`D` sleeve, the 1-year total return is
  ≈ `y0 − D·Δy`: carry minus the duration P&L of the *realized* yield change. Roll-down is
  the special case `Δy = (rolled − sleeve)`, i.e. the curve doesn't move and the bond just
  slides down a *fixed* curve. The moment rates move — a hiking cycle, a bear-steepening —
  the `−D·Δy` term dominates and can erase the entire carry+roll. This is standard
  fixed-income math (Tuckman & Serrat, ch. on return attribution); we make it empirical.
- **The expectations hypothesis vs the term premium.** Under the pure expectations
  hypothesis, an upward slope is the market *forecasting* rising short rates, and riding the
  curve earns nothing in expectation. Empirically the slope mixes an expectations component
  and a **term premium**: Fama & Bliss (1987), *The Information in Long-Maturity Forward
  Rates* (AER) and Campbell & Shiller (1991), *Yield Spreads and Interest Rate Movements*
  (Review of Economic Studies) show the slope predicts bond excess returns — i.e. roll-down
  "works" exactly to the extent it is harvesting a time-varying term premium, not a free
  lunch. Cochrane & Piazzesi (2005), *Bond Risk Premia* (AER), sharpen the predictor.

## Why our full-sample edge is regime-survivorship

- **The 1981–2021 secular bond bull.** Forty years of falling yields gave every duration
  sleeve a one-way tailwind that flattered all carry/roll strategies. Our 1990–2026 window
  inherits most of it (5y yield ~8% → ~4%, a −0.13 pts/yr drift). A backtest whose
  significance lives in a non-repeatable rate decline is the bond analogue of equity
  survivorship — named on the Signal axis. The 2010–2026 sub-period (rising-rate regime) is
  the honest out-of-regime check.
- **Overlapping-window inference.** Annual returns sampled daily overlap ~252×, inducing
  severe serial correlation; a naive t overstates significance by ≈√overlap. We use a
  **Newey-West / HAC** standard error (Newey & West, 1987, *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*,
  Econometrica) — the difference between the naive t (≈32) and the HAC t (≈2.5) is the whole
  story.

## Method lineage (the desk's shared engine)

- **Newey-West HAC t.** [`strategy.newey_west_t`](../curve_roll_down/strategy.py) — the
  honest standard error for overlapping annual excess returns (contrasted with
  [`strategy.naive_t`](../curve_roll_down/strategy.py) to expose the overlap inflation).
- **Promise-vs-reality + regime split.**
  [`strategy.promise_vs_reality`](../curve_roll_down/strategy.py) and
  [`strategy.regime_split`](../curve_roll_down/strategy.py) separate the textbook roll+carry
  from the realized return and from the rate-move regime.
- **Steep-curve placebo.** [`strategy.slope_timing_placebo`](../curve_roll_down/strategy.py)
  asks whether steep-tercile entries beat random timing (they do — but mechanically, via
  more carry, not via prediction).
- **Deterministic synthetic control.**
  [`data.synthetic_curve`](../curve_roll_down/data.py) plants a known slope→edge knob
  *beyond* the carry baseline; [`strategy.synthetic_edge_test`](../curve_roll_down/strategy.py)
  must recover a large planted edge **and** must NOT fire when only the carry baseline is
  present (edge = 0). The offline core runs with no network.

## Data sources used here

- **yfinance** daily constant-maturity Treasury yield indices `^IRX`, `^FVX`, `^TNX`,
  `^TYX`, 1990-01-02 → 2026-06-18, cached under `_cache/curve_yields.csv`. All headline
  numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 132 — Yield-Curve-Steepener](../132-yield-curve-steepener/)**: the slope as a
  *timing* signal for a long-duration sleeve — the sibling question to roll-down (does the
  slope predict, or just pay carry?).
- **[Study 364 — FX-Carry-Trade](../364-fx-carry-trade/)**: the same carry-as-premium /
  free-lunch-busted pattern in currencies — positive on average, brutal in the unwind.
- **[Study 115 — Credit-Spreads](../115-credit-spreads/)** and
  **[Study 119 — Real-Rate-Regime](../119-real-rate-regime/)**: neighbouring rates/macro
  premia and their regime dependence.
