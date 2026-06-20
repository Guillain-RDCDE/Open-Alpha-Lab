# References & literature map — Study 331 (Fifty-Two-Week-Range)

## The claim under test

- **George & Hwang (2004), *The 52-Week High and Momentum Investing* (Journal of
  Finance).** The seminal "nearness to the 52-week high" anomaly: stocks whose price is
  closest to its trailing 52-week high earn higher subsequent returns, and proximity to
  the high subsumes much of Jegadeesh-Titman individual-stock momentum. The original
  measure is the **ratio** `close / high_52w`. This study tests the natural
  generalisation — the **range position** `(close − low_52w) / (high_52w − low_52w)`,
  which anchors on *both* endpoints — and asks the only question that makes it a new
  study and not a re-run: **does the second anchor (the low) carry forward information the
  high alone does not?**

- **The "where in the range" folk read.** Practitioners routinely describe a stock as
  "trading at 90% of its 52-week range" rather than "5% off its high", treating range
  position as a richer momentum/confirmation read. The steelman: a stock high in a *wide*
  range that has also pulled clear of its low is "more confirmed" than one merely brushing
  a high it keeps revisiting.

## Why this is distinct from the desk's two neighbours

- **[Study 236 — Fifty-Two-Week-High](../../236-fifty-two-week-high/)** tests the raw
  high ratio `close / high_52w` (the George-Hwang original) as a momentum long.
- **[Study 202 — Fifty-Two-Week-Low](../../202-fifty-two-week-low/)** tests proximity to
  the 52-week *low* as a contrarian "bargain" long.
- **This study (331)** does neither in isolation. It runs a **head-to-head horse race**
  between range-position and the high ratio — a *paired-difference* HAC test and a
  cross-sectional **spanning regression** — to isolate the *incremental* value of adding
  the low as a second anchor. The verdict is about the marginal information of the second
  endpoint, not about either endpoint alone.

## The cross-section the signal lives in

- **Jegadeesh & Titman (1993), *Returns to Buying Winners and Selling Losers* (Journal of
  Finance).** Intermediate-horizon individual-stock momentum — the effect George-Hwang
  argue the 52-week-high proxy captures more cleanly.
- **Fama & MacBeth (1973), *Risk, Return, and Equilibrium: Empirical Tests* (Journal of
  Political Economy).** The period-by-period cross-sectional regression design used in
  [`strategy.spanning_regression`](../fifty_two_week_range/strategy.py) to read the
  incremental slope on range-position controlling for the high ratio.
- **Asness, Moskowitz & Pedersen (2013), *Value and Momentum Everywhere* (Journal of
  Finance).** Context for why a "more confirmed" momentum read need not earn more — the
  premium is in the cross-section, not the framing of the anchor.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West *t*-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.hac_tstat`](../fifty_two_week_range/strategy.py). Applied to the spread, the
  long-only excess, and crucially the **paired difference** of the two signals' spreads.
- **Circular block bootstrap.** Politis & Romano (1992/1994) — the block resampling in
  [`strategy.block_bootstrap_ci`](../fifty_two_week_range/strategy.py) preserves the
  autocorrelation that i.i.d. resampling would destroy.
- **Microstructure caution at the 1-day horizon.** Roll (1984), *A Simple Implicit
  Measure of the Effective Bid-Ask Spread* (Journal of Finance) — a fraction of any
  1-day cross-sectional spread is mechanical bid-ask bounce, not a tradable edge; the
  borderline 1-day result here is read against this.

## Data sources used here

- **Yahoo! Finance daily bars** (adjusted close/high/low) for a 20-name S&P 500 large-cap
  basket, **cache-first**: the study reuses the parquet cache populated by study 202
  (`ftw52l_*_1d.parquet`), so the reproducible core and tests never touch the network.
  Window 2013-01-02 → 2026-06-15, 3,383 panel-days. **Survivorship-biased** — every name
  still trades in 2026; the bias is named on the Signal axis throughout (see
  [`docs/results.md`](results.md) for the pinned, fingerprinted run).
- The offline core and the test-suite run on the deterministic
  [`data.synthetic_panel`](../fifty_two_week_range/data.py) generator, which plants two
  *separable* edges — a confound both signals share (`range_edge`) and the discriminating
  edge only the low anchor sees (`low_edge`) — never the network.

## Related desk studies

- **[Study 236 — Fifty-Two-Week-High](../../236-fifty-two-week-high/)** — the high ratio
  in isolation (inverted on this large-cap survivor sample).
- **[Study 202 — Fifty-Two-Week-Low](../../202-fifty-two-week-low/)** — the low in
  isolation (losers keep losing here).
- **[Study 75 — Knee-Jerk](../../75-knee-jerk/)** — a genuine short-horizon
  mean-reversion signal, the counterpoint to a non-signal like this one.
