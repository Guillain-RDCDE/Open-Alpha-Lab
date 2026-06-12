# References & literature map — Study 66 (Inverted)

## The signal and its source

- **Estrella, A., & Mishkin, F. (1996, 1998).** *The Yield Curve as a Predictor of U.S. Recessions* —
  the 10y−3m spread is the canonical recession leading indicator; the New York Fed publishes it monthly.
- **Harvey, C. (1988).** *The Real Term Structure and Consumption Growth* — the term structure and future
  growth; one of the earliest inversion-as-predictor results.
- **Vendor / macro belief** — yield-curve timing; backlog:
  [`docs/pwb_strategies_inventory.md`](../../../docs/pwb_strategies_inventory.md).

## On the long, variable lead (why it's not a timer)

- **Bauer, M., & Mertens, T. (2018, FRBSF).** *Economic Forecasts with the Yield Curve* — the inversion's
  signal is robust but the lead time to recession is long and variable (6–24 months); markets often rise
  into it. The basis for the FRAGILE tradability stamp.
- **Open-Alpha-Lab [Study 56 Tide-Table](../../56-tide-table/)** — the same shape as CAPE: a real
  long-horizon predictor that is a poor short-horizon timer.

## Data

- **Yahoo! Finance** — ^TNX (10-year), ^IRX (13-week) Treasury yields, and ^GSPC, monthly, 1985–2026.
  The offline synthetic world makes a (mean-reverting, occasionally inverting) curve forecast forward
  equity returns (and a null), so the result is provable offline.

*A real-but-slow macro signal, kin to [56 Tide-Table](../../56-tide-table/) (CAPE) — forecasts, doesn't time.*
