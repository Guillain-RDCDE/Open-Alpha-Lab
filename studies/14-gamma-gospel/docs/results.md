# Results — Study 14 (Gamma-Gospel) — pre-registration (real run pending)

> **This file is the pre-registered test, not yet the real-market numbers.** Study 14's
> reproducible core runs entirely offline (the synthetic panel + the full decomposition; see
> [`examples/run_synthetic_demo.py`](../examples/run_synthetic_demo.py) and the test-suite). The
> real-tape headline is produced by [`examples/verify.py`](../examples/verify.py), which needs a
> **paid** option-chain source: a reliable *historical* GEX requires open interest by strike, and
> the free sources we checked don't carry it (Alpha Vantage's `HISTORICAL_OPTIONS` has it but is a
> **premium** endpoint; DoltHub / OptionsDX have greeks but no OI; yfinance's live chain is a
> snapshot with unreliable OI). That data reality is *why* Study 14 ships pre-registered. Running
> `python examples/verify.py --fetch` with such a key **overwrites this file** with the as-of'd,
> fingerprinted real result. Until then, what is fixed in advance is below.

## What we will measure

For each trading day, GEX is computed at the **prior close** from the real SPY chain
(open-interest × gamma, calls long / puts short — the SqueezeMetrics dealer convention) and reduced
to a sign. The next session's character is read from daily SPY OHLC: a **range vol** (Parkinson)
and a **directional efficiency** `|close−open|/(high−low)` (high = trend day, low = range day). The
prior-close **VIX** is the confound.

## The pre-registered test and the mirage line

1. **Raw gap.** `regime_gap`: are negative-gamma ("amplifier") days more volatile / more
   directional than positive-gamma ("absorber") days? HAC *t* on the gap.
2. **The decisive control.** `partial_over_vix`: nest `y ~ vix` inside `y ~ vix + neg_gamma`. The
   surviving negative-gamma coefficient, its HAC *t*, and the **share of the raw gap that survives**
   are the verdict.

**Falsification, fixed in advance:** if the raw gap is real but the surviving coefficient collapses
toward zero (|HAC *t*| < 2 and survival share < ~30%) once VIX is partialled out, the GEX "regime"
is the **volatility regime relabeled** — *the VIX in a trenchcoat* — and the Signal stamp is
`WEAK`/`NONE`. The GEX sign earns `REAL` only if it adds significant, material predictive power
**beyond VIX**, on directional efficiency in particular (the range-vol leg is near-tautologically
tied to VIX). Either way, Tradability is judged on whether a regime *bias* — not an entry — could
survive a round-trip options-or-index spread; a sign that needs the whole-day hold to express is a
strong `MIRAGE` candidate regardless of the Signal.

## Offline validation (what already runs, every CI build)

The synthetic panel bakes a VIX-driven confound *and* an independent genuine gamma effect `beta`.
The decomposition recovers it: with `beta_de = 0.06`, the raw directional-efficiency gap (~+0.15)
shrinks under the VIX control to ~+0.056 (HAC *t* ≈ 5, ~37% surviving) — the baked-in truth. With
`beta = 0`, the same raw gap (~+0.09) collapses to ~0 (HAC *t* ≈ 0, ~0% surviving) — the trenchcoat,
caught. That is the machine the real run points at the market.
