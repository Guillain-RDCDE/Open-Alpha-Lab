# References & literature map — Study 868 (Global Curve-Slope Carry)

## The claim under test

- **The source framing.** Ralph S. J. **Koijen, Tobias J. Moskowitz, Lasse Heje Pedersen &
  Evert B. Vrugt**, *"Carry"* (Journal of Financial Economics, 2018). *Carry* — the return
  an asset earns if prices do not move — is a predictor of expected returns across asset
  classes, **including fixed income**, where the carry of a bond position is the yield plus
  the roll-down the term structure. A cross-sectional book long high-carry and short
  low-carry markets earns a premium in their sample.
- **The roll-down / steep-curve reading.** For a duration holder a **steep** yield curve
  pays twice: the higher yield *and* the capital gain as the bond ages and rolls *down* the
  curve toward lower yields. So across bond markets a duration investor should prefer the
  steep-curve / high-carry markets and avoid the flat / low-carry ones — the intuition this
  study puts on tradable ETFs.
- **The specific test here.** Six US + international sovereign-bond ETFs are ranked each
  month by a **yield-to-duration carry proxy** (a long-window realized yield ÷ published
  effective duration) known at the close of month `t−1` and held over month `t`; the
  dollar-neutral high-minus-low book is scored with a Newey-West *t*, a column-permutation
  placebo, a three-era robustness cut, a formation-window sweep, a costed backtest, and a
  20-seed synthetic control — always benchmarked against naive equal-weight buy-and-hold.

## What we measure, and the honesty rails

- **A price-only carry proxy, and why it is a proxy.** yfinance gives *total-return* levels
  (`auto_adjust=True`), which already blend coupon and price change; there is no clean
  forward yield in the feed. We proxy each sleeve's carry by its **trailing 36-month
  realized yield** (annualised mean monthly total return) so that transient price trends
  average out and the slow structural income/roll component dominates, then divide by the
  sleeve's **published effective duration** for a carry-per-unit-rate-risk score. This is a
  proxy, disclosed on the Signal axis — a large part of the honest teardown.
- **Effective durations as fixed characteristics.** SHY ≈ 1.85, IEF ≈ 7.3, TLT ≈ 16.4,
  BWX ≈ 7.9, IGOV ≈ 8.1, BNDX ≈ 6.9 years — public fund-fact-sheet figures (iShares / SPDR
  / Vanguard, 2024-2025 vintage). Durations drift slowly and the cross-sectional *rank*
  (short SHY ≪ belly/international ≪ long TLT) is stable, which is all the ranking uses.
- **Point-in-time positions, one documented lag.** The carry proxy **known at the close of
  `t−1`** (`.shift(1)`) sets the position held over month `t`. Zero look-ahead (verified in
  the test-suite by a future-shock invariance check).
- **The right benchmark.** A carry edge must beat *just holding the bonds*; the naive
  equal-weight buy-and-hold Sharpe is reported alongside every headline.
- **Robust inference.** Newey-West (HAC, Bartlett, 6-lag) *t* on the monthly book; a
  one-sample *t* cross-checks; a **3,000-draw column-permutation placebo** breaks the
  carry → forward-return link while preserving each month's cross-sectional return spread,
  to test whether the sort carries information beyond a random leg assignment.
- **Survivorship & power named on the Signal axis.** Only currently listed funds enter (a
  mild upper bound); `BNDX` lists from 2013 so the full six-market cross-section is short;
  the panel is tiny by construction — all three limit power and are stated with the numbers.
- **The backtest is costed separately.** One-way turnover cost per rebalance plus borrow on
  the short leg — the honest test of whether a thin monthly edge survives friction.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the monthly book return).
- **Koijen, R., Moskowitz, T., Pedersen, L. & Vrugt, E. (2018)** — *"Carry"*, the
  cross-asset carry framing that includes fixed income.
- **Asness, C., Moskowitz, T. & Pedersen, L. (2013)** — *"Value and Momentum Everywhere"*,
  the cross-sectional high-minus-low sort machinery this study reuses on carry.
- **Wilson, E. B. (1927)** — score interval for a binomial share (hit-rate uncertainty).

## Data sources

- **yfinance daily closes** (`auto_adjust=True`, total-return), six sovereign-bond ETFs
  (`SHY`, `IEF`, `TLT`, `BWX`, `IGOV`, `BNDX`), resampled to month-end, 2007-01-31 →
  2026-06-30, cached under `_cache/`.
- **Effective durations** — iShares, SPDR (State Street) and Vanguard published fund fact
  sheets (2024-2025), encoded as fixed characteristics in `curve_slope_carry/data.py`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [829-global-sovereign-bond-momentum](../../829-global-sovereign-bond-momentum/) — a
  **time-series momentum / trend** signal (each market signed by its own 12-1 trend). This
  study is **cross-sectional carry** (rank markets by a level-of-yield proxy, long high vs
  short low) — a different signal over a long window, not a fast trend.
- [826-treasury-duration-bab](../../826-treasury-duration-bab/) — a **US-only**,
  beta-neutral betting-against-beta book across the Treasury maturity ladder. This study is
  a **carry** sort **across US + international** markets, not a levered low-beta book.
- [380-curve-roll-down](../../380-curve-roll-down/) — a **single-curve** roll-down timer on
  one market. This study is the **cross-market** carry sort.
- [660-carry-everywhere](../../660-carry-everywhere/) — the **cross-asset-class** carry
  factor (FX / commodity / equity / bond pooled). This study isolates the
  **sovereign-bond curve-carry** sleeve on its own tradable ETF tape and costs it standalone.

None of the siblings ranks a **panel of US + international sovereign-bond ETFs by a
yield-to-duration carry proxy** and costs the cross-sectional high-minus-low book — the
curve-slope-carry axis this study tests.
