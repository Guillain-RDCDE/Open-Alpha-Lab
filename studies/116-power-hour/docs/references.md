# References & literature map -- Study 116 (Power-Hour)

## The claim under test

- **The folk recipe.** The "power hour" / "smart money hour" is one of the most cited
  day-trading myths: "the last hour of trading (15:00-16:00 ET) carries directional momentum
  -- smart money and institutions push the day's established trend into the close." The
  expected test is: the morning/midday return (09:30->15:00 ET) predicts the same-direction
  move in the last hour (15:00->16:00 ET). The prescription: follow the day's trend into the
  close, taking a long (short) position if the day is up (down) at 15:00. We steelman this as
  a regression / correlation test and a fair-comparison trade study: does following the morning's
  direction beat a random-direction coin in the last regular-session bar?

## The real effect the claim leans on -- and the contrary evidence

- **Intraday momentum -- late-day continuation.** Gao, Han, Li & Zhou (2018), *Market
  Intraday Momentum* (Journal of Financial Economics 137, 560-584), find that the first
  half-hour of trading positively predicts the *last* half-hour on SPY -- effectively a
  same-day momentum effect. This is the most rigorous academic support for the "power hour"
  idea and defines the effect precisely: it is *first half-hour* -> *last half-hour*, not
  morning-to-close. Our study finds the opposite with 1-hour bars (2023-2026): SPY and QQQ
  show zero continuation, IWM shows reversal. The Gao et al. sample ends in 2015; if the
  effect existed it may have been arbitraged away.
- **Intraday reversal at the close.** Bogousslavsky (2021), *Intraday Arbitrage Between ETFs
  and Their Underlying Portfolios* (Journal of Financial Economics), and Cushing & Madhavan
  (2000), *Stock Returns and Trading at the Close* (Journal of Financial Markets 3, 45-67),
  document end-of-day price reversals driven by institutional rebalancing, ETF creation/
  redemption, and index-inclusion effects -- all of which push prices *against* the day's
  trend near the close. Our finding (fade t = +3.62) is consistent with these reversal
  mechanisms dominating continuation at the close.
- **Opening-hour momentum and session-close base rate.** Heston, Korajczyk & Sadka (2010),
  *Intraday Patterns in the Cross-section of Stock Returns* (Journal of Finance 65, 1369-1407),
  document periodic return continuation at the *same time of day* across weeks. This is
  cross-week, not intraday continuation, and does not predict the power-hour claim.
- **Efficient market perspective.** Fama (1970), *Efficient Capital Markets: A Review of Theory
  and Empirical Work* (Journal of Finance 25, 383-417), and Lo & MacKinlay (1988), *Stock
  Market Prices Do Not Follow Random Walks* (Review of Financial Studies 1, 41-66). Short-
  horizon return predictability, when it exists, tends to be microstructure-driven and
  exploited away quickly. A two-year-old "power hour" pattern visible on free data is unlikely
  to represent a durable edge.

## The desk studies this is most related to

- **[Study 13 -- Crimson-Hour](../../13-crimson-hour/)**: the "opening candle" claim (edgeful's
  setup) -- a related intraday-session decomposition that finds the morning's direction is
  mechanically embedded in the full-day return but does not truly forecast the afternoon. Same
  family (morning vs rest-of-day), same method (session panel with a random-direction control).
- **[Study 87 -- Center-Line](../../87-center-line/)**: another intraday study using 5-minute
  and hourly bars, testing whether a midday reference level predicts afternoon direction.
- **[Study 73 -- First-Light](../../73-first-light/)**: the first-bar (09:30) return as a
  predictor of the rest of the session -- closely related to the Gao et al. (2018) first/last
  half-hour effect.
- **[Study 72 -- Loaded-Dice](../../72-loaded-dice/)**: the 5-minute SMA crossover scalp --
  same framework (signal vs random-direction control, symmetric payoff), different horizon.

## Method lineage (shared desk engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica 55,
  703-708) -- [`strategy.summarize`](../power_hour/strategy.py) implements the local inline
  version, consistent with [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Pearson correlation t-stat.** Fisher (1915) z-transform approximation; n-2 degrees of
  freedom; implemented in [`strategy.morning_last_corr`](../power_hour/strategy.py).
- **Random-direction control.** Same discipline as Study 72 (Loaded-Dice): hold the entry
  schedule fixed and replace the signal direction with an i.i.d. fair coin, seeded for
  reproducibility. This is the only honest null for "does the signal point the right way?".
- **Deterministic synthetic generator.** The ``continuation`` knob in
  [`data.synthetic_sessions`](../power_hour/data.py) bakes in a known morning->last-bar
  linear loading so tests can assert the engine detects continuation when it exists.

## Data sources

- **Yahoo! Finance 1-hour bars** (via `yfinance`): SPY, QQQ, IWM, ~730 sessions each
  (approx 2023-07-18 to 2026-06-12). Yahoo caps 1-hour history at roughly 730 calendar days.
  Each session is reduced to (morning_ret, last_ret) where morning_ret = open 09:30 -> close
  14:30, last_ret = open 15:30 -> close 15:30. The offline reproducible core and the test
  suite run on the deterministic [`data.synthetic_sessions`](../power_hour/data.py) generator.
