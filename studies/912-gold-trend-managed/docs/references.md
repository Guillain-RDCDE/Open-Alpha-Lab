# References & literature map — Study 912 (Gold + Trend)

## The claim under test

- **The trend-managed-gold thesis.** Gold is a volatile diversifier (annualised vol ~18%)
  with famously long *dead decades* — 1980–2001 and 2012–2018 — during which it lost real
  value while equities compounded. The folk fix, lifted straight from Faber's tactical
  playbook: hold gold only when its price is above its 200-day moving average, else sit in
  T-bills. The hope is that trend-managing gold ducks the dead decades, delivering a
  *better excess-of-cash Sharpe* and *much shallower drawdowns* than buy-and-hold gold —
  turning gold into a drawdown-managed diversifier you can actually live with.
- **The steelman.** The 200-day filter does not forecast gold; it responds to a confirmed
  trend. Because gold's dead decades were multi-year grinds, a slow trend filter should, in
  principle, stay out for most of them. This is a testable, mechanical claim — and the desk
  tests it on the real, costed, excess-of-cash tape, not on a promise.

## Why the rule *can* work — the mechanism

- **Faber (2007).** Mebane T. Faber, *A Quantitative Approach to Tactical Asset
  Allocation*, Journal of Wealth Management 9(4) (SSRN 962461). The canonical 10-month /
  200-day SMA rule, reported to lift Sharpe and slash drawdowns across five asset classes
  including gold. Our Study 110 replicates it on equities (SPY): a genuine drawdown shield
  there. Study 912 asks whether the same rule earns its keep on *gold* specifically.
- **Time-series momentum.** Moskowitz, Ooi & Pedersen (2012), *Time Series Momentum*,
  Journal of Financial Economics — trend persistence across 58 instruments, including
  commodities. Hurst, Ooi & Pedersen (2017), *A Century of Evidence on Trend-Following
  Investing*, AQR — trend as a diversifying return stream. The 200-day binary filter is the
  simplest single-asset expression of this literature.
- **Vol clustering underpins any timing power.** A trend filter inadvertently times
  volatility: it tends to exit during the high-vol declines and re-enter in calm uptrends,
  mechanically shrinking variance. Whether that variance cut *pays* (raises Sharpe) or
  merely *costs* (lowers return more than risk) is exactly the empirical question.

## Why it can fail on gold specifically

- **V-shaped recoveries.** Gold's largest up-moves often arrive as sharp reversals off the
  lows (2008→2009, 2018→2019, 2022→2024). A 200-day filter re-enters these *late*, so the
  overlay systematically forfeits the fastest gold rallies — the mirror image of the
  drawdowns it avoids. Our real-tape calendar table shows precisely this whipsaw pattern.
- **Zakamulin (2014), *The Real-Life Performance of Market Timing with Moving Average and
  Time-Series Momentum Rules*, Journal of Asset Management** — after correctly crediting the
  cash leg and time-in-market, the Sharpe advantage of MA timing is far smaller than
  headline back-tests, and often statistically absent post-2000. Our result is a clean
  single-asset instance: on gold the advantage is negative.
- **Post-publication decay / data-snooping.** Han, Zhou & Zhu (2016), *A Trend Factor*,
  Journal of Financial Economics — MA-rule advantages concentrate before publication and
  decay after. Our era cut (negative advantage in *both* 2007–2015 and 2016–2026) is
  consistent with there being no durable edge to decay.

## Related desk studies (dedup)

- **[Study 110 — Faber-Timing](../../110-faber-timing/)**: the *same* 200-day rule on
  **equities (SPY)**, where it is a genuine drawdown shield (−55% → −22%, era-robust,
  Real/Fragile). Study 912 is the honest counter-case: the identical mechanic applied to
  **gold** does *not* replicate — the dead-decade ducking is modest and not era-robust, and
  it costs Sharpe. Same rule, different asset, opposite verdict.
- **[Study 640 — Gold Overnight](../../640-gold-overnight/)**: a *close-to-open* overnight
  effect in gold — an intraday-timing edge, orthogonal to this study's *daily trend* filter.
- **[Study 649 — Gold Seasonality](../../649-gold-seasonality/)**: a *calendar* (month-of-year)
  effect in gold — a seasonal timing rule, not a price-trend overlay.
- **[Study 831 — Gold Real-Yield Timing](../../831-gold-real-yield-timing/)**: timing gold
  on the *real-yield* (TIPS) macro signal — a fundamental conditioning variable, not the
  self-referential 200-day moving average tested here.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite …
  Covariance Matrix*, Econometrica — [`strategy.newey_west_t`](../gold_trend/strategy.py)
  and [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Return-difference (Sharpe comparison) t-stat.** Jobson & Korkie (1981), *Performance
  Hypothesis Testing with the Sharpe and Treynor Measures*, Journal of Finance —
  [`strategy.sharpe_diff_tstat`](../gold_trend/strategy.py).
- **Circular block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap*, JASA —
  [`strategy.bootstrap_sharpe_ci`](../gold_trend/strategy.py) and
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).

## Data sources

- **GLD, IAU** (gold ETFs), **BIL** (1-3M T-bill / cash leg), **SPY, TLT** (diversification
  benchmarks) — daily **total-return** closes via `yfinance` (`auto_adjust=True`),
  2004→2026-06-30. The cash leg is BIL's *actual* total return, so the excess-of-cash race
  credits the real path of short rates (0% in 2009–2015; ~5% in 2023–2026) rather than a
  flat proxy — this matters: at 2023–2026 short rates, ignoring cash would flatter the
  overlay's many cash days.
- **As-of 2026-06-30.** The partial current month is dropped so the sample never creeps.
  The GLD∩BIL common window (from BIL's 2007 inception) is the honest tradable window; the
  full gold bull of 2004–2007 pre-dates a live cash ETF and is excluded from the race.
