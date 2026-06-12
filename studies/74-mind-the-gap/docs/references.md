# References & literature map -- Study 74 (Mind-the-Gap)

## The claim under test

- **The folk recipe.** A perennial fixture of retail trading forums, day-trading
  courses, and swing-trading Discord servers: *"An opening gap always fills. When a
  stock or index gaps up at the open, fade it -- sell short and target the prior
  close. When it gaps down, go long and target the prior close. Gaps always come back
  to where they started."*  There is no canonical paper -- it is oral tradition --
  so we steelman it as the strongest testable form: *the overnight gap predicts
  intraday mean reversion with enough probability and magnitude that a fade trade
  targeting the prior close is profitable after realistic costs on a symmetric
  payoff.*  We find the fill rate is real (medium gaps fill 65% of the time) but the
  *tradable* edge is not: HAC *t* = +0.30 on the symmetric fade.

## Why the steelman is partly right -- the real effect it leans on

- **Overnight drift and gap properties.** Cliff, Cooper & Gulen (2008), *Return
  Differences Between Trading and Non-Trading Hours: Like Night and Day* (SSRN), and
  Branch & Ma (2012), *The Overnight Return, One More Anomaly* (Journal of Applied
  Finance), document that overnight gaps carry a genuine statistical structure:
  overnight returns have a different mean and risk profile than intraday returns.
  The gap-fill frequency being above 50% for moderate gaps is consistent with these
  documented asymmetries.
- **Intraday mean reversion at the open.** Woodward & Anderson (2009), *Does Time of
  Day Affect the Profitability of Technical Trading Rules?* (Applied Financial
  Economics), and the opening auction literature (e.g. Cao, Ghysels & Hatheway 2000,
  *Price Discovery Without Trading* (Review of Financial Studies)) both suggest that
  opening prices can overshoot fair value, creating partial mean reversion in the
  first session hour.  This is the micro-mechanism the gap-fill story invokes.
- **The mechanical fill problem.** Bulkowski (2010), *Encyclopedia of Chart Patterns*
  (2nd ed., Wiley), documents gap fill rates empirically and finds that ~70% of gaps
  under 1% fill within the same session -- consistent with our 65% finding for medium
  gaps.  Importantly, he also notes that large gaps (above ~1%) fill less than 50% of
  the time, which our data confirm (38%).

## Why the tradable edge is weak -- what kills the recipe

- **The stop-loss dilemma.** The gap-fade has an asymmetric problem: to survive the
  35% of non-fills (which often move *against* the trade all day), the stop must be
  wide -- but a wide stop means the win/loss ratio is unfavorable even at 65%
  accuracy.  With a symmetric 1-ATR stop and 1-ATR target the expected value is
  approximately 65% * 1 ATR - 35% * 1 ATR = +30 bps before cost -- consistent with
  the +0.46 bps we observe, which is noise.
- **Turnover and cost.** Novy-Marx & Velikov (2016), *A Taxonomy of Anomalies and
  Their Trading Costs* (Review of Financial Studies), show that even real edges are
  rapidly consumed by turnover.  At ~1 trade/day a strategy needs a gross edge far
  above 0.46 bps to survive 0.5-2 bps round-trip costs.
- **Large gaps reverse the claim.** The 38% fill rate for large gaps directly
  contradicts the folk claim.  News-gap days tend to continue in the gap direction --
  consistent with Jegadeesh & Titman (1993), *Returns to Buying Winners and Selling
  Losers* (Journal of Finance), at short horizons after earnings or macro surprises.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica)
  -- [`strategy.summarize`](../mind_the_gap/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Wilson interval for proportions.** Wilson (1927), *Probable Inference, the Law
  of Succession, and Statistical Inference* (JASA) -- used for fill-rate CIs in
  [`strategy.fill_rate_by_bucket`](../mind_the_gap/strategy.py) because the normal
  approximation fails near 0 or 1 (small-gap fill rate is ~88%).
- **Symmetric barrier backtest.** Same engine as Study 72 (Loaded-Dice) -- enter at
  bar *t*, barriers checked against high/low of the same bar (daily), conservative
  stop wins a straddle, no look-ahead.
- **Average True Range.** Wilder (1978), *New Concepts in Technical Trading Systems*
  -- the symmetric risk unit for the barrier exits.

## Data sources used here

- **Yahoo! Finance daily OHLCV bars** (via `yfinance`, `auto_adjust=True`), 10-year
  history for SPY, AAPL, MSFT, TSLA, NVDA.  Split-and-dividend-adjusted.  Daily bars
  avoid the ~60-day sub-hourly cap, giving 2,514 sessions per ticker (~12,500 pooled
  trades) for strong statistical power.  Every headline is pinned with an as-of date
  (2026-06-12) and a per-tape content fingerprint (see [`docs/results.md`](results.md)).
  The offline reproducible core and the test-suite run on the deterministic
  [`data.synthetic_daily`](../mind_the_gap/data.py) generator, never the network.

## Related desk studies

- **[Study 13 -- Crimson-Hour](../../13-crimson-hour/)**: intraday continuation study
  using the opening candle -- the same open-to-close architecture, a different
  prediction sign (continuation rather than reversion).
- **[Study 32 -- Rip-Tide](../../32-rip-tide/)**: short-term mean reversion -- the
  same reversion mechanism this study tries to ride, tested on a different signal.
- **[Study 48 -- Groundhog](../../48-groundhog/)** and
  **[Study 42 -- Last-Call](../../42-last-call/)**: calendar-pattern studies that also
  use the open-vs-prior-close framing to define a setup.
- **[Study 72 -- Loaded-Dice](../../72-loaded-dice/)**: the SMA(5/10) intraday cross
  study -- the random-direction control discipline and symmetric-barrier engine used
  here were developed there.
