# References & literature map — Study 303 (Uranium-Revival)

## The claim under test

- **The "nuclear renaissance" trend trade.** The standard thematic-ETF / finance-Twitter
  framing: uranium is in a structural bull market (supply deficits, reactor restarts,
  AI-driven power demand), so rather than value it, *ride the trend* — hold the miner ETFs
  **URA** (Global X Uranium, listed 2010) and **URNM** (Sprott Uranium Miners, listed 2019)
  while price is above its 200-day moving average, step aside when it breaks down. The
  marketed result is "capture the rocket, skip the crashes." This is a testable hypothesis:
  a simple trend overlay on a single thematic ETF delivers a positive, *durable* timing edge
  over buy-and-hold. We test the **machinery** that would detect such an edge on
  deterministic regime controls, and we are explicit that no live URA/URNM tape ships here.

## The real effect the recipe leans on — time-series momentum (trend-following)

- **Time-series momentum.** Moskowitz, Ooi & Pedersen (2012), *Time Series Momentum*
  (Journal of Financial Economics) — a past-12-month trend predicts the next month across
  58 instruments spanning equities, bonds, currencies and commodities. The foundational
  evidence that trend-following is a real, broad, persistent effect (when measured across
  *many* markets).
- **A century of trend evidence.** Hurst, Ooi & Pedersen (2017), *A Century of Evidence on
  Trend-Following Investing* (Journal of Portfolio Management) — trend works back to 1880
  across asset classes. Note the unit of analysis is always a *diversified* portfolio, not
  a single asset: breadth is what averages out the per-asset regime luck.
- **Cross-sectional & relative-strength momentum.** Jegadeesh & Titman (1993), *Returns to
  Buying Winners and Selling Losers* (Journal of Finance) — the closely-related anomaly.
  Both literatures certify the *category*; neither certifies a *single thin sector ETF*.

## Why a single thematic ETF is the wrong vehicle

- **The single-asset / single-regime trap.** A trend rule on one volatile, hype-driven
  theme can post a large, cost-robust, low-drawdown backtest purely by stepping aside near
  the top of *one* boom-bust. In-sample, that is statistically indistinguishable from a
  genuine durable edge. The desk demonstrates this directly: holding the rule fixed and
  redrawing the synthetic boom-bust scrambles the HAC *t* across draws (Beat 4c) — the
  hallmark of regime luck, not signal.
- **Multiple testing / degrees of freedom.** Harvey, Liu & Zhu (2016), *…and the
  Cross-Section of Expected Returns* (Review of Financial Studies) — when the same rule is
  evaluated on a tape selected *because* it trended (a hot sector), the in-sample statistic
  is upward-biased by construction. The honest fix is breadth and out-of-sample tape, not a
  prettier single-asset backtest.
- **Thematic-ETF concentration & boom-bust.** Single-sector thematic funds (cannabis,
  SPACs, clean energy, AI, uranium) are launched into, and marketed at, the top of a
  narrative; their returns are dominated by one cycle. A timing overlay does not diversify
  this away — it is a concentrated bet dressed as risk management.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.hac_tstat`](../uranium_revival/strategy.py). The inference-bar statistic: a
  REAL Signal requires |t| ≥ 2 on the **real** tape.
- **Block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap* (JASA), and the
  circular block bootstrap (Künsch 1989) — resampling in blocks preserves the volatility
  clustering an i.i.d. bootstrap would destroy; used for the timing-edge CI
  ([`strategy.block_bootstrap_ci`](../uranium_revival/strategy.py)).
- **Trend signal (SMA).** The 200-day moving-average rule is the canonical long-only trend
  overlay (e.g. Faber 2007, *A Quantitative Approach to Tactical Asset Allocation*); here it
  is the deliberately-simple engine whose *vehicle* we interrogate.

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`), adjusted-close, for URA (2010–) and URNM
  (2019–) — *cache-only*; **no live tape ships with this study**. The offline reproducible
  core and the entire test-suite run on the deterministic
  [`data.synthetic_daily`](../uranium_revival/data.py) regime generator (seed=303), never
  the network. The cache-first [`data.load_real`](../uranium_revival/data.py) tries the
  study `_cache` parquet, then the shared `quantlab.data` cache, then raises (offline-safe).
  All headline numbers are pinned with an as-of date and content fingerprint (see
  [`docs/results.md`](results.md)).

## Related desk studies

- **The honesty bar for "trend" claims** is breadth: trend-following is REAL pooled across
  many markets, not on one thematic ETF — the same lesson the desk applies to every
  single-theme momentum or hot-sector timing claim. The vehicle, not the signal category,
  is what makes this one a `MIRAGE` / `HYPE ROCKET`.
