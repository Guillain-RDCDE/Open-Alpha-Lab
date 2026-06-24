# References & literature map — Study 420 (Awesome Oscillator)

## The claim under test

- **Williams, Bill (1995).** *Trading Chaos: Applying Expert Techniques to Maximize Your
  Profits*, Wiley. The origin of the Awesome Oscillator: AO = SMA₅(HL2) − SMA₃₄(HL2), where
  HL2 = (high + low) / 2 is the bar midpoint. Williams frames the AO as a "leading" momentum
  gauge — the histogram crossing the zero line, plus the "saucer" and "twin-peaks" patterns,
  signal momentum shifts. Followed up in Williams & Williams (2004), *Trading Chaos, 2nd ed.*
- **The folklore version.** Across TradingView scripts, YouTube tutorials, and broker
  education pages, the AO is pitched as a sharper, smarter alternative to the MACD because it
  uses the bar *midpoint* (claimed to filter closing-auction noise) and a fixed 5/34 pair
  "tuned to natural market swings." The cleanest tradable reading — and the steelman we test —
  is the zero-line rule: long when AO > 0, flat when AO ≤ 0.
- **The MACD it is compared to.** Appel, Gerald (1979/2005), *Technical Analysis: Power Tools
  for Active Investors* — the Moving Average Convergence/Divergence line, EMA₁₂(close) −
  EMA₂₆(close). We run the identical long/flat zero-line rule on the MACD line so the
  "AO beats MACD" claim is actually adjudicated rather than asserted.

## The statistical and financial literature behind moving-average timing

- **Brock, Lakonishok & LeBaron (1992).** *Simple Technical Trading Rules and the Stochastic
  Properties of Stock Returns*, Journal of Finance — the canonical systematic test of MA
  crossover rules; reported outperformance in pre-1987 data that later work attributes largely
  to data-snooping and non-trading-cost assumptions.
- **Sullivan, Timmermann & White (1999).** *Data-Snooping, Technical Trading Rule Performance,
  and the Bootstrap*, Journal of Finance — applies White's Reality Check to a universe of
  ~7,800 trading rules; the BLLB results do not survive a snooping correction. Directly
  motivates our permutation placebo and the inference bar.
- **Zakamulin, Valeriy (2014).** *The Real-Life Performance of Market Timing with Moving
  Average and Time-Series Momentum Rules*, Journal of Asset Management — after crediting the
  cash leg and adjusting for time-in-market, the Sharpe advantage of MA timing rules is much
  smaller than reported and often not statistically significant on post-2000 data. This is
  exactly our excess-of-cash result: the AO loses the fair race.
- **Marshall, Cahan & Cahan (2008).** *Does Intraday Technical Analysis in the U.S. Equity
  Market Have Value?* and related work on indicator families — broadly finds that popular
  oscillators add no value once costs and selection are accounted for.
- **Han, Zhou & Zhu (2016).** *A Trend Factor*, Journal of Financial Economics — the
  profitability of MA rules is concentrated early and decays; our 2010s sub-period (the AO's
  worst) is consistent with this.

## Why a slow trend filter behaves the way it does

- **Time-series momentum.** Moskowitz, Ooi & Pedersen (2012), *Time Series Momentum*, JFE —
  documents a genuine trend premium across 58 instruments. The AO/MACD zero-line rule is a
  binary, single-asset, very-slow version; on a smooth drifting index ETF the premium it
  could harvest is swamped by the upside it forgoes while flat.
- **Regime-switching structure.** Hamilton (1989), *A New Approach to the Economic Analysis
  of Nonstationary Time Series and the Business Cycle*, Econometrica — the two-state bull/bear
  model our synthetic generator (`data.synthetic_panel`) implements. When a persistent bear
  regime exists, a slow trend filter can step out of it and beat buy-and-hold on Sharpe — the
  positive control confirms the harness recovers this (*t* = +2.57 with a planted bear); SPY's
  history simply doesn't have enough of it for the AO to win.

## Related desk studies

- **[Study 104 — Bollinger-Reversion](../../104-bollinger-reversion/)**: another band-on-a-
  moving-average rule; same lesson that the impressive gross number is mostly bull-market drift
  a random buyer collects too.
- **[Study 105 — Coppock-Curve](../../105-coppock-curve/)**: a smoothed-momentum oscillator
  (weighted ROC) of the same "fast-minus-slow" genus as the AO.
- **[Study 106 — Supertrend](../../106-supertrend/)**: an ATR-band trend-flip rule pinned
  against a random-direction control — the same "is it timing or a coin?" question.
- **[Study 110 — Faber-Timing](../../110-faber-timing/)**: the 200-day SMA in/out rule. The
  direct cousin: a slow trend filter that, unlike the AO, *does* clear the risk axis (drawdown
  reduction with timing skill confirmed vs random). The contrast is instructive — the AO's
  5/34 reading is too fast and too noisy to bank Faber's risk benefit.
- **[Study 178 — CCI](../../178-cci/)**: the Commodity Channel Index, another oscillator from
  the same family with the same null result.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica —
  implemented in [`strategy._hac_tstat`](../awesome_oscillator/strategy.py).
- **Return-difference t-stat (Sharpe comparison).** Jobson & Korkie (1981), *Performance
  Hypothesis Testing with the Sharpe and Treynor Measures*, Journal of Finance —
  [`strategy.diff_tstat`](../awesome_oscillator/strategy.py).
- **Permutation / rotation placebo.** A circular-rotation null that preserves the signal's own
  serial dependence (cf. Politis & Romano (1994), *The Stationary Bootstrap*, JASA) —
  [`strategy.permutation_pvalue`](../awesome_oscillator/strategy.py).

## Data sources

- **SPY daily total-return OHLC** (via `yfinance`, `auto_adjust=True`), 1993-01-29 to
  2026-06-12 (n = 8,400; the in-progress bar dropped). The AO needs high/low (for the
  midpoint), so a close-only series will not do. Split/dividend adjustment is essential for a
  multi-decade buy-and-hold comparison.
- **Cash rate proxy:** a flat 4%/yr rate (FRED unavailable in this sandbox; conservative vs
  the actual Fed funds path of 0%–5.5% over the sample). Higher cash yields would narrow the
  AO's CAGR gap to buy-and-hold but cannot reverse the negative excess-Sharpe difference.
