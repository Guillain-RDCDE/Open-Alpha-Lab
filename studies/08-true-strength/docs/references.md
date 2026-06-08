# Sources & literature map — Study 08 (True-Strength)

## The claim under test

- **QuantifiedStrategies.com — *True Strength Index (TSI) Trading Strategy*** (Episode 100 of
  their 365 series, posted June 2026). The write-up this study tests. It presents the TSI as a
  *double-smoothed* momentum oscillator that shows "both the direction **and the strength** of
  price momentum, while smoothing out some of the noise that often appears in raw momentum
  indicators" — i.e. a *truer* strength gauge — and reports a gold (GLD) backtest: 232 trades,
  0.77% average gain, **40% win rate**, **1.7 profit factor**, **7.8% CAGR**, **50% exposure**,
  20% max drawdown. The exact trading rules sit behind their Skool community paywall, so we test
  the falsifiable claim the indicator's *name* makes — that it is a **distinct, truer** momentum
  read than the MACD or RSI — rather than a rule we cannot see.

## The indicator itself

- **William Blau (1991), *Momentum, Direction, and Divergence***, and his *Stocks & Commodities*
  articles introducing the **True Strength Index**: `TSI = 100 · EMA_s(EMA_r(Δclose)) /
  EMA_s(EMA_r(|Δclose|))`, classically `r = 25`, `s = 13`, with a signal-line EMA. The double
  smoothing is the indicator's defining feature and its entire claim to being "truer".
- **Gerald Appel (1979), the MACD** — `EMA_12 − EMA_26` with a 9-period signal line. The
  single-smoothed momentum oscillator the TSI is implicitly compared against.
- **J. Welles Wilder (1978), *New Concepts in Technical Trading Systems*** — the **RSI**, a
  bounded [0,100] momentum oscillator with Wilder smoothing (`α = 1/period`).

## What the literature already says about oscillator redundancy & TA edges

The TSI/MACD/RSI are all **functions of recent price changes**, so collinearity is the prior, not
a surprise; the question is its degree and whether it leaves any incremental edge.

- **Brock, Lakonishok & LeBaron (1992), *Simple Technical Trading Rules…*, J. Finance** and
  **Sullivan, Timmermann & White (1999), *Data-Snooping, Technical Trading Rule Performance, and
  the Bootstrap*, J. Finance.** STW show that once you correct for the *universe of rules
  searched* (White's Reality Check), the apparent edge of momentum/MA rules largely evaporates —
  the cautionary backbone for the parameter-grid Reality Check here.
- **Lo, Mamaysky & Wang (2000), *Foundations of Technical Analysis*, J. Finance.** Formal test of
  chart indicators; finds modest, fragile incremental information — consistent with "a real but
  generic momentum signal, mostly redundant".
- **Jegadeesh & Titman (1993), *Returns to Buying Winners and Selling Losers*, J. Finance.** The
  canonical momentum premium the oscillators are all crude, collinear proxies for.
- **Marshall, Cahan & Cahan (2008), *Does intraday technical analysis in the U.S. equity market
  have value?*** — generally "no" after costs. Representative of the modern verdict on retail TA
  oscillators.

## Desk method

- **White (2000), *A Reality Check for Data Snooping*, Econometrica** — applied here to the best
  of a 24-variant TSI parameter grid (`quantlab.bayes.reality_check`).
- House methodology: [`../../METHODOLOGY.md`](../../METHODOLOGY.md). Shared engine:
  [`../../quantlab/`](../../quantlab/).

## Related studies in this repo

- **[Study 03 — Fear-Gauge](../../03-fear-gauge/)** — another "what is this indicator *really*
  measuring?" teardown (the VIX).
- **[Study 06 — Clockwork-Vol](../../06-clockwork-vol/)** — distinguishing a real signal from a
  shape read into noise; same scepticism, different object.
- **[Study 07 — Coiled-Spring](../../07-coiled-spring/)** — the previous indicator study; finds a
  *weak* generic momentum pulse, the same flavour of "real but not what it's sold as". Shares this
  study's cached universe.
