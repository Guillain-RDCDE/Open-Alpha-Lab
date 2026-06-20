# References & literature map — Study 308 (Cocoa-Squeeze)

## The event under the microscope

- **The 2024 cocoa parabola.** Front-month cocoa (ICE `CC`) ran from ~$4,100/t in early
  2024 to an all-time high near $12,600/t in December 2024 — roughly a 3× move — then gave
  back about three-quarters of it. The driver was a genuine, non-financial supply shock:
  consecutive poor harvests in Côte d'Ivoire and Ghana (which together grow ~60% of the
  world's cocoa), aggravated by swollen-shoot virus, black-pod disease and adverse weather.
  Contemporary coverage: International Cocoa Organization (ICCO) *Quarterly Bulletin of
  Cocoa Statistics* (2023/24, 2024/25); Financial Times and Reuters commodity desks,
  2024. The folk question this study tests: *after a vertical move like that, was there a
  tradable edge — ride it up, or fade the crack — or just an unrepeatable freak?*

## The two competing folk theories

- **Time-series / trend momentum.** Moskowitz, Ooi & Pedersen (2012), *Time Series
  Momentum* (Journal of Financial Economics) — a positive, diversified trend premium
  across 58 futures including commodities. The folk corollary ("ride the squeeze") is that
  a market making new highs keeps going. Note the premium is a *diversified, many-market*
  effect; this study asks whether it survives on a *single* parabolic name (it does not).
- **Commodity momentum.** Erb & Harvey (2006), *The Strategic and Tactical Value of
  Commodity Futures* (Financial Analysts Journal); Miffre & Rallis (2007), *Momentum
  Strategies in Commodity Futures Markets* (Journal of Banking & Finance) — momentum works
  *cross-sectionally* across a basket, again not as a single-asset timer.
- **Bubbles and the predictability of crashes.** Greenwood, Shleifer & You (2019),
  *Bubbles for Fama* (Journal of Financial Economics) — sharp price run-ups raise crash
  probability but are notoriously hard to *time*. Sornette (2003), *Why Stock Markets
  Crash* — log-periodic super-exponential blow-offs; the peak is identifiable in
  hindsight, far less so in real time. This is the "fade the parabola" steelman, and the
  reason it is a widow-maker: the expected crash is real, the timing is not.
- **Short-horizon mean reversion / overreaction.** De Bondt & Thaler (1985), *Does the
  Stock Market Overreact?* (Journal of Finance). Extreme moves partially reverse — but the
  reversal horizon and magnitude are uncertain, which is exactly what kills a timed short.

## Why a single event cannot certify a signal

- **Multiple testing & the single-observation trap.** Harvey, Liu & Zhu (2016), *…and the
  Cross-Section of Expected Returns* (Review of Financial Studies). A pattern read off one
  spectacular chart, chosen *because* it was spectacular, is the purest form of selection
  bias. With n = 1 blow-off there is no out-of-sample, no cross-section, and no honest
  *t*-stat — the synthetic positive control is the only thing that can speak to machinery.
- **The synthetic control as machinery proof, not market evidence.** Per the desk's
  inference bar (METHODOLOGY → *The inference bar*): a planted-blow-off tape shows the
  engine *can* bank a tradable parabola; it can never back a Signal stamp on the real tape.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West *t*-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.hac_tstat`](../cocoa_squeeze/strategy.py).
- **Circular block bootstrap.** Politis & Romano (1992/1994), *The Stationary Bootstrap*
  (JASA) — block resampling preserves the volatility clustering an i.i.d. resample would
  destroy. [`strategy.block_bootstrap_ci`](../cocoa_squeeze/strategy.py).
- **One execution lag, costs one-way × turnover × NAV, shorts pay borrow.** The desk's
  standard honest-backtest discipline (see METHODOLOGY → *House rules*); implemented in
  [`strategy.book_returns`](../cocoa_squeeze/strategy.py).

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`), cocoa front-month continuous future
  `CC=F`, 2000-present. This is a **price-only** continuous roll, not a total-return index
  — labelled as such and never treated as a return stream. All headline numbers are pinned
  with an as-of date and content fingerprint (see [`docs/results.md`](results.md)). The
  offline reproducible core and the test-suite run on the deterministic
  [`data.synthetic_blowoff`](../cocoa_squeeze/data.py) generator, never the network.

## Related desk studies

- **[Study 227 — Natgas-Winter](../../227-natgas-winter/)**: the winter natural-gas spike —
  another single-commodity "obvious in hindsight" pattern that turns out to be a
  widow-maker (None / Mirage). Same lesson: a famous commodity move is not a strategy.
- **[Study 226 — Crude-Seasonality](../../226-crude-seasonality/)**: seasonality in oil —
  the disciplined cousin of "buy the obvious commodity pattern."
- **[Study 301 — Triple-RSI](../../301-triple-rsi/)**: where the desk's "the chart looks
  amazing, the *t*-stat says nothing" discipline (HAC *t*, random control, win-rate vs
  expectancy) was last applied to a viral single-asset claim.
- **[Study 196 — Long-Term-Reversal](../../196-long-term-reversal/)**: De Bondt-Thaler
  overreaction tested cross-sectionally — the *right* way to harvest mean reversion (a
  basket), in contrast to this study's single-event fade.
