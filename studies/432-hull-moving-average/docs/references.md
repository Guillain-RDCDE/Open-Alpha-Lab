# References & literature map — Study 432 (Hull Moving Average)

## The claim under test

- **The folk recipe.** Alan Hull introduced the Hull Moving Average (HMA) in 2005
  (Alan Hull, *"How to Reduce Lag in a Moving Average"*, alanhull.com / ActVest). The HMA
  stacks linearly-weighted moving averages — `HMA(n) = WMA(2·WMA(n/2) − WMA(n), √n)` — to
  produce a curve that hugs price with dramatically less lag than an SMA or EMA. The pitch,
  repeated across TradingView scripts, broker education pages and YouTube, is that *because*
  the HMA lags less, a trend rule built on it (price-vs-HMA, or HMA slope) **turns earlier
  and generates fewer false signals ("whipsaws") than a simple moving average** — so it both
  catches trends sooner and avoids being chopped up in noise. We steelman this as: *an
  HMA-slope long/flat timing rule on daily equity bars beats buy-and-hold on a net,
  excess-of-cash Sharpe basis, and beats the equivalent SMA rule, with fewer position changes.*

## Why the steelman is *almost* coherent — the real thing it leans on

- **Lag is real and the HMA does reduce it.** The construction is mathematically sound: the
  `2·WMA(n/2) − WMA(n)` term is a finite-difference extrapolation that genuinely pulls the
  average toward the most recent price, and the outer `WMA(√n)` re-smooths it. As a
  *display* indicator, the HMA does track turns faster than an SMA. The question is whether
  faster tracking translates into *better trading decisions*.
- **Moving-average trend rules can work — on trending series.** A price-vs-MA rule is a
  trend-following filter, and trend-following is a documented premium in some markets and
  horizons (Moskowitz, Ooi & Pedersen 2012, *"Time Series Momentum"*, Journal of Financial
  Economics; Hurst, Ooi & Pedersen 2017, *"A Century of Evidence on Trend-Following
  Investing"*, AQR). Faber (2007), *"A Quantitative Approach to Tactical Asset Allocation"*,
  popularised the 10-month SMA timing rule. The HMA folklore borrows this legitimacy.
- **Lag is not free of a trade-off.** The classic result (e.g. Ehlers 2001, *"Rocket Science
  for Traders"*) is that *any* lag reduction in a causal filter comes at the cost of higher
  noise sensitivity — you cannot reduce both lag and noise without more data. The HMA buys
  its low lag with a sharper, noisier response, which on choppy daily equity bars means
  **more** false crossings, not fewer.

## The failure mode exposed

- **Whipsaws go up, not down.** On SPY the HMA(16) slope rule fires ~32.5 position changes
  per year against the SMA(50)'s ~17.4 — nearly double — directly contradicting the "fewer
  false signals" pitch. Faster reaction on a noisy series is *more* whipsaw, by construction.
- **Timing-vs-holding is the only fair race.** A long-biased rule in a 33-year bull tape
  makes money; that is exposure, not skill. Measuring the *active spread* (strategy −
  buy&hold) and the position-shuffle permutation isolates the timing, and the timing is
  significantly negative (HAC *t* = −5.31, permutation *p* = 1.0). This is the
  alpha-vs-beta discipline of the desk.
- **Out-of-sample / data-snooping.** The choice of period (16) and the SMA comparator are
  conventional defaults; sweeping them to find the one window that "works" would be exactly
  the data-snooping that Sullivan, Timmermann & White (1999), *"Data-Snooping, Technical
  Trading Rule Performance, and the Bootstrap"* (Journal of Finance), and Bailey, Borwein,
  López de Prado & Zhu (2014), *"Pseudo-Mathematics and Financial Charlatanism"* (Notices of
  the AMS), warn against. Brock, Lakonishok & LeBaron (1992), *"Simple Technical Trading
  Rules and the Stochastic Properties of Stock Returns"* (Journal of Finance), and Park &
  Irwin (2007), *"What Do We Know About the Profitability of Technical Analysis?"* (Journal
  of Economic Surveys), document how fragile MA-rule profitability is out of sample.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *"A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix"* (Econometrica) —
  implemented in [`strategy.hac_tstat`](../hull_moving_average/strategy.py).
- **Permutation / placebo testing.** The circular-shift placebo on the realised position
  path follows the randomisation-inference tradition (Politis & Romano 1994, *"The
  Stationary Bootstrap"*, JASA) — [`strategy.permutation_pvalue`](../hull_moving_average/strategy.py).
- **Execution-lag & cost discipline.** One documented `shift` (signal on close of *t* earns
  *t+1*), costs one-way × NAV, shorts pay borrow — per [`METHODOLOGY.md`](../../../METHODOLOGY.md).

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`), full history to **2026-06-12**, across five
  liquid total-return tapes (SPY, QQQ, AAPL, MSFT, XLE). The offline reproducible core and
  the notebooks run on the deterministic [`data.synthetic_panel`](../hull_moving_average/data.py)
  generator when no cache is present, never the network. Each headline is pinned with an
  as-of date and a per-tape content fingerprint (see [`docs/results.md`](results.md)).

## Related desk studies

- **[Study 178 — CCI](../../178-cci/)**: Lambert's oscillator, the same "does a textbook
  indicator beat a fair benchmark?" question on daily equities — also no edge.
- **[Study 104 — Bollinger-Reversion](../../104-bollinger-reversion/)**: another canonical
  moving-average-derived rule (bands) raced against a fair control; same honest treatment.
- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: the 50/200 SMA golden cross — the
  archetypal "lagging moving-average crossover" teardown, the closest cousin to this one.
- **[Study 106 — Supertrend](../../106-supertrend/)**: a trend-following technical rule on
  the same infrastructure — same family, contrasting where a trend filter can and can't pay.
