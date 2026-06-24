# References & literature map — Study 425 (Detrended Price Oscillator)

## The claim under test

- **The folk recipe.** The Detrended Price Oscillator is a standard charting indicator,
  documented in Steven B. Achelis, *Technical Analysis from A to Z* (2nd ed., 2000) and on
  every major charting platform (StockCharts, Investopedia, TradingView). Definition:
  `DPO(t) = Close(t − (n//2 + 1)) − SMA_n(Close)(t)`. The selling point *over* a plain moving
  average is that it **removes the trend** so the market's short-term **cycle** stands out:
  troughs mark cycle lows (buy), peaks mark cycle highs (sell). We steelman this as: *the
  detrended residual contains a tradable cycle whose long/flat (or long/short) timing rule,
  net of costs and excess of cash, beats simply holding the asset.*

## Why the steelman is *almost* coherent — the real effect it leans on

- **Detrending is a real band-pass operation.** Subtracting an n-period SMA is a crude
  high-pass filter: it does mathematically isolate whatever oscillates faster than the
  window. If a genuine short-period cycle existed, the DPO would surface it — our synthetic
  positive control plants exactly such a cycle and the rule finds it (ΔSharpe +3.68 at a
  modest amplitude). The premise is not incoherent; it is empirically empty on real tapes.
- **Cyclicality in some markets.** Business-cycle and seasonal cyclicality is real in
  commodities and macro series (the spectral-analysis tradition: Granger 1966, "The Typical
  Spectral Shape of an Economic Variable", *Econometrica*). The leap the folklore makes is
  assuming a *fixed-period, tradable* cycle exists in daily equity prices.
- **Short-horizon reversal.** Jegadeesh (1990), *"Evidence of Predictable Behavior of
  Security Returns"* (Journal of Finance), and Lehmann (1990) document one-month / weekly
  reversal — the closest real cousin to what a cycle-low buy rule hopes to harvest. But that
  reversal is a cross-sectional, single-name effect, not a calendar cycle in a broad index.

## The failure mode exposed

- **Detrending fights the equity risk premium.** The long-run equity return *is* the trend.
  A rule designed to be trend-neutral is structurally short the one factor that has paid
  investors — so on mostly-trending tapes it sits in cash through the uptrend and gives up
  Sharpe. This is the mechanism behind the wrong-signed HAC *t*'s here.
- **Spurious cycles from filtering.** Detrending with a moving average can *manufacture*
  apparent periodicity that is not in the data — the Slutsky–Yule effect (Slutsky 1937,
  "The Summation of Random Causes as the Source of Cyclic Processes", *Econometrica*; Yule
  1927). A smoothed/differenced random walk looks cyclical; trading that artefact pays
  nothing.
- **The centered-DPO look-ahead trap.** The textbook DPO is drawn displaced back `n//2 + 1`
  bars, so the plotted value uses bars in its own future. A naive backtest on the centered
  series peeks; our tradable variant uses only `Close(t) − SMA_n(t)`.
- **Data-snooping in technical rules.** Brock, Lakonishok & LeBaron (1992), *"Simple
  Technical Trading Rules and the Stochastic Properties of Stock Returns"* (Journal of
  Finance); Sullivan, Timmermann & White (1999), *"Data-Snooping, Technical Trading Rule
  Performance, and the Bootstrap"* (Journal of Finance); and Park & Irwin (2007), *"What Do
  We Know About the Profitability of Technical Analysis?"* (Journal of Economic Surveys) —
  document how oscillator edges evaporate out of sample and across asset classes.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *"A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix"* (Econometrica) —
  implemented in [`strategy.hac_tstat`](../detrended_price_oscillator/strategy.py).
- **Permutation / circular-shift placebo.** A timing-randomisation test in the spirit of the
  stationary-bootstrap data-snooping literature (Politis & Romano 1994, *"The Stationary
  Bootstrap"*, JASA) — [`strategy.permutation_pvalue`](../detrended_price_oscillator/strategy.py).
- **Excess-vs-excess Sharpe race & one-day execution lag.** Desk house rules (see
  [`METHODOLOGY.md`](../../../METHODOLOGY.md) → *The inference bar*).

## Data sources used here

- **Yahoo! Finance daily adjusted bars** (via `yfinance`, `auto_adjust=True`), full
  histories across six liquid tapes (SPY, QQQ, IWM, EFA, GLD, DBC). The offline reproducible
  core and the positive control run on the deterministic
  [`data.synthetic_panel`](../detrended_price_oscillator/data.py) generator, never the
  network. Each headline is pinned with an as-of date (2026-05-31) and a per-tape content
  fingerprint (see [`docs/results.md`](results.md)).

## Related desk studies

- **[Study 178 — CCI](../../178-cci/)**: Lambert's Commodity Channel Index, the same
  detrend-and-fade-the-extreme family — same honest verdict on daily equities.
- **[Study 104 — Bollinger-Reversion](../../104-bollinger-reversion/)**: band-based
  mean-reversion, the Bollinger counterpart to DPO's oscillator framing.
- **[Study 127 — Williams-R](../../127-williams-r/)**: another normalised overbought/oversold
  oscillator, also carrying no exploitable edge on daily equity bars.
- **[Study 106 — Supertrend](../../106-supertrend/)**: a trend-*following* technical rule —
  the opposite design, same infrastructure and discipline.
- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: the 50/200 golden cross — the lagging-MA
  timing family DPO's SMA benchmark belongs to.
