# References & literature map — Study 428 (Stochastic RSI)

## The claim under test

- **The folk recipe.** The Stochastic RSI was introduced by Tushar Chande and Stanley Kroll
  in *The New Technical Trader: Boost Your Profit by Plugging into the Latest Indicators*
  (Wiley, 1994). It applies the Stochastic-oscillator transform to the RSI series rather than
  to price: StochRSI = (RSI − min RSI) / (max RSI − min RSI) over a look-back, then %K/%D
  smoothing. The pitch, repeated on essentially every charting platform (TradingView,
  Investopedia, StockCharts), is that this "indicator of an indicator" is **more sensitive**
  than plain RSI and so generates earlier, sharper overbought/oversold turns. The rule:
  %K < 20 → oversold → buy; %K > 80 → overbought → exit/short. We steelman this as: *a
  StochRSI long-flat timing rule on a broad index, net of costs, beats both buy-and-hold and
  the obvious simpler plain-RSI rule.*

## Why the steelman is *almost* coherent — the real effect it leans on

- **RSI's own pedigree.** J. Welles Wilder introduced RSI in *New Concepts in Technical
  Trading Systems* (1978). Decades of practitioner use give the 30/70 (and 20/80) thresholds
  a focal-point quality. StochRSI inherits that lineage and adds a range-normalisation that
  *does* make the line move faster — the question is whether faster means better.
- **Short-horizon reversal.** Jegadeesh (1990), *"Evidence of Predictable Behavior of
  Security Returns"* (Journal of Finance), and Lehmann (1990), *"Fads, Martingales, and
  Market Efficiency"* (QJE), document weekly/monthly reversal at the single-stock level — the
  effect an oversold-buy oscillator would proxy. But at the index level and the daily horizon
  this rule trades on, the evidence is far weaker, and momentum often dominates.
- **The double-normalisation rationale.** Ranking RSI within its own recent range is a
  legitimate way to make a bounded oscillator adapt to changing volatility regimes — the same
  idea behind the Stochastic oscillator (George Lane, 1950s). The machinery is sound; whether
  the *market* rewards it is the empirical question.

## The failure mode exposed

- **A *t* ≥ 2 that is pure beta.** The headline trap this study documents: the long-flat rule
  is invested ~55% of the time, so it inherits a slice of the equity risk premium. On an
  up-drifting index that alone produces a net HAC *t* of ~2 — *with negative timing alpha*.
  This is exactly the "alpha vs beta" decomposition the desk's protocol (step 4) insists on:
  Sharpe and *t* on a directional book must be read against a buy-and-hold and a
  same-exposure control, not zero.
- **No marginal value over the simpler indicator.** Stacking a second oscillator on RSI is a
  textbook over-elaboration. Park & Irwin (2007), *"What Do We Know About the Profitability
  of Technical Analysis?"* (Journal of Economic Surveys), survey how added indicator
  complexity rarely survives honest out-of-sample testing. The StochRSI−RSI Sharpe gap here
  has a 95% CI straddling zero.
- **Data-snooping on thresholds.** The 20/80 bands and the 14/14/3/3 parameter set are
  conventions, not estimates. Brock, Lakonishok & LeBaron (1992) and Sullivan, Timmermann &
  White (1999), *"Data-Snooping, Technical Trading Rule Performance, and the Bootstrap"*
  (Journal of Finance), show how much apparent technical-rule success is selection. We avoid
  re-snooping by fixing the canonical parameters and racing against fair controls.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *"A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix"* (Econometrica) —
  [`strategy.hac_t`](../stochastic_rsi/strategy.py) and
  [`quantlab.analytics`](../../../quantlab/analytics.py).
- **Block bootstrap / Sharpe-gap CI.** Politis & Romano (1994), *"The Stationary Bootstrap"*
  (JASA) — [`strategy.diff_sharpe_bootstrap`](../stochastic_rsi/strategy.py).
- **Permutation / sign-flip null.** A standard randomisation test of a symmetric-around-zero
  null — [`strategy.permutation_p`](../stochastic_rsi/strategy.py).
- **Reproducibility stamp.** [`quantlab/repro.py`](../../../quantlab/repro.py) — the as-of
  freeze and per-tape content fingerprint each headline run carries.

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`, auto-adjusted / total return), SPY back to
  1993 plus a QQQ/IWM/DIA/EFA panel. The offline reproducible core and the positive control
  run on the deterministic [`data.synthetic_panel`](../stochastic_rsi/data.py) generator,
  never the network. Each headline is pinned with an as-of date (2026-06-23, the last complete
  bar) and a per-tape content fingerprint (see [`docs/results.md`](results.md)).

## Related desk studies

- **[Study 178 — CCI](../../178-cci/)**: Lambert's oscillator in the same overbought/oversold
  family — same honest "does it beat a fair control?" treatment, same null result on equities.
- **[Study 104 — Bollinger-Reversion](../../104-bollinger-reversion/)**: the mean-reversion
  band-pierce rule; the Bollinger counterpart to a StochRSI oversold-buy.
- **[Study 106 — Supertrend](../../106-supertrend/)**: a trend-following technical rule (the
  other side of the family), same infrastructure.
- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: the 50/200 golden cross — the SMA(50/200)
  benchmark used here, taken apart in full.
