# References & literature map — Study 669 (RSI-Divergence)

## The claim under test

- **The folklore.** "Bullish RSI divergence marks reversals" — one of the most-taught patterns
  in retail technical analysis: if price prints a *lower low* while the Relative Strength Index
  makes a *higher low* at the same two points, momentum is "diverging" from price — the sellers
  are running out of conviction — and a bounce should follow.
- **The academic anchor.** Wilder (1978, *New Concepts in Technical Trading Systems*) defines
  RSI itself but does not test divergence as a signal. The broader technical-analysis literature
  is thin and mostly negative on chart patterns as tradable edges: Lo, Mamaysky & Wang (2000,
  *Foundations of Technical Analysis*, JF) find *some* informational content in head-and-
  shoulders / double-top style patterns via kernel smoothing, but Bulkowski's own compendia
  (*Encyclopedia of Chart Patterns*) and academic replications of oscillator-divergence rules
  routinely fail to survive out-of-sample testing once a fair (non-buy-and-hold, non-look-ahead)
  control is applied — exactly the kind of claim this desk exists to check.
- **The mechanism, steelmanned.** RSI is a bounded oscillator computed from the ratio of
  Wilder-smoothed average gains to average losses; "divergence" is the claim that the *shape* of
  that ratio carries information price itself has not yet priced in. There is no economic
  mechanism proposed beyond crowd psychology — which is precisely why it needs a hard empirical
  test rather than a plausibility argument.

## What we measure, and the honesty rails

- **A confirmed pattern, not a snooped one.** A swing low can only be *known* `order`=5 trading
  days after it prints (you need to see the following bars to know nothing undercut it) — this
  is how the pattern is *defined*, the same way an RSI(14) reading needs 14 days of history to
  exist at all. It is not a look-ahead violation; the study's single documented **execution**
  lag is the next session's open entry *after* that confirmation.
- **Three comparisons, not one.** Divergence-conditional forward returns are measured against
  (a) the **unconditional** forward-return distribution of the same six tickers under the
  identical formula, and (b) a **random-signal placebo** — the same signal *count* per ticker,
  drawn from random eligible dates, over 20 seeds × 200 draws. Comparison (a) alone can be
  fooled by the basket's own drift; comparison (b) is the fairest test because it shares that
  drift by construction.
- **Newey-West cross-check.** Forward-return windows of length *h* overlap by construction (two
  divergences close together share tape), so the Welch split is cross-checked with a pooled
  dummy regression using a Newey-West (1987) HAC standard error with lag = h.
- **Costs on the timer, one execution lag, long only.** The pattern is bullish by definition, so
  the third-axis timer is long-only, no borrow; costs are 2 × one-way × NAV per round trip
  (5 / 10 bps).

## Why "beats a coin" is the wrong bar here

- SPY/QQQ/IWM/AAPL/MSFT/NVDA, 2010→2026, is one of the strongest bull-market baskets
  imaginable — the **unconditional** 10-day forward hit rate on this basket is already **59.7%**,
  not 50%. A divergence signal that "wins" 56% of the time is *underperforming* a coin that
  already knows the basket goes up most of the time. The fair control is the unconditional /
  random-signal bar, not literal 50% — which is exactly why this study reports both.

## Data sources

- **Daily OHLC**, SPY + QQQ/IWM/AAPL/MSFT/NVDA — yfinance (no key), cached under `_cache/`
  (`rsidiv_<ticker>.csv`), 2010-01-04 → 2026-06-30.
- Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.
- Lo, A. W., Mamaysky, H., & Wang, J. (2000). *Foundations of Technical Analysis: Computational
  Algorithms, Statistical Inference, and Empirical Implementation.* Journal of Finance, 55(4).
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [109-obv-divergence](../109-obv-divergence/) — the **volume**/price divergence sibling
  (On-Balance Volume vs price, Granville's "volume precedes price"). Same divergence *shape*,
  a completely different indicator (volume flow, not a bounded momentum oscillator) — also a
  `None`/`Mirage`.
- [75-knee-jerk](../75-knee-jerk/) — RSI(2) **mean reversion** (buy an extreme *low* RSI reading
  outright, no divergence, no swing-low structure). This study never asks "is RSI low"; it asks
  "does RSI *disagree* with price at two confirmed swing points."
- [301-triple-rsi](../301-triple-rsi/) — a **multi-timeframe RSI alignment** filter (three RSI
  windows agreeing), no divergence concept at all.
- [428-stochastic-rsi](../428-stochastic-rsi/) — the Stochastic transform applied *on top of*
  RSI (an "indicator of an indicator"), tested as a long-flat timer — no divergence, no
  swing-low pairing.
- [178-cci](../178-cci/) — the Commodity Channel Index oversold/overbought breach rule, a
  different oscillator, no divergence structure.

None of the siblings test the specific claim here: **two confirmed swing lows where price and
RSI(14) disagree in direction.** That pairwise price-vs-oscillator divergence structure is this
study's own axis.
