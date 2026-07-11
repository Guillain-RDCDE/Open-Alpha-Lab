# References & literature map — Study 668 (Williams VIX Fix)

## The claim under test

- **The folklore.** The **Williams VIX Fix** (WVF) — attributed to **Larry Williams**, the
  same trader behind %R (sibling [127-williams-r](../127-williams-r/)) — reconstructs a
  "synthetic VIX" from ordinary OHLC bars alone:

  ```
  WVF(t) = (highest_close(22) - low(t)) / highest_close(22) * 100
  ```

  Because the real CBOE VIX needs an options chain, Williams built WVF for markets or
  eras with no listed vol surface — futures, FX, small caps, crypto, the pre-VIX decades.
  The retail-platform staple **"CM_Williams_Vix_Fix"** (ChrisMoody's TradingView script,
  ~2014) popularized the exact recipe tested here: fire when WVF pokes above a Bollinger
  band on itself (20-session mean + 2σ) — a "capitulation bottom" — and buy.
- **The mechanism claimed.** WVF ≈ 0 when the low sits at/near the recent close-highs;
  it spikes when the low undercuts them sharply — exactly the geometry of a real implied-vol
  spike (the VIX itself jumps on the day price gaps or wicks down hard). The claim is that
  this price-only proxy inherits the VIX's most-quoted property: extreme readings mark
  fear-driven, mean-reverting bottoms.
- **What this is NOT.** Bill Williams (no relation, despite the shared surname) is a
  *different* trader — his fractals ([184-williams-fractals](../184-williams-fractals/))
  and Alligator ([421-williams-alligator](../421-williams-alligator/)) are unrelated
  chart-geometry systems. The VIX Fix's author is **Larry** Williams, same as %R.

## What we measure, and the honesty rails

- **Forward returns at [5, 10, 20] days**, WVF-spike **onset** (first day of a capitulation
  episode, not every day the condition holds) vs an unconditional day, Welch *t* pooled
  across an eight-ticker basket (the planned primary), a per-ticker Newey-West (HAC, lags =
  horizon) dummy regression as the overlap-robust cross-check — forward-return windows of
  length *h* are mechanically autocorrelated at lag < *h*, and Welch alone would overstate
  precision.
- **One documented execution lag.** WVF is computed on bars through the close of day *t*;
  every trade enters at day *t+1*'s open and exits at the close *h* sessions later. No
  second shift hides anywhere in the pipeline.
- **A random-calendar placebo** (20 seeds × 500 draws) at the primary 10-day horizon:
  does the observed spike-day mean beat a purely random 3,782-day calendar drawn from the
  same pool? (It does not — see `results.md`.)
- **The third axis is the honest question a fair review would ask first:** WVF's only
  ingredient beyond a plain "how far below its recent high is price" drawdown measure is
  the **intraday low** instead of the close. A sibling indicator built identically but from
  the close alone, put through a two-dummy HAC regression against WVF on the same days,
  isolates whether that extra ingredient (the wick) earns its keep.
- **Survivorship, named on the Signal axis.** Three of the eight names (SPY, QQQ, IWM) are
  broad index ETFs — no survivorship, they hold whoever the index holds today. The other
  five (AAPL, MSFT, JPM, XOM, JNJ) were picked *because* they are still trading today with a
  clean multi-decade tape — any capitulation-bounce pattern this half of the basket shows
  can only be a bounce in names that lived to bounce back.
- **Multiple comparisons, named.** 8 tickers × 3 horizons = 24 raw Welch tests. At a
  nominal two-sided 5% rate, ≈1.2 false positives are expected from noise alone; we observe
  2 (SPY, QQQ, both at 5 days only, neither survives to 10 or 20 days) — consistent with
  chance, not evidence.

## Data sources

- **Daily OHLC** for SPY, QQQ, IWM, AAPL, MSFT, JPM, XOM, JNJ — yfinance (no key),
  split+dividend adjusted (`auto_adjust=True`), cached under `_cache/`
  (`wvf_<TICKER>.csv`), 2000-01-03 → 2026-06-30 (IWM from its 2000-05-26 inception).
- **The Williams VIX Fix formula** — Larry Williams; the specific Bollinger-band trigger
  and "CM_Williams_Vix_Fix" naming are ChrisMoody's widely-copied TradingView port
  (~2014), the version actually traded by retail today. No paywalled academic citation
  exists for WVF itself — it is pure chart-room folklore, which is exactly why this desk
  tests it against real forward returns rather than taking the claim on faith.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [111-vix-term-structure](../111-vix-term-structure/) — the **real** ^VIX/^VIX3M curve
  slope as an equity-timing signal. Needs actual options-implied vol data; this study
  needs none — WVF is a price-only proxy built for markets where the real VIX doesn't
  exist. Different data, different question (curve slope vs single-day capitulation spike).
- [92-easy-money](../92-easy-money/) — the VIX-futures **contango carry** (shorting VIXY).
  A risk-premium/roll-yield question, not a spike-timing one; no relation to WVF's
  formula or trigger.
- [127-williams-r](../127-williams-r/) — Larry Williams' **other** indicator, %R: a
  bounded oscillator normalizing the close's position in a 14-bar range, tested with
  zone-entry and cross-back framings. Same author, unrelated formula (%R uses high/low/
  close every bar; WVF uses only the rolling high-of-closes and today's low), and this
  study's spike-onset framing is closer to a volatility-regime flag than an oscillator
  crossing.
- [184-williams-fractals](../184-williams-fractals/) and
  [421-williams-alligator](../421-williams-alligator/) — **Bill** Williams (not Larry), a
  different trader; 5-bar swing geometry and a moving-average trend system respectively.
  Neither computes anything resembling implied volatility.

None of the siblings test whether a **price-only VIX proxy's spike** predicts forward
returns, or whether that spike is anything beyond a repackaged drawdown signal — this
study's own two axes.
