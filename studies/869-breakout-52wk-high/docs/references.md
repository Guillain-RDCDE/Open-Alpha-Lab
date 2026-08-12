# References & literature map — Study 869 (52-Week-High Breakout Drift)

## The claim under test

- **The event, not the level.** The famous 52-week-high anomaly of **George, T. J. &
  Hwang, C.-Y. (2004)**, *"The 52-Week High and Momentum Investing"* (Journal of
  Finance, 59(5)), ranks stocks by their *nearness* to the 52-week high (a continuous
  level, `close / 252-day high`) and finds the near-high names carry momentum. This
  study tests a **different object**: the discrete **breakout event** — the *first close
  above* the prior 252-day high. The trading-desk folklore around breakouts ("buy new
  highs", the Darvas box, Donchian channels) claims a fresh high begets more highs
  (breakout momentum); the competing behavioural reading is **anchoring / resistance** —
  the round-number 52-week high acts as a psychological ceiling, so the breakout *fades*.
  We let the tape adjudicate: after a fresh 52-week-high close, is the forward 5/20-day
  return above or below the rest of the cross-section?
- **The behavioural readings, in tension.**
  - *Breakout momentum / underreaction* — investors anchor on the old high and are slow
    to revalue, so a decisive break above it releases pent-up demand and the name keeps
    running (the George-Hwang underreaction story, applied to the event rather than the
    level).
  - *Resistance / anchoring fade* — the salient 52-week high is a reference point at
    which anchored sellers supply stock and momentum-chasers take profits, so the first
    print above it mean-reverts.
- **The specific test here.** We flag every fresh-52-week-high day point-in-time
  (`Close[t]` strictly tops `rolling(252).max().shift(1)`), enter one day later
  (`Close[t+1]`), and hold 5 or 20 trading days. The daily long-just-broke-out /
  short-the-rest forward-return spread is judged with a Newey-West *t* (lags scaled to
  the overlap horizon), a permutation placebo, a two-era robustness cut, a costed timer,
  and a seeded synthetic positive control.

## What we measure, and the honesty rails

- **A discrete breakout flag, no free model.** `flag[t]` is True when `Close[t]` strictly
  exceeds the maximum of the *prior* 252 closes — a fresh 52-week high, computed
  vectorised via `rolling(252).max().shift(1)`.
- **Point-in-time, one documented lag.** The breakout is known at the close of `t`; the
  position is opened at the close of `t+1` (`fwd[t]=Close[t+1+h]/Close[t+1]−1`). Zero
  look-ahead.
- **Robust inference.** Newey-West (HAC, Bartlett) *t* with lags = 2×horizon on the daily
  spread — the forward windows overlap and breakouts cluster (a name sits at highs for
  many consecutive days), so a plain *t* badly overstates significance (here the 20-day
  one-sample *t* of +2.48 collapses to a HAC *t* of +1.20). A one-sample *t* and a pooled
  Welch *t* (breakout events vs rest events) cross-check. A **1,000-permutation placebo**
  breaks the event → forward-return link to confirm the drift isn't a lucky flag
  alignment.
- **Survivorship is named on the Signal axis.** The universe is a **current-membership**
  set of ~50 liquid mega-caps, pulled through the `quantlab.universe` survivorship guard
  (`allow_survivorship_bias=True`, an explicit opt-in). Survivors over-print new highs, so
  the breakout base rate and any drift are an **upper bound** — this panel flatters the
  claim, and it still fails to clear the bar.
- **The timer is graded separately.** Costs are 2 sides × (in+out) one-way per event, the
  short book pays borrow — the honest test of whether a small event drift survives
  friction.

## Shared method citations

- **George, T. J. & Hwang, C.-Y. (2004)** — the 52-week-high momentum anomaly (the
  *nearness* level tested in study 236; this study tests the *breakout event*).
- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the overlapping spread series).
- **Wilson, E. B. (1927)** — score interval for a binomial share (the day-level hit rate).
- **Donchian, R. (1960s)** — the n-day channel breakout rule, the trading-desk ancestor
  of "buy new highs" (tested in its own right in study 437).

## Data sources

- **yfinance daily OHLC** (`auto_adjust=True`, total-return), 50 liquid US large-caps,
  2010-01-04 → 2026-06-30, cached under `_cache/` through `quantlab.universe`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [236-fifty-two-week-high](../../236-fifty-two-week-high/) — George-Hwang **nearness** to
  the 52-week high: a *continuous level* (`close / 252-day high`) ranked cross-sectionally
  into quintiles. This study tests the discrete **breakout event** (the first close
  *above* the prior high), a different object with a different (momentum-vs-fade) question.
- [202-fifty-two-week-low](../../202-fifty-two-week-low/) — the symmetric **low**-side
  52-week anchor. This study is the **high**-side breakout.
- [331-fifty-two-week-range](../../331-fifty-two-week-range/) — position **within** the
  52-week high-low *range* (a normalised level). This study is the **event** of tagging the
  top of that range for the first time in a year, not the standing range position.
- [437-donchian-breakout](../../437-donchian-breakout/) — the classic **Donchian** channel
  breakout (a generic shorter n-day high, a trend-following entry). This study fixes the
  window at the **52-week** high and studies the cross-sectional forward drift of the event.

None of the siblings measure the **forward return of the fresh-52-week-high breakout event
itself** across the cross-section — this study's own axis.
