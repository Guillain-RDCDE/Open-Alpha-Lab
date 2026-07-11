# References & literature map — Study 642 (Turnaround Tuesday)

## The claim under test

- **The folklore.** "**Turnaround Tuesday**" — a trading-desk staple: if Monday closes
  down, Tuesday tends to bounce back, so a down Monday is a buy signal into the close.
  It is the mean-reversion cousin of the (largely dead, see [90-weekend](../90-weekend/)
  and [224-monday-effect](../224-monday-effect/)) "Monday Effect" — French's (1980)
  finding of significantly *negative* Monday close-to-close returns, 1953-1977. The
  turnaround version doesn't claim Monday is always negative; it claims that *when*
  Monday is negative, Tuesday reverses.
- **The academic anchor.** Short-horizon return reversal is a documented phenomenon —
  Jegadeesh (1990, *Evidence of predictable behavior of security returns*, JF) and
  Lehmann (1990, *Fads, martingales, and market efficiency*, QJE) both find that
  individual stocks which fall one week tend to rebound the next, a "contrarian"
  effect distinct from cross-sectional momentum at longer horizons. Turnaround Tuesday
  is the calendar-specific, retail-facing packaging of that same mean-reversion
  intuition, applied to a single weekday pair on the index level rather than to
  individual names or a rolling window.
- **The open question we test.** Is a **down Monday specifically** a sharper
  reversal signal than "any down day"? If turnaround Tuesday is nothing more than
  Jegadeesh/Lehmann-style reversal relabelled with a calendar hook, it should show up
  equally on *every* weekday pair (Tue→Wed, Wed→Thu, …) — not just Monday→Tuesday.

## What we measure, and the honesty rails

- **E[Tuesday | Monday < 0] vs unconditional Tuesday vs all days** — Welch *t* for
  both contrasts (the events are single, weekly, non-overlapping — Welch is the
  planned primary), plus a **Newey-West (1987)** 5-lag *t* on the down-Monday-Tuesday
  dummy regression as the autocorrelation-robust cross-check.
- **Pairing, precisely.** A Tuesday only enters the conditional sample if the
  *immediately preceding trading session* is a Monday — a Monday market holiday
  (e.g. Presidents' Day, Memorial Day) breaks the adjacency and correctly drops that
  Tuesday, since there is no Monday close to condition on.
- **The Monday-specificity test is the honesty rail against relabelled reversal.**
  We run the identical down-day → next-day split on all five weekday pairs and Welch-t
  the pooled non-Monday result — see [`docs/results.md`](results.md) for the finding
  that only Mon→Tue clears the bar.
- **Hit rate carries a Wilson (1927) interval**; the placebo is a 20-seed × 1,000-draw
  random-pair null (reshuffle which Tuesdays carry the down-Monday label, count held
  fixed); the era split (2000-01-01, named in the brief) is tested as a **difference**,
  not eyeballed; a robustness cut drops the 2008-09 and 2020-03 crisis windows entirely
  to rule out a handful of macro days dominating the mean.

## Why the timer is graded separately (and fragile)

- **Costs are charged one-way × NAV per leg** (5 / 10 bps; SPY spreads are pennies on
  hundreds of dollars of NAV, so 5 bps already includes slippage headroom beyond the
  raw spread). Entry is the Monday close — the down-Monday flag is knowable at that
  exact instant (it *is* that close), so this is a zero-look-ahead scheduled-style
  entry, the study's single documented execution convention.
- The net edge is **cost-sensitive by construction**: gross Sharpe ≈ 0.70 collapses to
  0.37 at 5 bps and to 0.04 at 10 bps. This is the honest reason the tradability stamp
  is `FRAGILE` rather than `INVESTABLE` — a real, certified signal that thins out fast
  once realistic frictions are charged, on a position that is only on ≈ 8% of trading
  days.

## Data sources

- **SPY daily raw OHLC + adjusted (total-return) close** — yfinance (no key), cached
  under `_cache/` (`tt_spy.csv`), 1993-01-29 → 2026-06-30.
- **No hardcoded event calendar.** Unlike a scheduled-announcement study, "Monday" and
  "Tuesday" are read directly off the trading-day index's day-of-week — there is no
  external table to source.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [224-monday-effect](../224-monday-effect/) — the **unconditional Monday level**
  (French 1980's original claim: is Monday, on average, negative?). SPY Monday is
  **positive** (+5.64 bps) on the modern tape — that study's verdict is `NONE`. This
  study never claims Monday is negative on average; it conditions on the Mondays that
  *are* negative and asks what Tuesday does next.
- [90-weekend](../90-weekend/) — the **day-of-week level table** (which single weekday
  is best/worst, unconditionally). It even *names* "turnaround Tuesday" in its own
  claim text but tests it as an unconditional Tuesday-vs-rest mean (Weak,
  *t* = +1.00) — never conditioned on the prior Monday's sign. This study is the
  **conditional** version of that same folklore phrase: E[Tuesday | Monday < 0], not
  E[Tuesday].
- [116-power-hour](../116-power-hour/) — an **intraday** continuation/reversal claim
  (does the last trading hour follow the morning?) on a completely different
  timescale and instrument set. Same reversal *family* of ideas, different clock.

None of the siblings test the **conditional** claim — a down Monday specifically
predicting a Tuesday bounce — which is this study's own axis.
