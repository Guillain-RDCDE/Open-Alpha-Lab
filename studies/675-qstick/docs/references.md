# References & literature map — Study 675 (Qstick)

## The claim under test

- **The indicator.** **Tushar Chande's Qstick** (Chande & Kroll, *The New Technical Trader*,
  Wiley, 1994) is a moving average of the daily "candle body" — close minus open — over
  ``N`` days (Chande's own examples use an 8-day window; other charting packages default to
  10 or 20). A run of green-bodied closes lifts Qstick above zero ("buyers in control"); a run
  of red bodies pushes it below.
- **The folklore.** Charting sites (StockCharts.com, Investopedia, TradingView script
  libraries) teach Qstick's **zero-cross** as a trend-timing signal: crossing up through zero
  means buying pressure has taken over and a rise should follow, mirroring the pitch made for
  Balance of Power (Livshin, 2001) and the Force Index (Elder, 1993) — the whole "who won
  today's bar" family of oscillators.
- **The steelman question we test.** Does Chande's close-minus-open smoothing capture something
  a plain trailing price average does not — or is it, as the brief for this study puts it, "just
  a slow trend proxy"?

## What we measure, and the honesty rails

- **Zero-cross entry, one documented lag.** Smoothed Qstick is a causal (trailing-only) 8-day
  SMA; a long fires when it crosses up through zero on the close of *t*, entered at the **next**
  close, forward H-day returns measured at H ∈ {5, 10, 20, 60}.
- **Drift-matched random-entry baseline.** Long-only entries on an upward-drifting index look
  profitable by construction; the honest test is cross-vs-**random** (same instrument, same
  epoch, same hold), with a Welch *t* on the difference — not the one-sample *t* against zero,
  which only measures the tide.
- **Sign-scramble ordering placebo.** Permute the per-bar bodies in time (keeping the marginal
  distribution) so the smoothed series and its crosses become temporally meaningless — the
  honest "does the body **sequence** carry information?" null, following the same design as
  sibling study 473's BOP placebo.
- **The trend-proxy structural check.** By construction, Qstick_N = SMA_N(close) − SMA_N(open);
  when the open tracks the prior close closely (routine for liquid, continuously-traded ETFs),
  this telescopes toward the trailing N-day average price change — a signal that never looks at
  the open at all. We measure the empirical correlation directly and race the naive
  momentum-only cross against the same random baseline, rather than asserting the algebra.
- **Normalisation, named.** We divide the raw close-open body by the prior close so the
  indicator is comparable across instruments at different price levels (SPY ~$400+ vs GLD
  ~$180+); this rescaling by a positive constant never changes which bars are positive/negative,
  so it cannot manufacture or destroy a zero-cross relative to Chande's raw price-unit version.

## Data sources

- **SPY, QQQ, IWM, DIA, GLD daily OHLC** (yfinance, no key, auto-adjusted total return),
  2005-01-03 → 2026-06-30, cached under `_cache/` (`bars_<TICKER>_1d.parquet`). Same basket as
  sibling study [473-balance-of-power](../473-balance-of-power/) for a direct, apples-to-apples
  comparison across the "who's-in-control" oscillator family.
- Chande & Kroll, *The New Technical Trader* (Wiley, 1994) — the Qstick formula and its
  original 8-day default window.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py); the real-tape data stamp is printed via
  `quantlab.repro.data_stamp` for every ticker.

## Related desk studies (the dedup map — what this study is NOT)

- [423-force-index](../423-force-index/) — Elder's Force Index, `(close − close_prev) ×
  volume`, EMA-smoothed. Same "who won the bar" intuition, but keyed on the *change* in price
  weighted by *volume*, not on the bar's own body. Also lands None × Mirage — a different
  formula reaching the same conclusion, not a repeat of this study.
- [473-balance-of-power](../473-balance-of-power/) — Livshin's BOP, `(close − open) /
  (high − low)`: the body **normalised by the bar's own range**, so it's bounded in [−1, +1]
  and reads as a *share* of the day fought over. Qstick's body is normalised by the **prior
  close** instead (a price-change proxy, not a range-share proxy) — a related but distinct
  read-out, tested here with the identical honesty rails (random-entry baseline,
  sign-scramble placebo) for a clean side-by-side.
- [185-chande-momentum](../185-chande-momentum/) — the **same author's** other oscillator, the
  Chande Momentum Oscillator (net momentum from up-sum/down-sum over a window). Unrelated
  formula (no open/close term at all); this study's dedicated close-minus-open claim is
  entirely separate from CMO's momentum-normalisation claim, which also lands None × Mirage.
- [129-heikin-ashi](../129-heikin-ashi/) — trades a **smoothed candle's own colour flip**
  (HA_close vs HA_open), a recursive resmoothing of the whole OHLC bar rather than a moving
  average of a single close-minus-open statistic. Different mechanism, same "smoothing implies
  signal" folklore, same landing.
- [421-williams-alligator](../421-williams-alligator/) — three offset moving averages of the
  median price; no body/range term anywhere. Unrelated mechanism; listed only because it was
  flagged as a possible dedup target for this study and is, on inspection, a different claim
  entirely (median-price MA spread, not open/close body).

None of the siblings test **Chande's specific close-minus-open smoothing and its zero-cross**;
that is this study's own axis, run with the same drift-vs-random and ordering-placebo honesty
rails as its closest cousin (473) so the two studies are directly comparable.
