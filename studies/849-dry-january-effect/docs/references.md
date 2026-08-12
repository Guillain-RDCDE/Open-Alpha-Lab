# References & literature map — Study 849 (Dry January / Veganuary)

## The claim under test

- **Dry January.** A public-health campaign (popularised by *Alcohol Change UK* from 2013)
  in which participants abstain from alcohol for the whole of January. Surveys report
  millions taking part across the UK/US, and drinks makers routinely flag a soft January in
  their trading updates. The folklore trade: alcohol names (`BUD` AB InBev, `STZ`
  Constellation, `TAP` Molson Coors, `DEO` Diageo, `SAM` Boston Beer) *under*-perform in
  January as demand evaporates — and maybe *bounce* in February as abstainers fall off the
  wagon.
- **Veganuary.** A parallel campaign (the UK non-profit *Veganuary*, from 2014) pledging a
  plant-based diet for January; participation and grocery "plant-based" launches spike each
  January. The folklore trade: a plant-based name (`BYND` Beyond Meat) *out*-performs in
  January on the demand pulse.
- **The specific test here.** Because the calendar is fixed and known decades in advance, we
  run a clean **monthly calendar-seasonality** test: the mean **abnormal** return
  (`group − SPY`, stripping out the market's own January seasonality) for the alcohol basket,
  the plant name, and a staples control, across every January (and February) in the sample —
  with a one-sample *t* on the independent yearly observations, a Newey-West *t* on a January
  dummy, a Wilson hit-rate, a twelve-month calendar placebo, a two-era cut, a costed timer,
  and a seeded synthetic positive control.

## What we measure, and the honesty rails

- **Abnormal return, not raw move.** Every headline is `group − SPY` on the shared
  month-ends, so the market's own turn-of-year seasonality is differenced out and a low-beta
  staple like `XLP` is not penalised for lagging a rising market.
- **The right unit.** A calendar seasonal has one *independent* observation per year, so the
  primary *t* is a one-sample *t* across the ≤ 27 Januaries — not a daily panel *t* that would
  fake significance from within-month autocorrelation. The Newey-West dummy regression
  cross-checks with a HAC standard error.
- **Zero look-ahead by construction.** January is always January; the timer enters at the
  December close and exits at the January close, both calendar-known.
- **Small-n, named.** `BYND` lists only in 2019, giving **7** Januaries — the plant and
  plant-minus-alcohol legs are explicitly flagged as thin-sample curiosities in
  [`results.md`](results.md).
- **The timer is graded separately.** Round-trip friction on two legs plus short borrow — the
  honest test of whether a once-a-year spread survives cost and scale.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* on the January-dummy coefficient).
- **Wilson, E. B. (1927)** — score interval for a binomial hit share.
- **Welch, B. L. (1947)** — unequal-variance two-sample *t* (cross-checks used in the kit).
- **Bouman, S. & Jacobsen, B. (2002)**, *"The Halloween Indicator: Sell in May and Go Away"*
  (American Economic Review) — the canonical monthly-calendar-seasonality study; the
  market-wide analogue this single-theme test is distinct from.

## Data sources

- **yfinance daily adjusted closes** (`auto_adjust=True`, total-return), 8 tickers
  (`BUD STZ TAP DEO SAM BYND XLP SPY`), 1999-01-04 → 2026-06-30, cached under this study's own
  `_cache/` as one parquet.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [55-summer-lull](../../55-summer-lull/) — the *summer* (Jun–Aug) volume/return lull, a
  different season and a **market-wide** claim, not a January consumer-demand theme.
- [95-holiday-cheer](../../95-holiday-cheer/) — the late-December "Santa-rally" window on the
  broad market, not a themed consumer basket.
- [641-sell-in-may](../../641-sell-in-may/) — the May→October Halloween/Sell-in-May
  market-wide seasonal (Bouman-Jacobsen), again the whole market, not alcohol/plant demand.
- [723-guacamole-bowl](../../723-guacamole-bowl/) — a Super-Bowl / avocado **single-event**
  demand pulse on one date, not a whole-calendar-month theme.
- [775-halloween-candy](../../775-halloween-candy/) — a fixed-date (Oct-31) **single-name**
  Hershey run-up window, not a calendar-*month* seasonal across a themed multi-name basket.

None of the siblings test the **January abnormal return of alcohol vs plant-based consumer
names driven by the Dry-January / Veganuary campaigns** — this study's own axis.
