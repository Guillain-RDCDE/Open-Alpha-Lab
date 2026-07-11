# References & literature map — Study 653 (Dividend-Cut-Drift)

## The claim under test

- **The folklore.** "Never catch a falling dividend" — once a board cuts or omits its dividend,
  the stock keeps drifting down: management has just delivered the most credible negative signal
  it can send (a real cash commitment, reversed), and the market chronically underreacts to it.
  The counter-claim, just as old on the Street: a dividend cut is *backward-looking* — by the
  time it's official, the bad news is already priced, and the stock left holding the "cutter"
  label is a contrarian buy on a cleared-out cap structure.
- **The academic anchor.** Michaely, Thaler & Womack (1995, *Price reactions to dividend
  initiations and omissions: overreaction or drift?*, JF) is the canonical study: NYSE/AMEX
  dividend omissions 1964-1988 show a mean **−9.5% market-adjusted return in the following
  year**, continuing for up to three years — a genuine post-event drift, not a one-day
  overreaction. Boehme & Sorescu (2002, *The long-run performance following dividend
  initiations and resumptions*, JF) extend the initiation side. Healy & Palepu (1988,
  *Earnings information conveyed by dividend initiations and omissions*, JFE) supplies the
  fundamentals-side mechanism (subsequent earnings do deteriorate after an omission,
  consistent with the signal being informative, not noise).
- **The honest caveat, stated up front.** All three anchors are 1960s-90s samples, on a market
  structure (pre-decimalization, pre-algorithmic, thinner analyst coverage) very different from
  today's. Whether the drift **survived into the modern era** — more index arbitrage, faster
  information diffusion, deeper short interest — is exactly what this study tests, not assumes.

## What we measure, and the honesty rails

- **Detection, no look-ahead.** A *cut* = a scheduled payment ≤ 70% of the prior payment on a
  **split-adjusted** dividend stream (raw yfinance dividend history is NOT split-adjusted, and
  several names in this universe — PEP, MO, C, SLB, MS — would otherwise show a phantom
  90%+ "cut" purely from a stock split). An *omission* = a gap ≥ 1.8× the ticker's own trailing
  typical payment interval, restricted to genuinely regular (sub-200-day-cadence) payers. Both
  detectors run on a dividend stream with one-off special-dividend / spinoff / stub artifacts
  stripped (a documented spike-or-dip-then-revert heuristic — see
  [`strategy.strip_special_dividends`](../dividend_cut_drift/strategy.py) — needed live: a
  Citigroup spinoff distribution, an Exxon and a Medtronic one-off stub payment, and Wynn
  Resorts' pre-2010 special-only history all manufactured false "cuts" without it).
- **CAR vs SPY, both total-return.** Abnormal return = ticker log return minus SPY log return,
  both auto-adjusted, so the ex-date's *mechanical* price drop never contaminates the abnormal
  return (an unadjusted close would show a phantom "cut effect" on every ex-date, cut or not).
- **One execution lag, documented.** The cut/omission is only knowable from the close of the
  event day (the date the payment record shows it); every trade — and the post-drift window
  itself — starts from the close **one session later**.
- **Two inference layers.** A cross-sectional one-sample *t* on each event's CAR (the primary,
  planned test) plus a Newey-West(5) *t* on the daily calendar-time equal-weight "cutters"
  portfolio (the overlap-robust cross-check — events share calendar time, so the cross-sectional
  test's i.i.d. assumption is imperfect on its own). A 20-seed × 200-draw random ticker/date
  placebo. Hit rates carry a Wilson (1927) interval.
- **Survivorship, named on the Signal axis, not buried.** The universe is 101 tickers **alive as
  of 2026**. Names that cut a dividend and later went to zero or were delisted outright — Lehman
  Brothers, Washington Mutual, pre-2009 GM, Eastman Kodak, Circuit City, RadioShack, Bear
  Stearns — are absent by construction. Every drift estimate in this study is plausibly biased
  **upward** (too benign) relative to the true population of "companies that cut their
  dividend," because the worst outcomes never make it into a survivor-only basket.

## Why the tradable expressions are graded separately

- Both timed trades (short-the-cutter, buy-the-cutter) hold for 120 trading days (~6 months) —
  long enough that ordinary market beta dominates the *gross* number in either direction. The
  "excess" column benchmarks against the **matched-exposure** position (long SPY for the long
  leg, short SPY for the short leg) over the identical window, isolating whatever is
  cutter-specific from what is just riding the market.
- Costs: 2 × one-way × NAV per round trip (5/10 bps); shorts additionally pay a flat 50 bps/yr
  borrow, accrued over the holding period — conservative relative to most large/mid-cap
  borrow rates but a simplification (real borrow spikes hard around distress, exactly when a
  "cutter" short is most attractive on paper).
- The short leg's documented **−467% single-event** tail (a squeeze in a name that recovered
  hard after its cut) is the concrete illustration of why "short the bad news" is a much riskier
  trade than the CAR chart alone suggests.

## Data sources

- **Per-ticker adjusted closes + raw `Dividends`/`Stock Splits` streams**, and **SPY** — yfinance
  (no key), cached under `_cache/` (`dcd_<TICKER>_px.parquet`, `dcd_<TICKER>_div.parquet`),
  1996-01-02 → 2026-06-30.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [240-dividend-initiation](../240-dividend-initiation/) — the **START** side of the same
  life-cycle event (a firm's *first* dividend). Verdict there: `NONE` (too few events/year to
  measure a one-year forward premium, n=8). This study is the mirror event — the **cut/omission**
  side — with a much larger sample (172 events) precisely because cuts are more common than
  initiations in a mature-payer universe.
- [143-dividend-capture](../143-dividend-capture/) — whether you can profit from the mechanical
  **ex-dividend price drop** on a *regular, uncut* payment (buy before, sell after, pocket the
  dividend). A market-microstructure question about routine payments, unrelated to whether a
  payment was ever cut.
- [201-dividend-growth](../201-dividend-growth/) — whether **consecutive dividend raises**
  (the opposite behavior from a cut) predict forward outperformance. Verdict there: `WEAK`
  (point estimate goes the *wrong* way, −0.73%/yr). That study asks about the *growers*; this
  one asks about the *cutters* — different tail of the same distribution.
- [233-shareholder-yield](../233-shareholder-yield/) — a **composite valuation factor**
  (dividends + net buybacks, cross-sectional quintile sort) testing whether high total cash
  return predicts outperformance in the *level*, not an *event study* around a specific
  corporate action.

None of the siblings test the object of this study: **what happens to a stock's price in the
120 trading days after it specifically cuts or omits its dividend.**
