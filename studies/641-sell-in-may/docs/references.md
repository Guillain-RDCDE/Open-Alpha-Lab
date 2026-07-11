# References & literature map — Study 641 (Sell in May)

## The claim under test

- **The folklore.** "Sell in May and go away" — U.S. (and most developed-market) equities do
  almost all of their work Nov→Apr; May→Oct is dead money, so a disciplined investor should hold
  stocks in winter and step aside in summer.
- **The academic anchor.** **Bouman & Jacobsen (2002, *The Halloween Indicator, "Sell in May and
  Go Away": Another Puzzle*, American Economic Review)** documented the Nov–Apr outperformance
  across 36 of 37 countries studied, calling it "one of the most stubborn calendar anomalies
  ever published" — durable across markets and pre-dating any known publication-driven decay.
  Subsequent literature (Jacobsen & Zhang 2018, extending the sample back to 1693 for the UK;
  Andrade, Chhaochharia & Fuerst 2013 on persistence post-publication) is generally supportive
  of the *pattern's existence*, less clear on whether it is exploitable after costs.
- **What we add.** The literature usually reports the raw seasonal split. This study adds a
  **year-block bootstrap and sign test** (so no single overlapping-window artefact inflates the
  significance), an explicit **decomposition of how much of the gap is a handful of famous crash
  autumns**, and — the beat most write-ups skip — an honest, cost-charged **timer backtest on a
  genuine total-return tape**, racing **excess-of-cash Sharpe**, not raw CAGR.

## What we measure, and the honesty rails

- **Winter (Nov→Apr) vs summer (May→Oct) monthly log return** — Welch *t* (unequal-variance
  group split, the planned primary) and a Newey-West (1987) 3-lag HAC *t* on the winter-dummy
  regression as the serial-correlation-robust cross-check.
- **Deep history vs the tradable tape, both reported.** ^GSPC's long price-only history
  (1950–2026, matching the brief) clears the desk's *t* ≥ 2 bar; the dividend-inclusive tapes
  that describe an actual portfolio (^SP500TR since 1988, SPY since 1993) do not. Per house
  precedent (see [89-turn-of-the-month](../89-turn-of-the-month/)), a deep price-only sample
  clearing the bar while the modern, economically correct sample cannot certify it **reads
  WEAK, not REAL** — the literature-adjacent long tape is supportive, not decisive on its own.
- **Year-block bootstrap, not month-level.** Resampling individual months would let a bootstrap
  draw straddle a season boundary and destroy the very correlation structure ("does winter beat
  summer *within the same year*?") the test is supposed to respect; resampling whole Halloween
  years fixes that. A sign test with a Wilson (1927) interval is the model-free companion.
- **The "handful of bad autumns" decomposition** is a *diagnostic*, not a snoop — the sample is
  never re-run excluding these months for the headline number; it is reported once, explicitly,
  as its own line in the results.

## Why the tradable timer is graded separately, and against cash

- The timer's Sharpe is computed **excess of cash on both legs** (timer and buy-and-hold) — a
  strategy that sits in cash part of the year must not race a raw buy-and-hold Sharpe, which
  would flatter the cash-heavy leg's lower volatility without pricing what it gave up. House
  rule: compare excess-of-cash to excess-of-cash.
- **Price-only vs total-return, stated as a decision.** ^GSPC (used for the deep-history
  significance test to match the brief's "1950→last complete month" instruction) carries **no
  dividends**. Racing a *timer that substitutes cash for equity in summer* against a *price-only
  buy-and-hold* mechanically favors the timer (it swaps a no-dividend equity leg for a
  positive-yielding cash leg, while buy-and-hold never collects dividends at all) — an artefact
  named explicitly in `docs/results.md`, not hidden. The tradability verdict is drawn from the
  dividend-inclusive tapes (SPY, ^SP500TR) instead.
- **^IRX is a discount-yield quote, not a total return** — converted to an approximate monthly
  cash rate (`yield / 100 / 12`); a real money-market fund would earn a similar but not
  identical number after its own costs. Named, not hidden.
- Costs are charged one-way × NAV per leg (5/10 bps), 2 × per switch, 2 switches a year — a
  *cheap* execution assumption (a fixed calendar rule needs no urgency and no market impact),
  which makes the MIRAGE verdict conservative, not generous.

## Data sources

- **^GSPC daily Close (price-only)**, **SPY daily Close (dividend-adjusted)**, **^SP500TR daily
  Close (total-return index)** and **^IRX daily Close (13-week T-bill discount yield)** —
  yfinance (no key), cached under `_cache/` (`sim_gspc.csv`, `sim_spy.csv`, `sim_sp500tr.csv`,
  `sim_irx.csv`), 1950-01-03 → 2026-06-30 (SPY from 1993-02, ^SP500TR from 1988-01, ^IRX from
  1960-01).
- Bouman, D. & Jacobsen, B. (2002). *The Halloween Indicator, "Sell in May and Go Away": Another
  Puzzle*. American Economic Review, 92(5), 1618–1635.
- Jacobsen, B. & Zhang, C. Y. (2018). *The Halloween Indicator: A Statistical Fluke, or an
  Effect That Is Deeply Rooted?* — the long-history (UK, back to 1693) robustness follow-up.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [55-summer-lull](../55-summer-lull/) — the **closest sibling**: the same Halloween claim, on
  ^GSPC 1928–2026 (auto_adjust=True), reaching a compatible **WEAK / MIRAGE / Not-supported**
  verdict at Welch *t* = 1.3. This study **deepens, not repeats, that finding**: a stricter
  inference stack (year-block bootstrap + sign test, not just the group Welch *t*), an explicit
  price-only-vs-total-return decomposition (55 uses one auto-adjusted tape throughout; this
  study runs ^GSPC price-only, SPY and ^SP500TR **separately** and shows they disagree), the
  "handful of bad autumns" decomposition (new), and a timer that races **excess-of-cash Sharpe**
  with named turnover/costs rather than a single buy-and-hold race.
- [89-turn-of-the-month](../89-turn-of-the-month/) — a different calendar seasonal (the first/
  last few trading days of each month), not the Nov–Apr/May–Oct split — but the direct
  **methodological precedent** for how this study resolves the deep-history-clears / modern-
  sample-doesn't split (both call it WEAK, not REAL).
- [290-september-effect](../290-september-effect/) — tests whether **September alone**
  underperforms the other eleven months; a single-month question. This study asks about the
  **six-month Nov–Apr vs May–Oct block**, and separately shows September is the one calendar
  month that drags the summer half down (see the by-calendar-month table) — a complementary
  finding, not a duplicate test.
- [136-mark-twain](../136-mark-twain/) — tests the "October is dangerous" myth specifically
  (busted: October's 76-year mean is positive). This study's "handful of bad autumns" section
  independently confirms October (and September) carry the fattest summer-half crash tail
  without re-running Mark Twain's single-month test.

None of the siblings run the year-block bootstrap, the price-only-vs-total-return disagreement,
or the excess-of-cash Halloween-timer race — this study's own contribution on top of 55's
already-honest verdict.
