# References & literature map — Study 800 (High-Frequency / Weekly Reversal)

## The claim under test

- **The folklore/claim.** A **weekly** cross-sectional reversal: sort a liquid US
  cross-section on last week's five-day return; last week's biggest *losers* beat last
  week's biggest *winners* next week. It is pitched as the amplified, cleaner, faster-
  turning cousin of the one-month reversal — shorter horizon, higher turnover, "more signal
  per unit time." The open question this study answers is whether the extra signal is real
  or is simply **more bid-ask bounce** at the faster clock.
- **The academic anchor.** Short-horizon return reversal is documented at the weekly
  horizon by **Lehmann (1990, *Fads, Martingales, and Market Efficiency*, QJE)** — a
  contrarian portfolio formed on the prior week's returns earns positive returns — and at
  the monthly horizon by **Jegadeesh (1990, *Evidence of Predictable Behavior of Security
  Returns*, JF)**. Both flagged, and later work confirmed, that a large part of the raw
  reversal is **bid-ask bounce / non-synchronous trading**, not economic mean-reversion.
- **The microstructure critique — the spine of this study.** **Lo & MacKinlay (1990,
  *When Are Contrarian Profits Due to Stock Market Overreaction?*, RFS)** decompose
  contrarian profits and show a substantial share is mechanical. **Roll (1984)** derives
  the effective spread from the negative serial covariance of price changes — the exact
  bounce that fakes a one-period reversal. **Corwin & Schultz (2012, *A Simple Way to
  Estimate the Bid-Ask Spread from Daily High and Low Prices*, JF)** give the high-low
  spread estimator we use for the empirical bounce haircut. **Asparouhova, Bessembinder &
  Kalcheva (2013)** show bid-ask bounce biases mean returns and how to correct it.
- **Post-publication decay.** **McLean & Pontiff (2016, *Does Academic Research Destroy
  Stock Return Predictability?*, JF)** — anomalies fade after publication as arbitrageurs
  compete the rent away; our 2010→2026 sub-period split tests exactly this at the weekly
  horizon.

## What we measure, and the honesty rails

- **The loser-minus-winner quintile spread**, dollar-neutral, formed each Friday on last
  week's return and held one week — the Lehmann (1990) specification. Reported with a
  **Newey-West HAC *t*** (weekly overlapping-book returns are mildly autocorrelated) as the
  primary, plus loser/winner leg stats with a **Wilson (1927)** hit-rate interval.
- **Killer #1 — the skip-a-week gap.** ``skip=1`` inserts one week between the formation
  close and the holding week, so the same close cannot both form the signal and price the
  entry. If the edge only survives ``skip=0`` it is bid-ask bounce, not a tradable signal.
  This is the single decisive microstructure control.
- **Killer #2 — the empirical bounce haircut.** Each leg is charged its **own**
  Corwin-Schultz effective spread on its weekly turnover (illiquid losers pay more than
  winners), so the haircut is not a flat basis-point guess but a name-specific liquidity
  cost — plus a flat cost sweep and a short-borrow charge.
- **No look-ahead**: one shift, applied once in ``trailing_return`` — the signal for
  holding week t is the return of week ``t-1-skip``, known at that close; the return earned
  is week t's. A **random-portfolio null** (loser-leg size, drawn each week) confirms any
  loser excess is signal, not concentration.
- **Survivorship is named on the Signal axis.** The universe is the *current* S&P 500
  projected backwards; delisted names — the very losers a reversal book holds — are absent,
  so every positive spread is an upper bound.

## Data sources

- **Weekly total-return close panel** — the shared daily S&P 500 OHLCV panel already in the
  repo (`studies/01-overnight-anomaly/.../panel_*.parquet`, yfinance `auto_adjust=True`),
  resampled to `W-FRI` and cached under this study's own `_cache/` (`hfr_weekly_close.parquet`,
  `hfr_weekly_spread.parquet`), 2010-01-08 → 2026-05-29.
- All headline numbers are pinned in [`results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py) (fingerprint `e9d0f3b08c1c`).

## Related desk studies (the dedup map — what this study is NOT)

- [329-one-month-reversal](../../329-one-month-reversal/) — the **monthly** Jegadeesh
  (1990) reversal (rank on last *month*'s return, hold a month). Same family, a **4×-slower
  clock**; that study finds the monthly spread real on the raw tape but killed by a one-
  *month* skip and decayed post-2002. This study is the **weekly** horizon with the same
  microstructure autopsy — and finds the bounce contamination is, if anything, sharper.
- [196-long-term-reversal](../../196-long-term-reversal/) — the **De Bondt-Thaler** 3-to-5
  *year* reversal (the opposite end of the autocorrelation spectrum: multi-year losers
  rebound). No microstructure issue at that horizon; a completely different mechanism.
- [538-industry-relative-reversal](../../538-industry-relative-reversal/) — reversal
  measured **relative to a stock's industry** (fade the idiosyncratic, not the total,
  weekly move). This study is the **plain total-return** weekly cross-section, and its
  focus is the **bid-ask-bounce haircut** rather than the industry demeaning.

None of the siblings pair the **weekly** horizon with the **skip-a-week + empirical
Corwin-Schultz bounce haircut** microstructure autopsy — this study's own axis.
