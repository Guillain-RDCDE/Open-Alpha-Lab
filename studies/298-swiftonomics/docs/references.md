# References & literature map -- Study 298 (Swiftonomics)

## The claim under test

"Swiftonomics" is the popular-press thesis that Taylor Swift's *Eras Tour*
(2023-2024) was a macro-scale economic force. The financial-markets version we
can actually test on a clean tape: each tour news event should have produced a
measurable, tradable jump in **Live Nation Entertainment (LYV)** -- the concert
promoter and Ticketmaster parent that sold and operated the tour. The folklore
trade: "buy LYV when Taylor announces a leg."

- **Common-Wealth / press coverage of "Swiftonomics."** Bloomberg, Reuters, WSJ,
  and the QuestionPro consumer-spending estimate ($5bn+ in US consumer spending)
  popularized the term in 2023. The US Federal Reserve's *Beige Book* (July 2023,
  Philadelphia district) even noted that Taylor Swift concerts boosted hotel
  revenue -- the high-water mark of the macro claim.

## Why the LYV event signal is the testable core

- **Event-study methodology.** MacKinlay, A. C. (1997). "Event Studies in
  Economics and Finance." *Journal of Economic Literature*, 35(1), 13-39. The
  canonical reference for the market-model abnormal-return / CAR / CAAR design we
  use: estimate beta on a clean pre-event window, subtract the market-explained
  return inside the event window, cumulate, and test the cross-event mean.

- **Brown, S. J. & Warner, J. B. (1985).** "Using daily stock returns: The case
  of event studies." *Journal of Financial Economics*, 14(1), 3-31. Establishes
  the daily-return event-study conventions (estimation window length, the
  cross-sectional t-test on CARs, the importance of a non-overlapping estimation
  window) that this study follows.

- **Semi-strong efficiency.** Fama, E. F. (1970). "Efficient Capital Markets: A
  Review of Theory and Empirical Work." *Journal of Finance*, 25(2), 383-417. A
  pre-announced, unambiguous, single-name event edge would violate semi-strong
  efficiency -- exactly the kind of "obvious" signal markets price instantly.

## Why the effect is absent (and mildly negative)

- **The news was anticipated.** The tour's commercial success was widely
  expected; by the time each leg was *announced*, the incremental revenue was
  already in consensus estimates. An event study measures *surprises*, and there
  was little surprise left to capture.

- **The loudest event was a regulatory negative.** The November 2022 Ticketmaster
  pre-sale meltdown triggered Congressional hearings and ultimately a 2024 US
  Department of Justice / state antitrust lawsuit seeking to break up Live Nation
  and Ticketmaster. That single event -- the most prominent Eras Tour news -- was
  *bad* for LYV (CAR around -1300 bps in our window), which is why the
  announcement subset is the most negative of all.

- **Single-name noise.** LYV daily idiosyncratic volatility is large (beta ~1.15
  to the S&P, plus a wide residual). With only 16 events, a small mean effect
  cannot be distinguished from noise (placebo p ~ 0.07, cross-event |t| < 2).

## Method lineage

- **Market model.** OLS `r_LYV = alpha + beta * r_mkt` on a 120-day estimation
  window ending 5 trading days before each event window. Abnormal return
  `AR_t = r_LYV_t - (alpha + beta * r_mkt_t)`.
- **CAR / CAAR.** Cumulative abnormal return per event over `[-1, +3]`; the
  cross-event average is the CAAR.
- **Cross-event t-test.** Each event's CAR is one observation; `scipy.stats.t`
  two-sided test of `H0: mean CAR = 0`. This is the Brown-Warner cross-sectional
  test.
- **Newey-West HAC** t-stat on the daily AAR path, to respect serial correlation
  within the event window.
- **Placebo test.** Re-run the event study on thousands of random event-date sets
  of the same size; the placebo p-value is the share of placebo mean-CARs at least
  as extreme as the observed.

## Data sources

- **LYV + ^GSPC daily adjusted closes.** Yahoo! Finance via `yfinance`,
  2021-06-01 to 2025-05-30 (~1005 trading days), `auto_adjust=True`. LYV pays no
  dividend over the sample (price = total return); ^GSPC is price-only. Cached at
  `_cache/lyv_gspc.parquet`, cache-only by default (network only on `fetch=True`).
- **Eras Tour event table.** Hardcoded in `data.py`. Sources: Live Nation press
  releases & SEC filings, Billboard, Pollstar, Variety, and mainstream financial
  press for the tour announcement, ticketing dates, box-office records, concert
  film, and finale.

## Related desk studies

- **[Study 158 -- Super-Bowl](../../158-super-bowl/)**: the same "obvious cultural
  event moves the market" structure, tested with the correct null and tiny-n power
  analysis.
