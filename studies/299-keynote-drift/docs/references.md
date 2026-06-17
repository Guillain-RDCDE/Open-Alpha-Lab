# References & literature map — Study 299 (Keynote-Drift)

## The claim under test

**"Buy the rumor, sell the news."** A piece of market folklore as old as
markets, recycled before every Apple keynote: AAPL allegedly drifts higher into
a keynote (WWDC, the September iPhone event) on anticipation, then sells off once
the products are revealed. There is no single academic paper; the claim lives in
financial media, trader chat, and recurring "how to trade the Apple event"
pieces. We test it as a clean event study on AAPL daily returns, benchmarking the
stock against its own unconditional mean so its enormous 2008–2025 trend does not
masquerade as a keynote effect.

## Why the illusion is so persuasive — and why it fails the honest test

- **The benchmark trap.** AAPL rose ~50x over 2008–2025. With a strong positive
  drift, *every* short window — keynote or not — has a positive average return.
  Pointing at the raw pre-keynote run-up and calling it a "rumor" effect is a
  base-rate error. The correct benchmark is AAPL's own unconditional daily mean
  (a constant-mean market model); the abnormal CAR is then ~zero.

- **Small-n on a single name.** There are ~48 keynotes in the modern era, each a
  noisy single-stock window (~3.5% standard deviation over 5 days). The minimum
  detectable mean CAR at |t| = 2 is ~1.4%/event — far above any plausible drift.

- **Many windows, many slices.** WWDC vs September vs Spring; pre vs post; 5-day
  vs 10-day. Testing many variants inflates the chance of a spurious "hit." The
  one mildly negative slice (September post-event) does not survive its own
  subset t-test.

## Method lineage — event studies and abnormal returns

- **Fama, E. F., Fisher, L., Jensen, M. & Roll, R. (1969).** "The Adjustment of
  Stock Prices to New Information." *International Economic Review*, 10(1), 1–21.
  The original event-study methodology: align returns to event time, accumulate
  abnormal returns around the event date.

- **MacKinlay, A. C. (1997).** "Event Studies in Economics and Finance."
  *Journal of Economic Literature*, 35(1), 13–39. The standard reference for
  cumulative abnormal returns (CARs), the constant-mean and market models, and
  the t-tests we use on per-event CARs.

- **Brown, S. J. & Warner, J. B. (1985).** "Using Daily Stock Returns: The Case
  of Event Studies." *Journal of Financial Economics*, 14(1), 3–31. Documents the
  pitfalls of daily-return event studies (non-normality, clustering) and supports
  simple per-event-CAR tests and non-parametric checks like our permutation null.

## Literature on data-snooping and small-sample mirages

- **Sullivan, R., Timmermann, A. & White, H. (1999).** "Data-Snooping, Technical
  Trading Rule Performance, and the Bootstrap." *Journal of Finance*, 54(5),
  1647–1691. How searching many calendar/technical rules inflates apparent
  significance; the correct benchmark accounts for the implicit search.

- **Harvey, C. R., Liu, Y. & Zhu, H. (2016).** "… and the Cross-Section of
  Expected Returns." *Review of Financial Studies*, 29(1), 5–68. Argues the t-stat
  hurdle for a credible anomaly is ~3.0, not 2.0 — a single-name keynote drift
  with t < 0.5 is not remotely close.

## Related desk studies

- **[Study 158 — Super-Bowl](../../158-super-bowl/)**: the folklore-table pattern
  this study mirrors — a hardcoded event table tested against the honest base rate.
- **[Study 223 — Same-Month-Seasonality](../../223-same-month-seasonality/)**: the
  cache-only real-tape (yfinance) pattern, with survivorship/selection named on the
  Signal axis.

## Data sources

- **AAPL daily adjusted close.** Yahoo! Finance via yfinance (`auto_adjust=True`),
  price-only (split- and dividend-adjusted), 2007-06-01 → 2025-12-30. Cache-only
  by default at `_cache/aapl_daily.parquet`; the network is touched solely on an
  explicit `fetch_prices(fetch=True)`.
- **Apple keynote table.** Hardcoded in `data.py`. Sources: Apple Newsroom press
  archive, Wikipedia "Apple Worldwide Developers Conference" and "Apple special
  event" pages. Each row is the main day of the event.

## A note on selection / survivorship

This is a single, surviving mega-cap, so classic survivorship bias does not bite
the price series. But the **keynote calendar is curated ex-post** and covers only
the modern (post-2008) iPhone era — a selection that is named on the Signal axis.
Pre-2008 Macworld keynotes (a different regime, smaller company) are deliberately
excluded.
