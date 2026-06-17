# References & literature map — Study 283 (Hurricane-Season)

## The claim under test

The folk belief: the **Atlantic hurricane season** (June 1 – November 30) drags on
US equities — either the broad market (storm damage, refinery/port shutdowns, a
risk-off mood) or, more specifically, the **property-and-casualty insurers** who
carry the catastrophe risk. Every late summer financial-media cycle revives some
version of "watch out for hurricane season." We test both the broad-market and the
insurer-basket version against the honest baseline.

## What the academic literature actually finds

- **Lanfear, M. G., Lioui, A. & Siebert, M. G. (2019).** "Market anomalies and
  disaster risk: Evidence from extreme weather events." *Journal of Financial
  Markets*, 46. Finds that hurricane *landfalls* depress returns of locally
  exposed firms in a short window, but the broad-market effect is small and quickly
  reversed — consistent with an attention/liquidity story, not a persistent
  seasonal drag.

- **Born, P. & Viscusi, W. K. (2006).** "The catastrophic effects of natural
  disasters on insurance markets." *Journal of Risk and Uncertainty*, 33(1–2).
  Documents that catastrophes raise insurer prices *afterwards* (hard-market
  pricing), so a large storm can be net-positive for surviving insurers' forward
  returns — the opposite sign to the naive "storms hurt insurers" intuition.

- **Shelor, R. M., Anderson, D. C. & Cross, M. L. (1992).** "Gaining from loss:
  Property-liability insurer stock values in the aftermath of the 1989 California
  earthquake." *Journal of Risk and Insurance*, 59(3). The classic "gaining from
  loss" result: insurer equity often *rises* after a major catastrophe as investors
  price in the subsequent premium increases.

- **Worthington, A. & Valadkhani, A. (2004).** "Measuring the impact of natural
  disasters on capital markets: an empirical application using intervention
  analysis." *Applied Economics*, 36(19). Event-study methodology for disasters;
  finds short-lived, mostly insignificant aggregate-market responses.

## Why a clean hurricane-season equity signal is unlikely

- **The season is half the calendar.** Jun 1 – Nov 30 is ~6 months, ~half the
  trading days. "Avoid the season" is therefore just a market-timing rule that sits
  out half the year — it must overcome the equity risk premium it forgoes. The
  honest baseline is buy-and-hold, not cash.

- **Seasonal-anomaly confound.** The hurricane window overlaps almost exactly with
  the "Sell-in-May / Halloween" weak-season window (May–Oct) studied by **Bouman &
  Jacobsen (2002)** ("The Halloween indicator, 'Sell in May and Go Away'",
  *American Economic Review*, 92(5)). Any in-season weakness is more parsimoniously
  attributed to that documented (and itself contested) calendar effect than to
  hurricanes per se. We do **not** claim to disentangle the two.

- **Tiny effective n for landfalls.** There are only a few dozen dateable major US
  landfalls in the modern (post-1992) catastrophe-accounting era. With ~1% daily
  equity volatility, a handful of event windows cannot resolve anything smaller
  than a several-percent abnormal return — far larger than any plausible signal.

- **Insurer "gaining from loss."** Because catastrophes trigger hard-market pricing,
  the sign of the insurer effect is theoretically ambiguous; a naive short-insurers-
  in-season rule has no firm prior to stand on.

## Method lineage

- **Welch t-test.** `scipy.stats.ttest_ind(equal_var=False)` on in-season vs
  off-season daily returns. Optimistic — it ignores autocorrelation.
- **HAC / Newey-West t-stat.** The honest standard error for the in-season mean,
  with a Bartlett kernel and the `floor(4*(n/100)^(2/9))` lag rule. Daily equity
  returns have volatility clustering and mild autocorrelation; the i.i.d. t
  overstates significance. **This HAC t on the real tape is the bar for a REAL
  verdict.**
- **Block permutation.** Rotate the in-season mask within each calendar-year block
  and recompute the spread 5,000 times; the p-value is the two-sided tail.
- **Event study.** Cumulative average abnormal return (CAAR) over [-5, +20] trading
  days around each landfall, abnormal = asset return − contemporaneous market return
  (beta fixed at 1, conservative), with a one-trading-day execution lag.

## Data sources

- **^GSPC daily.** Split-adjusted, price-only daily closes, staged at the
  repo-level `_cache/^GSPC_split_only.parquet`. Price return only (no dividends) —
  labelled on the Signal axis.
- **Insurer basket.** Equal-weight average of 8 large US P&C / reinsurance names
  (TRV, CB, ALL, PGR, AIG, HIG, CINF, WRB) via yfinance (auto-adjusted /
  total-return), staged at the study `_cache/hurricane_insurers.parquet`. The basket
  is **survivorship-biased** (current members projected backward) — a real result
  would be an upper bound. Named on the Signal axis.
- **Major-landfall table.** Hardcoded in `data.py`. Sources: NOAA National Hurricane
  Center storm reports; Aon and Swiss Re *sigma* catastrophe reports; Wikipedia
  "List of costliest Atlantic hurricanes." Insured-loss figures are widely-cited
  round magnitudes, used for context, not precision.

## Related desk studies

- **[Study 158 — Super-Bowl](../../158-super-bowl/)**: the canonical small-n
  folklore teardown whose structure this study mirrors.
- The **Sell-in-May / Halloween** calendar effect is the more parsimonious
  explanation for any in-season weakness and the proper confound to keep in mind.
