# References & literature map — Study 789 (SUE Earnings-Surprise Drift)

## The claim under test

- **The anomaly.** **Post-earnings-announcement drift (PEAD):** after a quarterly earnings print,
  prices keep drifting in the direction of the surprise for weeks. The canonical sort is on
  **standardized unexpected earnings (SUE)** — a hedge portfolio long the highest-SUE decile and
  short the lowest earns a significant drift over the ~60 trading days after the announcement.
- **The academic anchors.**
  - **Ball & Brown (1968)**, *An Empirical Evaluation of Accounting Income Numbers* (JAR) — first
    documented that prices keep adjusting *after* an announcement.
  - **Foster, Olsen & Shevlin (1984)**, *Earnings Releases, Anomalies, and the Behavior of
    Security Returns* (The Accounting Review) — formalised SUE via the seasonal random walk and
    showed the drift is monotone in SUE.
  - **Bernard & Thomas (1989)**, *Post-Earnings-Announcement Drift: Delayed Price Response or Risk
    Premium?* (JAR) and **(1990)** *Evidence that stock prices do not fully reflect the
    implications of current earnings for future earnings* (JAE) — the definitive SUE-decile drift;
    Eugene **Fama (1998, JFE)** called PEAD the "granddaddy of anomalies."
- **How we operationalise SUE.** Following Foster-Olsen-Shevlin, the *unexpected* earnings under a
  seasonal random walk is `u_q = EPS_q − EPS_{q−4}` (this quarter minus the same quarter a year
  ago). **SUE** standardizes it by the volatility of those seasonal differences known at `q`:
  `SUE_q = u_q / std(last ~8 prior u)`. Everything in the denominator is strictly lagged.
- **No look-ahead / timing.** Each event is anchored at the **10-Q/10-K filing date** reported by
  EDGAR (the date the number became public), takes the first trading session on/after it, then
  **enters one day later** and holds — so the drift is measured strictly *after* the EPS figure is
  disclosed (the standard event-study convention, and the study's single documented execution lag).

## Why a flat / null result here is the *expected* outcome

- **Limits to arbitrage / liquidity.** Chordia, Goyal, Sadka, Sadka & Shivakumar (2009,
  *Liquidity and the post-earnings-announcement drift*, FAJ) show the whole drift family
  concentrates in **small, illiquid** names and largely vanishes among liquid large-caps — exactly
  the conservative universe we use by construction.
- **Post-publication decay.** McLean & Pontiff (2016, *Does academic research destroy stock return
  predictability?*, JF) and Harvey, Liu & Zhu (2016, *…and the Cross-Section of Expected Returns*,
  RFS) document that published anomalies weaken sharply out of sample. Bernard-Thomas is 35+ years
  old; a null on a 2012–2026 large-cap survivor sample is consistent with both liquidity limits and
  decay. Our result — a *non-monotone*, sign-flipping, robust-*t*-near-zero long-short — is that
  predicted null.

## Why a high *t* would still need a placebo + clustering check

- **One-sample / Welch *t*** (Welch 1947) for the long-short mean against zero. But filings cluster
  in **earnings seasons**, so naive per-event *t*-stats overstate significance (indeed our naive
  one-sample *t* is a spurious −2.3 that the robust checks dissolve). We therefore lead with a
  **calendar-time long-short + Newey-West (1987) HAC *t*** on ~50 earnings-season buckets, and add
  a **within-quarter block placebo** (Fisher randomization; Efron & Tibshirani 1993) that respects
  the clustering, plus a global label-shuffle placebo and a Wilson (1927) interval on the win-rate.

## Method lineage (the desk's shared engine)

- **Tercile long-short + one-sample *t*.** [`strategy.long_short_drift`](../sue_drift/strategy.py)
  / [`strategy.ttest_vs_zero`](../sue_drift/strategy.py).
- **Calendar-time Newey-West HAC.** [`strategy.calendar_time_ls`](../sue_drift/strategy.py) /
  [`strategy.newey_west_t`](../sue_drift/strategy.py) — the autocorrelation-robust headline.
- **Label-shuffle & block placebo.** [`strategy.placebo_pvalue`](../sue_drift/strategy.py) and
  [`strategy.block_placebo_pvalue`](../sue_drift/strategy.py).
- **Deterministic synthetic control.** [`data.synthetic_sue`](../sue_drift/data.py) plants a known
  post-event drift proportional to the surprise; with the edge set to zero the inference must NOT
  manufacture significance (0/20 seeds fire), and a planted edge lights up (t = +21.5).

## Data sources used here

- **yfinance** daily adjusted closes for a fixed 30-name large-cap basket.
- **EDGAR** `companyconcept` XBRL API (`data.sec.gov`) — frame-tagged quarterly **diluted EPS**
  (`EarningsPerShareDiluted`, falling back to `EarningsPerShareBasic`), with the 10-Q/10-K filing
  date. Reuses the desk-wide `_cache/edgar_eps.parquet` (`tools/fetch_altdata.py "EDGAR"`) when
  present, else fetches per-CIK directly. Cached under this study's `_cache/sue_prices.csv` and
  `_cache/sue_events.csv`. All headline numbers are pinned in [`docs/results.md`](results.md) and
  reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- **[363 — PEAD-Drift](../363-pead-drift)** — measures the drift after sorting on the **price-gap
  reaction** (the announcement-day return / CAR), a *price* proxy for the surprise with **no
  fundamentals**. This study sorts on the **fundamental SUE** built from reported EPS.
- **[369 — Earnings-Revision-Momentum](../369-earnings-revision-momentum)** — sorts on **analyst
  estimate revisions** (a forward, sell-side signal). This study uses no estimates at all — the
  surprise is the seasonal-random-walk EPS change, an *ex-post* accounting number.
- **[534 — Revenue-Surprise-Drift](../534-revenue-surprise-drift)** — the same machinery on the
  **revenue (sales)** surprise (SUR). This is the **EPS** surprise (SUE) — the original
  Bernard-Thomas object, not the Jegadeesh-Livnat revenue variant. On this same conservative
  large-cap survivor basket, both come up empty — the contrast is the point.
