# References & literature map — Study 760 (Michigan-Sentiment-Day)

## The claim under test

- **The release itself.** University of Michigan, **Surveys of Consumers**, *Index of
  Consumer Sentiment* (1966:Q1 = 100). A **preliminary** reading is released mid-month
  (traditionally the second Friday, ~10:00 ET) and a **final** reading late-month. It is one
  of the most-watched U.S. sentiment prints; financial-media "the market moved on Michigan
  sentiment" segments are the folklore behind the **release-day drift** leg. Series home:
  `sca.isr.umich.edu`; FRED mirror **`UMCSENT`** (final) and `UMCSENT1`.
- **The contrarian bottom-timer.** *Fisher, K. L. & Statman, M. (2003), "Consumer Confidence
  and Stock Returns," Journal of Portfolio Management* — high consumer confidence predicts
  **low** subsequent S&P 500 returns and vice-versa; sentiment is a **contrarian** signal at
  the extremes. The popular distillation — *buy when sentiment is low and turning up,
  "low-then-rising marks the bottom"* — is the strongest form of the **level/regime** leg we
  test. (Warren Buffett's "be greedy when others are fearful" is the same idea in prose.)
- **Sentiment and the cross-section.** *Baker, M. & Wurgler, J. (2006), "Investor Sentiment
  and the Cross-Section of Stock Returns," Journal of Finance* — sentiment predicts returns,
  concentrated in hard-to-arbitrage stocks; a broad-index test (SPY) is the *weakest place*
  to find it, which is part of why the aggregate bottom-timer is fragile.
- **Does confidence lead spending / the cycle?** *Ludvigson, S. (2004), "Consumer Confidence
  and Consumer Spending," Journal of Economic Perspectives* — consumer confidence has only
  modest, largely coincident predictive content for the real economy; a caution against
  reading sentiment as a clean *leading* indicator for anything, equities included.

## Why the sentiment data isn't fetched live here — and what we do

- **FRED CSV endpoint firewalled.** The free `fred.stlouisfed.org/graph/fredgraph.csv?id=…`
  endpoint times out in this build's sandbox. Following the desk convention for small,
  public macro series — **Study 385 (Jobless-Claims)** hardcodes FRED `IC4WSA`, **Study 268
  (Sahm-Rule)** hardcodes `UNRATE`, **Study 358 (Watch-Index)** and **Study 708
  (Eurovision-Effect)** hardcode labelled proxy series — we **hardcode a monthly snapshot**
  of `UMCSENT` (final print, 1978→2026), as-of the 2026-04 vintage. It is the settled value,
  not the real-time preliminary vintage; that revision caveat is named on the Signal axis.
- **Release dates are a labelled proxy.** We generate the **second Friday** of each month as
  the preliminary-release schedule; the true calendar shifts a day or two around holidays.
  Labelled as a proxy, never as the official calendar.
- **Equities.** SPY daily adjusted close via **yfinance** (no key), total-return adjusted —
  daily for the event study, month-end sampled for the regime test; labelled as such.

## Why the identification hinges on overlapping-return inference

- **Overlapping long-horizon returns inflate the t.** *Richardson, M. & Stock, J. (1989),
  "Drawing Inferences from Statistics Based on Multi-year Asset Returns," Journal of Financial
  Economics*; *Boudoukh, Richardson & Whitelaw (2008), "The Myth of Long-Horizon
  Predictability," Review of Financial Studies* — 12-month overlapping monthly returns are
  strongly autocorrelated, so a naive t/Newey-West with too few lags **over-rejects**. The
  study's headline naive *t* = 3.55 is exactly this trap; the block bootstrap corrects it.
- **Block / stationary bootstrap.** *Politis, D. & Romano, J. (1994), "The Stationary
  Bootstrap," JASA*; *White, H. (2000), "A Reality Check for Data Snooping," Econometrica* —
  resampling in blocks preserves the serial dependence the inference must respect. We use a
  **circular block bootstrap** with a 12-month block, matched to the worst overlap.
- **Clustered events, not independent draws.** The 64 signal months collapse to ~**21
  independent episodes** (post-crash recoveries). A significance statistic that treats them as
  64 i.i.d. observations is the core error the study isolates.
- **Welch two-sample t.** *Welch, B. L. (1947), Biometrika* — unequal-variance test of the
  regime-set forward mean vs the unconditional mean.

## Method lineage (this study's engine)

- **Release-day event study.**
  [`strategy.release_day_summary`](../michigan_sentiment_day/strategy.py),
  [`strategy.drift_by_surprise`](../michigan_sentiment_day/strategy.py) (next-day drift by
  surprise sign, entered at the release-day close — no look-ahead).
- **Regime split + honest inference.**
  [`strategy.summarize_regime`](../michigan_sentiment_day/strategy.py) (LOW / LOW&RISING vs
  base, Welch *t*), [`strategy.block_bootstrap_p`](../michigan_sentiment_day/strategy.py) (the
  autocorrelation-aware p), [`strategy.n_episodes`](../michigan_sentiment_day/strategy.py) (the
  independent-episode count), [`strategy.timing_overlay`](../michigan_sentiment_day/strategy.py)
  (the buy-the-bottom overlay, 1-month lag, one-way costs).
- **Deterministic synthetic control.**
  [`data.synthetic`](../michigan_sentiment_day/data.py) plants a known low-then-rising→forward
  link over the next 12 months; `edge = 0` must not manufacture significance, a large `edge`
  must light up *and* the block bootstrap must fire (proving it isn't merely conservative).

## Data sources used here

- **FRED `UMCSENT`** (hardcoded monthly snapshot, 1978→2026) + **yfinance SPY** daily
  adjusted close, 1993→2026, cached under `_cache/spy.csv`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 385 — Jobless-Claims-Momentum](../385-jobless-claims-momentum/)**: the same
  hardcoded-FRED-snapshot + SPY method on the "claims lead the market" folklore — a sibling
  macro-crystal-ball teardown.
- **[Study 387 — Economic-Surprise-Index](../387-economic-surprise-index/)**: a transparent
  CESI proxy (which includes `UMCSENT`) asking whether *beats* time equities.
- **[Study 384 — ISM-PMI-Regime](../384-ism-pmi-regime/)**: another leading-indicator regime
  split, same regime-vs-base machinery.
