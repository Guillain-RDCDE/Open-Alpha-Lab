# References & literature map — Study 368 (Buyback-Drift)

## The claim under test

- **The folklore.** A staple of financial media and retail commentary: *"company X just
  authorized a $Y billion buyback — the stock will drift up for months."* The intuition is
  that a board authorizing repurchases is "management buying its own stock," a bullish signal
  the market should chase. The believers' version is a **multi-month abnormal drift** that
  starts on the announcement and that you can position into.
- **The crucial distinction — authorization vs execution.** A buyback **authorization** is a
  board/press-release *option* to repurchase up to $Y; it is **not** a purchase. Actual
  execution trickles out over *years*, is discretionary, and frequently lapses unused. The
  headline reaction — what the drift story is about — is to the *authorization*, so that is
  what we time the event on. (Stephens & Weisbach, 1998, *Actual Share Reacquisitions in
  Open-Market Repurchase Programs*, Journal of Finance, document that completion rates are far
  below authorized amounts and highly variable — the announcement is a weak commitment.)

## What the academic literature actually says

- **The original buyback-announcement event study.** Ikenberry, Lakonishok & Vermaelen (1995),
  *Market Underreaction to Open Market Share Repurchases*, Journal of Financial Economics 39 —
  the canonical paper claiming a **long-run drift** of ~12% over four years post-announcement,
  concentrated in value ("glamour vs value") stocks. This is the academic backbone of the
  folklore. Note the effect is (a) *long-horizon* (years), (b) concentrated in a *style*
  subset, and (c) sample-period and risk-model dependent.
- **The replication and decay literature.** Subsequent work shows the announcement drift is
  fragile: Peyer & Vermaelen (2009), *The Nature and Persistence of Buyback Anomalies*, Review
  of Financial Studies, find it survives but is driven by *undervaluation/contrarian* signals,
  not the buyback per se. Fama & French (1996, 2008) and the broader anomaly-decay literature
  (McLean & Pontiff, 2016, *Does Academic Research Destroy Stock Return Predictability?*,
  Journal of Finance) show that published anomalies — buyback drift among them — shrink sharply
  out-of-sample and post-publication.
- **The short-run announcement return.** A modest **positive announcement-day abnormal return**
  (~2–3%) is well documented (Comment & Jarrell, 1991, *The Relative Signalling Power of
  Dutch-Auction and Fixed-Price Self-Tender Offers and Open-Market Share Repurchases*, Journal
  of Finance). That is a *jump*, not a multi-month *drift*, and a study that enters one day
  **after** the announcement has already missed it.
- **Why a hand-picked headline sample is hard to certify.** Famous, large authorizations are
  selected on visibility and on the firm having survived — a textbook selection problem (Harvey,
  Liu & Zhu, 2016, *…and the Cross-Section of Expected Returns*, Review of Financial Studies, on
  the multiple-testing / selection bar a single famous effect must clear).

## Why our sample is a hard-coded table — and what we do instead

- **No free point-in-time authorization feed.** Clean, point-in-time press-release timestamps
  for thousands of repurchase authorizations are not available through yfinance (per-ticker
  OHLCV only). We therefore hard-code a **transparent, named table of ~30 notable
  authorizations** (mega-cap, well-documented announcement dates and headline sizes), and we
  say so on the Signal axis. The lesson is statistical: a few-dozen single-name events are
  dominated by their own idiosyncratic variance, and that holds a fortiori for the *true*
  population once it includes the small, unglamorous programs the headlines skip.
- **Abnormal returns, not raw.** We measure **abnormal** drift = event-stock return *minus
  SPY* over the same window, so "the stock went up because the whole market rallied" cannot
  masquerade as buyback drift.

## Why ~30 single-name events cannot be an edge — the statistics

- **Small-sample inference / power.** With *k* ≈ 30 single-name events whose forward returns
  carry full single-stock volatility, the standard error of the mean abnormal return is large;
  a few-percent drift cannot be distinguished from zero. We test the mean against zero with a
  **one-sample t** and, because the names are heterogeneous, with a **same-names placebo /
  randomization test** — re-drawing each event's entry on a random date *for the same ticker*
  and asking how often chance matches the announcement set (Fisher's randomization logic; Efron
  & Tibshirani, 1993, *An Introduction to the Bootstrap*). This controls for each stock's own
  drift and vol, not merely the market's.
- **Win-rate vs the base rate.** A single-name abnormal return is roughly a coin-flip around
  zero, so a ~55% win-rate is ≈1 SE of a proportion from the 50% null — not evidence of an edge.

## Method lineage (the desk's shared engine)

- **Abnormal forward returns with execution lag.**
  [`strategy.abnormal_returns`](../buyback_drift/strategy.py) enters one day after the
  announcement (no look-ahead) and subtracts SPY over the same window.
- **One-sample t + same-names placebo p-value.**
  [`strategy.welch_t`](../buyback_drift/strategy.py) and
  [`strategy.placebo_pvalue`](../buyback_drift/strategy.py) — the Signal-axis tests.
- **Deterministic synthetic control.**
  [`data.synthetic_events`](../buyback_drift/data.py) injects a known abnormal drift per event;
  the offline core runs with no network and confirms the inference recovers a planted edge and
  refuses to manufacture significance when the true edge is zero.

## Data sources used here

- **yfinance** daily adjusted closes for SPY + the ~30 event tickers, 2010-01-04 →
  2026-06-18, cached under `_cache/event_prices.csv`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 240 — Dividend-Initiation](../../240-dividend-initiation/)** and
  **[Study 197 — Dividend-Payout-Ratio](../../197-dividend-payout-ratio/)**: the payout-policy
  cousins — does *returning cash* (dividends instead of buybacks) carry a return signal?
- **[Study 228 — Pre-Earnings-Runup](../../228-pre-earnings-runup/)**: another single-name
  event-drift claim tested on a small hand-built event set — the same small-sample pathology.
