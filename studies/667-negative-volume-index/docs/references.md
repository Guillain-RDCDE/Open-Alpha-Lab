# References & literature map — Study 667 (Negative Volume Index)

## The claim under test

- **The folklore.** Norman G. Fosback, *Stock Market Logic: A Sophisticated Approach
  to Profits on Wall Street* (The Institute for Econometric Research, 1976). Fosback
  built the **Negative Volume Index (NVI)** — cumulate the index's daily return only on
  days total volume FALLS versus the prior day, freeze the line otherwise — on the
  theory that "smart money" (informed, patient capital) does its buying and selling
  quietly, on low-volume days, while the crowd chases headlines and pushes volume up on
  the days that matter least for information content. His own back-test on 1941–1975
  NYSE data claimed that whenever NVI sits above its 1-year moving average, the odds of
  being in a bull market are **96% reliable**; below the average carries much weaker
  information (Fosback himself treated it mainly as a bull-market confirming signal,
  not a bear-market timing signal).
- **The popularization.** The rule is reproduced almost verbatim across retail
  technical-analysis references — StockCharts.com's ChartSchool entry on
  "Negative Volume Index (NVI)", Investopedia's NVI page, and MetaStock/TC2000 built-in
  indicator documentation — nearly always quoting the 96% figure without re-testing it
  on a longer or different sample.
- **The companion indicator.** Fosback also defined the mirror-image **Positive Volume
  Index (PVI)** (cumulate returns on volume-UP days), which he treated as a much weaker,
  more ambiguous signal — out of scope here; this study tests the headline NVI claim
  only, as specified.

## What we measure, and the honesty rails

- **Two tests of the SAME claim, at two levels of statistical power.** (1) Fosback's
  own annual bull/bear framing, replicated on 74 complete calendar years of ^GSPC
  (1952–2025) — the historically faithful test, benchmarked against the
  **unconditional base rate** the original write-ups never report, with a 20,000-draw
  label-shuffle placebo. (2) A higher-power daily cross-check (21/63/252-day forward
  returns) with a Newey-West HAC *t* (lag = horizon) — because daily forward returns
  overlap by construction, a **naive Welch *t* on overlapping windows is reported
  alongside the HAC number specifically to name the overlapping-return trap**, not to
  claim it.
- **Execution.** One documented lag throughout: NVI/EMA state is fully known at the
  close of day *t* (that day's volume has printed); every forward-return window and
  every timer position uses that state to predict or hold from the close of *t+1*
  onward — a single shift, applied once, never doubled.
- **Volume source, named as a proxy.** Fosback's 1976 test used the NYSE's official
  composite tape. No free, long-history composite-volume feed exists today; the
  standard modern implementation (StockCharts, MetaStock, and this study, as the brief
  specifies) computes NVI on a security's **own** reported volume — Yahoo's ^GSPC
  index-level vendor tape for the 1950-onward replication, SPY's own consolidated tape
  for the tradable third axis. Both are named proxies for the tape Fosback actually
  used, not a silent substitution.
- **Costs, one-way × NAV per leg** (0/5/10 bps swept) on the timer; long-only, no
  borrow (the rule never shorts).

## Data sources

- **^GSPC daily OHLC + Volume** and **SPY daily total-return OHLC + Volume** —
  yfinance (no key), cached under `_cache/` (`nvi_gspc.csv`, `nvi_spy.csv`),
  1950-01-03 → 2026-06-30 (^GSPC; the first session Yahoo! reports non-zero S&P 500
  volume) and 1993-01-29 → 2026-06-30 (SPY, ETF inception).
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related academic literature on volume and returns

- Karpoff, J.M. (1987). *The Relation Between Price Changes and Trading Volume: A
  Survey*. Journal of Financial and Quantitative Analysis. The foundational survey on
  what volume actually correlates with (volatility, dispersion of beliefs) — a useful
  contrast to Fosback's specific "low volume = informed" causal story, which was never
  itself subjected to peer-reviewed testing that we could find.
  Fosback's book is the primary, and largely sole, source for the 96% figure.
- Gervais, S., Kaniel, R. & Mingelgrin, D.H. (2001). *The High-Volume Return Premium*.
  Journal of Finance. The **opposite-direction, peer-reviewed** volume-return claim —
  unusually **high** (not low) volume precedes higher returns, an attention/visibility
  story rather than a "quiet accumulation" story. Directly relevant contrast; see the
  dedup map below (sibling study 512).

## Related desk studies (the dedup map — what this study is NOT)

- [492-up-down-volume](../492-up-down-volume/) — a **breadth** measure (the
  cross-market up-volume vs down-volume share, a proxy for NYSE advance/decline
  volume) used as a **selling-climax contrarian entry signal**. This study's NVI is a
  **single-instrument** cumulative indicator built from one security's own volume —
  no cross-sectional breadth, no climax/contrarian framing, and a **trend-confirmation**
  claim (bull market ongoing) rather than a reversal-timing claim.
- [109-obv-divergence](../109-obv-divergence/) — Granville's On-Balance Volume, which
  cumulates volume itself (signed by price direction) into a running total and reads
  its **trend or its divergence from price** as the signal. NVI instead cumulates
  **price return**, gated by whether volume rose or fell — a different construction
  (cumulate returns vs. cumulate volume) testing a different mechanism (quiet-day
  smart money vs. volume-leads-price divergence).
- [511-volume-momentum](../511-volume-momentum/) — Lee & Swaminathan's cross-sectional
  **double sort**: does trailing dollar volume condition the *cross-sectional momentum*
  premium (high-volume winners vs. low-volume losers, and the speed of reversal)? A
  panel-level conditioning study on a 40-name basket, not a single-instrument
  regime-timing indicator.
- [116-power-hour](../116-power-hour/) — an **intraday** continuation/reversal claim
  about the last trading hour of the session. Different frequency (intraday bars vs.
  daily), different mechanism (institutional close-of-day flow vs. NVI's
  quiet-day-accumulation story) entirely.
- [512-high-volume-return-premium](../512-high-volume-return-premium/) — the
  **opposite-signed** academic claim (Gervais-Kaniel-Mingelgrin): **high**, not low,
  abnormal volume precedes higher forward returns, tested as a cross-sectional
  long-short. Worth reading alongside this study — between the two, this desk has now
  tested "low volume is bullish" (Fosback, here) and "high volume is bullish" (GKM,
  512) on modern US tapes, and **neither** direction of the volume-precedes-price story
  survives HAC-robust inference.

None of the siblings test **Fosback's specific NVI construction** (cumulate return,
gated on falling volume, versus its 1-year EMA) — that headline replication, and its
base-rate teardown, is this study's own axis.
