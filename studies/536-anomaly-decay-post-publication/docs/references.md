# References & literature map — Study 536 (Anomaly-Decay-Post-Publication)

## The core paper under test — does publication erode anomalies?

- **McLean & Pontiff (2016),** *Does Academic Research Destroy Stock Return Predictability?*
  (Journal of Finance, 71(1), 5-32). The study this teardown rebuilds. They replicate **97
  characteristics** shown to predict the cross-section of returns and find the long-short
  return is on average **~26% lower out-of-sample** and **~58% lower post-publication** than
  in the original sample — i.e. publication roughly *halves* the premium. Their interpretation:
  publicising an anomaly invites arbitrage that erodes it (post-publication trading volume and
  correlation among anomaly stocks rise). The ~0.5 post/pre ratio is the benchmark we hold the
  real tape against.
- **Schwert (2003),** *Anomalies and Market Efficiency* (Handbook of the Economics of Finance) —
  the earlier observation that several documented anomalies (size, value, weekend) weakened or
  disappeared after they were published, plausibly because practitioners traded them away.
- **Chordia, Subrahmanyam & Tong (2014),** *Have Capital Market Anomalies Attenuated in the
  Recent Era of High Liquidity and Trading Activity?* (Journal of Accounting and Economics) —
  finds anomaly profits attenuate with rising liquidity and arbitrage capital, complementary
  to the publication channel.

## The classic anomalies in the battery (and their publication dates)

- **12-1 momentum.** Jegadeesh & Titman (1993), *Returns to Buying Winners and Selling Losers*
  (Journal of Finance) — the 3-12 month relative-strength effect. Our split year: **1993**.
- **Low volatility / low beta.** Ang, Hodrick, Xing & Zhang (2006), *The Cross-Section of
  Volatility and Expected Returns* (Journal of Finance) — high idiosyncratic vol predicts low
  returns; later framed as the low-volatility anomaly (Baker, Bradley & Wurgler, 2011). Split
  year: **2006**.
- **Short-term (1-month) reversal.** Jegadeesh (1990), *Evidence of Predictable Behavior of
  Security Returns* (Journal of Finance); Lehmann (1990), *Fads, Martingales, and Market
  Efficiency* (QJE) — last month's losers beat last month's winners. Split year: **1990**.
- **Long-term (3-year) reversal.** De Bondt & Thaler (1985), *Does the Stock Market
  Overreact?* (Journal of Finance) — 3-5 year losers outperform 3-5 year winners. Split year:
  **1985**.

## The method lineage (the desk's shared machinery)

- **Cross-sectional tercile long-short.** Fama & French (1992, 1993) sort methodology, here as
  a dollar-neutral top-minus-bottom tercile rebuilt each month
  ([`strategy.long_short_returns`](../anomaly_decay/strategy.py)).
- **One execution lag.** The signal formed at the close of month *t* trades month *t+1*'s
  return — applied once in [`strategy.long_short_returns`](../anomaly_decay/strategy.py) via a
  forward-return shift; the synthetic generator mirrors the same lag so the planted edge is
  recoverable.
- **Label-shuffle placebo.** A permutation null on the pre/post boundary
  ([`strategy.placebo_split`](../anomaly_decay/strategy.py)): randomly re-partitioning the
  months into same-size pre/post groups destroys the publication boundary, so a real
  publication-driven decay should be rarer than the shuffled ones.
- **Synthetic positive control.** A deterministic planted-decay panel
  ([`strategy.synthetic_control`](../anomaly_decay/strategy.py), 20-seed averaged) proves the
  split engine recovers a known pre/post step and that a fully-decayed post leg cannot fake
  significance — a faithfulness check only, never a real-tape claim.

## Data sources used here

- **Yahoo! Finance** (via `yfinance`), auto-adjusted daily closes for a fixed 40-name
  large-cap **survivor** basket, resampled to month-end total returns, 1980–2026. Cached to
  this study's own `_cache/` parquet. Headline numbers are pinned with an as-of date (2026-05,
  last full month) and a content fingerprint (see [`docs/results.md`](results.md)). The offline
  reproducible core runs on the deterministic
  [`data.synthetic_panel`](../anomaly_decay/data.py) generator, never the network.
- **Data limitation.** The 3-year-reversal split has only ~24 pre-publication months (the
  basket history starts ~1980, the paper is 1985), so its decay estimate is noisy — flagged in
  results, not hidden.

## Related desk studies

- **[Study 345 — Survivorship-Bias](../../345-survivorship-bias/)**: the survivor-basket
  distortion this study explicitly leans into and names; here it inflates the *pre*-publication
  half most.
- **[Study 346 — Multiple-Testing](../../346-multiple-testing/)**: the FWER/FDR correction
  bake-off across a family of named effects — the sister methodology demo. 536 is the
  *out-of-sample-decay* angle rather than the *in-sample family-wise* angle.
- **[Study 363 — PEAD-Drift](../../363-pead-drift/)**: the same tercile long-short / one-sample-*t*
  / label-shuffle machinery on a real event anomaly, with the same survivor-basket caveat.
