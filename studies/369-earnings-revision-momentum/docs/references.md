# References & literature map — Study 369 (Earnings-Revision-Momentum)

## The claim under test

- **The revision factor.** The believers' pitch: stocks whose forward earnings estimates are
  being revised **up** by sell-side analysts continue to outperform, because consensus
  estimates adjust *slowly* to news. Givoly & Lakonishok (1979), *The information content of
  financial analysts' forecasts of earnings* (Journal of Accounting & Economics), and Stickel
  (1991), *Common stock returns surrounding earnings forecast revisions* (The Accounting
  Review), are the early evidence; the strategy is sold as the "earnings-revision" or
  "estimate-momentum" factor across the practitioner and quant literature.
- **Post-Earnings-Announcement Drift (PEAD) / SUE.** The closely-related anomaly: prices drift
  in the direction of an earnings *surprise* for weeks after the release. Ball & Brown (1968),
  *An empirical evaluation of accounting income numbers* (JAR); Bernard & Thomas (1989),
  *Post-earnings-announcement drift: delayed price response or risk premium?* (JAR); Foster,
  Olsen & Shevlin (1984) on standardized unexpected earnings (SUE). Our **realized surprise**
  proxy is the SUE half of the signal; the **q/q estimate change** is the revision half.
- **The folklore hook.** "Buy what the analysts are upgrading" is repeated as a near-free factor
  — a tilt with a long academic pedigree that *sounds* obviously profitable. The factor-zoo
  question is whether it survives **out-of-sample, after costs, on data you can actually get**.

## Why true analyst-revision data is not free — and what we do instead

- **IBES / consensus-estimate feeds.** The canonical inputs (I/B/E/S or FactSet consensus
  estimate *levels and revisions* at daily frequency) are paid, licensed datasets, **not**
  available through free yfinance. yfinance *does* expose `get_earnings_dates`: per-quarter
  **consensus EPS estimate**, **reported EPS**, and **surprise %** back to ~2001 for long-listed
  large-caps. We therefore build a transparent **proxy** for "estimates being revised up": a
  cross-sectional z-score of the realized surprise plus the q/q change in the consensus estimate.
  This is a *noisier, lower-frequency, lagged* stand-in for a live revision feed — and we say so
  on the Signal axis. Every input is a public field; nothing is fabricated.
- **Survivorship.** A fixed basket of 40 surviving large-caps excludes firms whose estimates
  collapsed into delisting. That tilts the long leg mildly *up* and removes the worst of the
  short-leg tail — a bias that can only **flatter** the spread we report, so it does not threaten
  a WEAK/MIRAGE verdict (named explicitly on the Signal axis per house rules).

## Why the spread must clear inference *and* costs — the statistics

- **Cross-sectional long-short inference.** We form a per-quarter top-minus-bottom tercile spread
  and test its mean against zero with a one-sample **t** (quarters as approximately independent
  draws), plus a **label-shuffle placebo**: permute the revision score within each quarter,
  rebuild the book, and ask how often a shuffled book matches the real one (Fisher's
  randomization logic; Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993). The
  placebo is the honest null for a ranked long-short book.
- **Costs are the binding constraint, not significance.** Novy-Marx & Velikov (2016), *A
  taxonomy of anomalies and their trading costs* (Review of Financial Studies), show many
  published long-short anomalies — especially high-turnover ones rebalanced every quarter — do
  not survive realistic transaction costs. A quarterly-rebalanced revision book pays four
  crossings per name per quarter plus short borrow; the gross spread here is the same order of
  magnitude as those frictions.
- **Multiple testing / the factor zoo.** Harvey, Liu & Zhu (2016), *…and the Cross-Section of
  Expected Returns* (RFS), and McLean & Pontiff (2016), *Does academic research destroy stock
  return predictability?* (Journal of Finance), formalise why a famous, much-published factor
  needs a *higher* bar than a naive t-stat and why edges shrink post-publication — directly
  relevant to a four-decade-old "buy the upgrades" claim.

## Method lineage (the desk's shared engine)

- **Revision score + tercile book.**
  [`strategy.build_rev_score`](../earnings_revision_momentum/strategy.py) and
  [`strategy.quantile_spread`](../earnings_revision_momentum/strategy.py) — cross-sectional
  z-score and the long-top/short-bottom spread, excess of SPY.
- **One-day lag + forward excess returns.**
  [`strategy.event_excess_returns`](../earnings_revision_momentum/strategy.py) enters the close
  one day after the earnings date (no look-ahead) and measures each leg vs SPY.
- **Welch t + label-shuffle placebo.**
  [`strategy.welch_t`](../earnings_revision_momentum/strategy.py) and
  [`strategy.placebo_pvalue`](../earnings_revision_momentum/strategy.py) — the Signal-axis tests.
- **Costs + borrow.** [`strategy.net_of_costs`](../earnings_revision_momentum/strategy.py) —
  one-way costs × turnover plus a short-leg borrow charge.
- **Deterministic synthetic control.**
  [`data.synthetic_panel`](../earnings_revision_momentum/data.py) plants a known
  `edge · rev_score` link; the control confirms the engine is unbiased (edge 0 ⇒ no spread) and
  powered (a modest edge ⇒ t ≫ 2). The offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted closes (SPY + a fixed 40-name large-cap basket, 1995→2026) and
  `get_earnings_dates` per-quarter EPS estimate / reported / surprise (39 names, 2001→2026),
  cached under `_cache/`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- The desk's broader **factor-zoo** teardowns (momentum, value, quality and friends) — the
  recurring lesson is the same: a gross in-sample tilt that thins to nothing once you demand a
  robust *t*, an out-of-sample window, and realistic execution costs. Earnings-revision momentum
  is a particularly clean case because the gross whisper is *exactly* the size of the frictions.
