# References & literature map — Study 533 (Analyst-Dispersion)

## The claim under test

- **The puzzle.** Karl Diether, Christopher Malloy & Anna Scherbina, *Differences of Opinion
  and the Cross-Section of Stock Returns* (2002, Journal of Finance 57(5)): stocks with **high
  dispersion in analysts' earnings forecasts** subsequently earn **lower** returns. A portfolio
  long low-dispersion and short high-dispersion stocks earned a sizeable monthly premium in
  their 1983–2000 sample. This is a *puzzle* because the naive risk intuition runs the other
  way — more disagreement ought to mean more uncertainty and therefore a *higher* required
  return; DMS find the opposite.
- **The mechanism (Miller 1977).** Edward Miller, *Risk, Uncertainty, and Divergence of
  Opinion* (1977, Journal of Finance): when **short-sale constraints** bind, prices reflect the
  views of the *optimists* (pessimists cannot fully express their view by shorting). High
  dispersion then means high over-pricing, which corrects as the disagreement resolves — so
  high-dispersion stocks under-perform. DMS is the cross-sectional confirmation of Miller's
  over-pricing hypothesis.

## Dispersion proxy — what we measure, and its honest limits

- **Academic measure.** DMS use the **standard deviation of analyst EPS forecasts scaled by
  the absolute mean forecast** (from I/B/E/S), a monthly cross-sectional panel.
- **Our free proxy.** yfinance `Ticker.earnings_estimate` exposes, per name, the current
  consensus EPS estimate for the current fiscal year (`0y`) with the analyst **low / mean /
  high**. We form `dispersion = (high − low) / |mean|` — the *range*-based analogue of the
  DMS spread, scaled the same way. Range and standard deviation are monotonically related
  across names with similar analyst counts, so the rank sort is faithful to the DMS ordering.
- **The binding limitation.** yfinance gives only a **current snapshot** — there is **no
  historical dispersion series**. DMS's result is a *forward* monthly panel; we can only sort
  today's dispersion against the *trailing* return realised into the snapshot. This is a
  contemporaneous association, not a tradable forward strategy, and it rests on a single
  cross-section of ~40 survivor names. A true replication needs vendor dispersion history
  (I/B/E/S / Refinitiv). We name this on the Signal axis and do not over-claim.

## Replication, decay, and the survivorship caveat

- **Subsequent literature.** Avramov, Chordia, Jostova & Philipov (2009, *Dispersion in
  Analysts' Earnings Estimates and Credit Rating*, JFE) tie the dispersion effect to **credit
  risk** — it concentrates in low-rated, distressed names. Sadka & Scherbina (2007,
  *Analyst Disagreement, Mispricing, and Liquidity*, JF) link it to **liquidity** and limits
  to arbitrage. Both imply the effect is weak-to-absent among liquid large-caps — exactly our
  universe — so a thin or wrong-sign large-cap reading is consistent with the refined picture.
- **Anomaly decay.** McLean & Pontiff (2016, *Does Academic Research Destroy Stock Return
  Predictability?*, JF) document that published anomalies shrink ~58% post-publication; the
  dispersion effect, published in 2002, is a prime candidate for decay. Harvey, Liu & Zhu
  (2016, *…and the Cross-Section of Expected Returns*, RFS) caution that a single significant
  cross-section is not evidence against the multiple-testing null.
- **Survivorship.** Our basket is names still trading in 2026. The surviving high-dispersion
  names are the multi-baggers (NVDA, TSLA, ORCL, META); their failed peers are gone. This
  biases the high-dispersion (short) leg's realised return *upward*, working **against** the
  DMS prediction at long horizons — the mechanism behind our wrong-sign 12-month result.

## Why a lone significant cross-section is not enough

- **One-sample t** of the pooled low-minus-high long-short sample against zero
  ([`strategy.ttest_vs_zero`](../analyst_dispersion/strategy.py)).
- **Label-shuffle placebo** ([`strategy.placebo_pvalue`](../analyst_dispersion/strategy.py)) —
  20,000 random re-sorts of the same returns; Fisher's randomization logic (Efron & Tibshirani,
  *An Introduction to the Bootstrap*, 1993). On a 40-name single cross-section even a real
  effect rarely clears t ≥ 2 robustly — and a lone window's t > 3 that reverses at every other
  horizon is the canonical **WEAK** (not REAL) call.
- **Deterministic synthetic control** ([`data.synthetic_panel`](../analyst_dispersion/data.py))
  plants a known dispersion→return drag; seed-robust over 25 seeds it recovers the planted edge
  (avg t ≈ 2.63, 100% of seeds clear t > 2) and returns ~0 under the null — proving the
  cross-sectional sort engine is faithful and adequately powered.

## Data sources used here

- **yfinance** `Ticker.earnings_estimate` (0y EPS low/mean/high → dispersion) and daily
  adjusted closes for a fixed 40-name large-cap basket, snapshot pulled 2026-06-26, cached under
  `_cache/disp_snapshot.csv` and `_cache/disp_prices.parquet`. All headline numbers are pinned
  in [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [538/239/240-style fundamental sorts](../532-firm-age-anomaly/) — the firm-age anomaly is the
  nearest neighbour in method (a survivor-basket cross-sectional sort whose sign is flipped by
  survivorship). [`363-pead-drift`](../363-pead-drift/) is the sibling event study on the same
  yfinance earnings plumbing.
