# References & literature map — Study 398 (Entropy-Efficiency)

## The claim under test

- **The "efficiency clock" framing.** A recurring idea in econophysics and the
  technical-analysis fringe: the market has a *measurable randomness*, and when that randomness
  **falls** — returns become more *ordered*, more *predictable*, lower-entropy — a tradable
  window opens. The believer's pitch is that low entropy ⇒ forecastable ⇒ profitable: trade
  when the tape is "structured", stand aside when it is "noise". Entropy is sold as an
  early-warning / regime gauge that beats raw volatility.
- **The two complexity measures.** The clock is usually built from one of two estimators:
  - **Permutation entropy** — Bandt & Pompe (2002), *Permutation Entropy: A Natural Complexity
    Measure for Time Series*, Phys. Rev. Lett. 88(17): the Shannon entropy of the frequencies
    of the ordinal patterns (the argsort shapes) of short sub-windows, normalised to [0, 1].
    Fast, robust to monotone transforms, near-1 for white noise.
  - **Sample entropy** — Richman & Moorman (2000), *Physiological time-series analysis using
    approximate entropy and sample entropy*, Am. J. Physiol. 278: the negative log of the
    conditional probability that two sub-sequences alike for *m* points remain alike for *m+1*;
    low for self-similar/forecastable series. (Predecessor: Pincus 1991, *Approximate
    entropy*.)

## The financial-entropy literature

- **Entropy & the efficient-market hypothesis.** A body of econophysics work measures the
  "informational efficiency" of markets via entropy and finds it **high and time-varying but
  near-maximal** for liquid indices: Risso (2008), *The informational efficiency and the
  financial crashes*, Research in International Business and Finance; Zunino et al. (2009),
  *Forbidden patterns, permutation entropy and stock market inefficiency*, Physica A — using
  permutation entropy precisely to *rank* markets, finding developed-market indices sit close
  to the random-walk ceiling. Ortiz-Cruz et al. (2012) apply multiscale entropy to crude oil.
- **The decisive caveat.** These papers measure *predictability of structure*, not *expected
  return*. A lower-entropy stretch can be more forecastable in **shape** (autocorrelation,
  ordinal pattern) while carrying **no mean-return advantage** — the gap our study targets.
- **Hurst / long memory cousins.** The same "is the tape more orderly now?" question is asked
  by the Hurst exponent and rescaled-range analysis (Mandelbrot & Wallis, 1969); see the
  desk's **[Study 397 — Hurst Regime](../397-hurst-regime/)** for the persistence-based version
  of this regime question. Entropy and Hurst are two lenses on the same object.

## Why the inference must be conservative — the statistics

- **Overlapping windows + a smooth, autocorrelated signal.** The entropy clock is a 60-day
  rolling statistic — adjacent values share 59 days — and low-entropy days arrive in long
  *clusters*, while forward returns over 5/21 days overlap heavily. A naive i.i.d. t-stat over
  such data is wildly optimistic. We use a **Newey-West / HAC** standard error (Newey & West,
  1987, *A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent
  Covariance Matrix*, Econometrica) and a **stationary block bootstrap** (Politis & Romano,
  1994, *The Stationary Bootstrap*, JASA) that resamples regime membership in blocks, so the
  null preserves the clustering instead of pretending the days are independent.
- **Base rates and the predictability/profitability gap.** US equities rise most days/months
  unconditionally, so a high post-signal win-rate is expected under the null; the right
  comparison is the **excess** of low- over high-entropy forward returns, judged against a
  clustered null — not the raw win-rate (the base-rate fallacy, Kahneman & Tversky, 1973).
- **Selection on a flexible knob.** Entropy clocks have many free choices (estimator, window,
  embedding *m*, quantile threshold). Rules tuned on the same tape they are tested on inflate
  significance; Harvey, Liu & Zhu (2016), *…and the Cross-Section of Expected Returns* (RFS),
  and Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*, motivate the higher bar and
  the quantile-sweep robustness check.

## Method lineage (the desk's shared engine)

- **Entropy estimators.** [`strategy.rolling_entropy`](../entropy_efficiency/strategy.py) wraps
  causal permutation entropy (`_perm_entropy_window`, Bandt-Pompe) and sample entropy
  (`_sample_entropy_window`, Richman-Moorman); the window ends at *t-1* so no future leaks.
- **HAC t + block-bootstrap null.** [`strategy.low_minus_high_t`](../entropy_efficiency/strategy.py)
  (Newey-West) and [`strategy.block_bootstrap_p`](../entropy_efficiency/strategy.py)
  (stationary block bootstrap of the regime labels) — the Signal-axis tests.
- **Deterministic synthetic control.**
  [`data.synthetic_returns`](../entropy_efficiency/data.py) toggles a random (high-entropy)
  regime against a zero-sum cyclic **structured** (low-entropy) regime with a *planted* forward
  edge; with the edge at zero the control must NOT manufacture significance, with a large edge
  it must. The offline core runs with no network.
- **Forward returns with execution lag + costs.**
  [`strategy.regime_forward`](../entropy_efficiency/strategy.py) enters one day after the signal
  (no look-ahead); [`strategy.net_of_costs`](../entropy_efficiency/strategy.py) charges one-way
  costs × turnover on the long-in-low-entropy rule.

## Data sources used here

- **yfinance** daily adjusted closes for SPY, 1995-01-04 → 2026-06-18, cached under
  `_cache/spy_prices.csv`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 397 — Hurst Regime](../397-hurst-regime/)**: the long-memory / persistence lens on
  the same "is the market more orderly now, and does it pay?" question. Entropy and the Hurst
  exponent are complementary measures of the same regime object; reading them together shows
  the gauge changes but the verdict — predictability ≠ profitability — does not.
