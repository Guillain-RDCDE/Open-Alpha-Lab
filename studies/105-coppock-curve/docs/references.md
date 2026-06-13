# References & literature map — Study 105 (Coppock-Curve)

## The claim under test

- **The original rule.** Edwin Sedgwick Coppock, *"E.S.C.'s Titanic Indicator"*,
  Barron's, October 15, 1962. Coppock — an economist and fund manager — designed the
  indicator at the request of the Episcopal Church to identify "good buying opportunities"
  in equities. His derivation was based on grief counselling research: he noted that
  human beings take 11–14 months to recover emotionally from a major loss, and mapped
  this to market psychology. The indicator is the WMA(10) of (ROC(14) + ROC(11)) on
  monthly closes. The buy rule fires when the curve turns up from below zero — signalling
  that the pessimism "trough" has passed. No sell rule was originally specified.
  The anecdotal design methodology is explicitly acknowledged in the study's write-up.

- **Steelmanned claim.** *The Coppock buy signal, applied to the S&P 500 on monthly bars,
  delivers higher forward 12-month returns than random-timing entries at the same frequency
  and exceeds the buy-and-hold average return.* This is the sharpest empirically testable
  form of the original assertion.

## Why the claim might have merit — momentum and mean reversion

- **Long-term momentum (12-month).** Jegadeesh & Titman (1993), *"Returns to Buying
  Winners and Selling Losers: Implications for Stock Market Efficiency"*, Journal of
  Finance — documented cross-sectional 12-month momentum in stocks; the Coppock ROC
  windows (11 and 14 months) fall squarely in this momentum horizon. At the index level,
  monthly ROC over this window carries some autocorrelation.
- **Reversal after extreme drawdowns.** DeBondt & Thaler (1985), *"Does the Stock Market
  Overreact?"*, Journal of Finance — documented long-horizon mean reversion after extreme
  multi-year losses. The Coppock signal fires after bear markets that have induced
  multi-month ROC extremes, putting it in the overreaction-reversal camp.
- **Presidential / business cycle timing.** Yale Hirsch and the *Stock Trader's Almanac*
  line of research identifies recurring 4-year market cycles linked to the US presidential
  cycle. Coppock signals cluster at cycle troughs, which may partially explain their
  timing power as a business-cycle proxy.

## The critical confounds — why the verdict is WEAK

- **Market timing = beta, not alpha.** The Coppock signal fires near bear-market troughs.
  A long-only investor who simply held through the trough already participates in the
  subsequent recovery. The outperformance vs buy-and-hold is largely explained by the
  concentrated timing beta at depressed prices rather than genuine alpha. Brinson, Hood
  & Beebower (1986), *"Determinants of Portfolio Performance"*, Financial Analysts Journal,
  show that timing accounts for a small fraction of return variation versus asset allocation.
- **Very small n.** 19 signals over 76 years (1950–2026). Any single outlier dominates the
  statistics. The December 2001 signal lost 26.6% over the next 12 months — one signal
  would shift the mean by ~0.8 ppt. Harvey et al. (2016), *"…and the Cross-Section of
  Expected Returns"*, Review of Financial Studies, document the multiple-testing problem in
  factor research; with n=19 the bar for "real" is effectively higher than |t|=2.
- **Lagging by design.** The 23-month warm-up plus the "trough turn-up" criterion mean
  the signal fires *after* the worst is over. In 2009 it fired in May (2 months after the
  March trough); in 1975 it fired 12 months after the trough. The early recovery months
  are the most profitable — Coppock misses them by construction.
- **Grief-counselling derivation.** The period choices (11 and 14 months) were not
  calibrated by statistical optimisation but by an analogy to human bereavement timelines.
  That makes the parameter choices arbitrary (within the momentum-horizon range that
  happens to be effective), lending some concern about data-mining even in the original
  paper.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *"A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix"*, Econometrica —
  [`strategy.summarize`](../coppock_curve/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Block bootstrap CI.** Politis & Romano (1994), *"The Stationary Bootstrap"*, JASA —
  available via [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).
- **Linearly weighted moving average.** The WMA(10) formulation follows Coppock's original
  specification; implementation in [`strategy.wma`](../coppock_curve/strategy.py).
- **Reproducibility stamp.** Content fingerprints in [`data.fingerprint`](../coppock_curve/data.py).

## Data sources

- **Yahoo! Finance monthly bars** (via `yfinance`), daily bars resampled to month-end.
  ^GSPC (S&P 500) from 1950-01-31; SPY from 1993-01-31. Long monthly history is available
  (unlike intraday bars) giving 70+ years of data — still only 19 Coppock signals.
  Every headline is pinned with an as-of date and content fingerprint; see
  [`docs/results.md`](results.md).

## Related desk studies

- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: the 50/200-day "golden cross" on daily
  bars — the same long-term momentum-filter family, same lagging characteristic.
- **[Study 16 — Storm-Shy](../../16-storm-shy/)** and
  **[Study 68 — All-Weather](../../68-all-weather/)**: market timing / allocation signals
  that also trade on macro-regime detection, with similar n-is-small problems.
- **[Study 86 — Tail-Radar](../../86-tail-radar/)**: volatility-index-based regime signals,
  another "when to hold vs sidestep" family tested on monthly bars.
- **[Study 73 — First-Light](../../73-first-light/)**: intraday momentum in the first
  trading hour — a different (higher-frequency) slice of the momentum literature.
