# References & literature map — Study 168 (Advance-Decline)

## The claim under test

**The folk recipe.** "When the generals are advancing but the soldiers are retreating" —
market folklore for when the index (e.g. the S&P 500) makes a new high while the cumulative
Advance-Decline (A/D) line, which tracks how many stocks are going up vs down, fails to confirm
the new high. Technical analysts cite this as a bearish divergence signalling a fragile,
narrow rally where only a few large-cap stocks are doing the work — and thus predicting an
imminent market top or correction. The A/D line itself is a century-old technical indicator.

## The history of the A/D line

- **Breadth as a market-health measure.** McClellan, S. & McClellan, T. (1998), *Patterns for
  Profit: The McClellan Oscillator and Summation Index* — the canonical practitioner reference
  for A/D-based indicators including the McClellan Oscillator (momentum of the A/D line) and
  Summation Index. The McClellan Oscillator remains widely cited in technical-analysis circles.

- **Early academic assessment.** Colby, R.W. & Meyers, T.A. (1988), *The Encyclopedia of
  Technical Market Indicators* — catalogs the A/D line and its variants (cumulative, net, NYSE
  vs S&P) alongside their backtested performance; the results are weak and pre-cost.

- **The "generals and soldiers" narrative.** Fosback, N. (1976), *Stock Market Logic* —
  an early popularisation of the idea that a market supported by broad participation
  is healthier than one driven by a narrow cadre of "generals." This is perhaps the most
  cited book-length treatment of the breadth-divergence idea.

## Why breadth MIGHT predict — the steelman

- **Liquidity/participation theory.** If buying pressure is genuinely broad, more capital
  is entering the market; a narrowing rally (a few stocks lifting the cap-weighted index
  while most names lag) may indicate exhaustion of the bull move and vulnerability to
  reversal when the leaders also stall.

- **Small-cap leading.** Lo, A.W. & MacKinlay, A.C. (1990), *When are Contrarian Profits
  Due to Stock Market Overreaction?* (Review of Financial Studies) — documents return
  cross-autocorrelations where small stocks lead large stocks at weekly horizons. If A/D
  captures the broad small-stock universe, it could theoretically lead the cap-weighted index.

- **Breadth and future returns.** Zweig, M. (1990), *Winning on Wall Street* — claims that
  certain breadth thrust signals (very high advance-decline ratios) predict strong forward
  returns. Note this is a *bullish* breadth claim (broad participation = bullish), not the
  bearish divergence claim this study tests.

## Why breadth probably does NOT reliably predict — the counter-literature

- **Technical indicators and data-snooping.** Sullivan, R., Timmermann, A., & White, H.
  (1999), *Data-Snooping, Technical Trading Rule Performance, and the Bootstrap* (Journal
  of Finance) — apply White's Reality Check to 7,846 technical trading rules; most fail
  to survive the multiple-comparisons correction. Breadth indicators are in this zoo.

- **Market efficiency and predictability.** Fama, E.F. (1991), *Efficient Capital Markets:
  II* (Journal of Finance) — reviews the evidence on short-run return predictability;
  the consensus is that simple technical filters including moving averages and breadth
  indicators add little after costs and data-snooping adjustments.

- **Coincident vs leading.** The core problem with A/D divergence: breadth and the
  cap-weighted index are *largely coincident*, not lead-lag. When they diverge it is
  often a composition artefact (a few mega-caps dominating the index weight) that
  resolves by the mega-caps catching down or breadth catching up — and the direction is
  not predictable in advance. This study finds no reliable directional signal.

- **Multiple-comparisons / specification search.** The lookback sweep (21d, 42d, 63d,
  126d, 252d) shows that the sign of the divergence-vs-baseline difference *flips*
  across window choices, a classic marker of a spurious specification choice. Bonferroni
  on 5 lookbacks × 3 horizons = 15 comparisons would require |t| > 3.3 for 5% FWE,
  which no single specification clears.

- **Survivorship in the panel.** The S&P 500 constituent panel used here is current
  membership projected backwards; delisted names (bankruptcies, mergers, deletions for
  poor performance) are absent. For a breadth study this is non-trivial: firms that were
  ultimately removed from the index declined and likely contributed more decliners during
  their tenure — meaning the true historical A/D line (including losers) was probably
  weaker than what we observe, and our divergence signal frequency / direction may differ.

## Method lineage

- **Newey-West HAC t-stat.** Newey, W. & West, K. (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*
  (Econometrica) — used in `strategy.summarize` to correct for the serial correlation
  in overlapping 21-day forward returns.

- **Permutation test.** The permutation baseline (drawing random sub-samples of the
  same size as the divergence count) provides a mild multiple-comparisons check:
  it asks whether the divergence sub-sample mean is extreme relative to what any
  randomly chosen sub-sample of the same size would produce.

- **Survivorship guard.** The `allow_survivorship_bias=True` opt-in pattern mirrors
  `quantlab.hf_data.SurvivorshipBiasError` and `quantlab.universe.sp500_symbols` —
  the desk-wide discipline that forces explicit acknowledgment of panel conditioning.

## Related desk studies

- **[Study 50 — High-Water](../../50-high-water/)**: 52-week-high nearness vs momentum —
  another breadth / price-level signal family; cross-sectional rather than aggregate.
- **[Study 24 — Stampede](../../24-stampede/)**: breadth thrusts (very high advance counts)
  tested as a *bullish* signal — the flip side of the bearish divergence claim here.
- **[Study 80 — Cold-Open](../../80-cold-open/)**: seasonal / calendar patterns tested
  with the same "does the signal survive an unconditional baseline?" discipline.
- **[Study 76 — Rice-Paper](../../76-rice-paper/)**: the Bonferroni / multiple-comparisons
  teardown methodology applied to a similar specification-search problem.
