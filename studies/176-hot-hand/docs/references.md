# References & literature map — Study 176 (Hot-Hand)

## The two folk claims under test

- **The Hot-Hand.** Popularised in sports psychology — *"a player who has made several
  shots in a row is more likely to make the next one."* Gilovich, Vallone & Tversky (1985),
  *The Hot Hand in Basketball: On the Misperception of Random Sequences* (Cognitive Psychology),
  coined the term and — controversially — argued the hot hand in basketball was an illusion.
  Miller & Sanjurjo (2018), *Surprised by the Hot Hand Fallacy? A Truth in the Law of Small
  Numbers* (Econometrica), revisited the work with a subtle selection-bias correction and
  found the hot hand is statistically real in basketball after all. On daily equity returns the
  question is symmetric: after N consecutive up-days, does the market "keep rolling"?

- **The Gambler's Fallacy.** The mistaken belief that independent coin flips "correct" for
  past outcomes: *"after five heads, a tail is overdue."* Kahneman & Tversky (1974),
  *Judgment Under Uncertainty: Heuristics and Biases* (Science), document this as a
  near-universal cognitive bias — we expect short random sequences to be balanced. Applied
  to markets: after N consecutive down-days, *"a bounce is overdue."* Ironically, the
  gambler's fallacy turns out to be *correct* for down-streaks on the S&P 500 — but via
  a real mechanism (panic→mean reversion), not via the random-walk logic the bias assumes.

## Academic literature on serial dependence in equity returns

- **Lo & MacKinlay (1988).** *Stock Market Prices Do Not Follow Random Walks: Evidence from
  a Simple Specification Test* (Review of Financial Studies). Documents significant positive
  autocorrelation in weekly returns for small-cap stocks and the index — the original
  empirical challenge to the random walk. The effect is small and does not cleanly map to
  single-day streaks.

- **Conrad & Kaul (1988).** *Time-Variation in Expected Returns* (Journal of Business). Find
  evidence of positive autocorrelation in daily and weekly returns at short horizons, but
  emphasise it is too small to be exploitable after costs — consistent with our finding.

- **Jegadeesh (1990).** *Evidence of Predictable Behavior of Security Returns* (Journal of
  Finance). Finds negative serial correlation at the monthly horizon (1-month reversal effect)
  and positive at longer lags (momentum). The one-day horizon studied here sits in a different
  regime: auto-correlations are near zero at the index level.

- **Lehmann (1990).** *Fads, Martingales, and Market Efficiency* (Quarterly Journal of
  Economics). Shows strong one-week reversal in individual stocks, driven by microstructure
  and liquidity; at the index level the effect is greatly reduced — consistent with our
  null result on up-streaks.

## Down-streak reversal / panic and mean reversion

- **De Bondt & Thaler (1985).** *Does the Stock Market Overreact?* (Journal of Finance). The
  foundational paper on contrarian effects at multi-year horizons: past losers outperform
  past winners. The down-streak reversal in this study is the *daily* analogue — a short-term
  version of the same behavioural overshooting mechanism.

- **Jegadeesh & Titman (1993).** *Returns to Buying Winners and Selling Losers* (Journal of
  Finance). Establishes momentum at 3–12 month horizons. Notably absent at the 1-day horizon
  studied here — up-streaks of 1–6 days carry no continuation signal on the index.

- **Cox & Peterson (1994).** *Stock Returns following Large One-Day Declines: Evidence on
  Short-Term Reversals and Longer-Term Performance* (Journal of Finance). Directly relevant:
  large single-day drops are followed by significant positive returns — the daily-scale panic
  reversal this study quantifies across sustained multi-day down runs.

- **Poterba & Summers (1988).** *Mean Reversion in Stock Prices: Evidence and Implications*
  (Journal of Financial Economics). Documents mean reversion at long horizons; the
  short-term flavour seen in down-streak bounces is consistent with the broader reversion
  literature but operates at a much shorter time scale.

## Multiple comparisons and the inference bar

- **Harvey, Liu & Zhu (2016).** *… and the Cross-Section of Expected Returns* (Review of
  Financial Studies). Argues the bar for a new trading factor should be |*t*| ≥ 3.0, not 2.0,
  given the scale of simultaneous testing in the empirical finance literature. With 12
  sub-hypotheses in this study, the Bonferroni threshold is ~0.004 (|*t*| ≈ 2.9 for large
  n), so only the down-streak reversal's *t* = 4.91 clears even this stricter bar.

- **Sullivan, Timmermann & White (1999).** *Data Snooping, Technical Trading Rule Performance,
  and the Bootstrap* (Journal of Finance). Documents how even legitimate looking t-stats can
  be artefacts of searching many rules — our 12 sub-tests are transparent and corrected, but
  the spirit of this caution applies.

## Method lineage (shared desk engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.summarize`](../hot_hand/strategy.py) and [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Binomial test.** Two-sided exact binomial (`scipy.stats.binomtest`) on the continuation
  count, testing H₀: p = 0.5 (a fair coin for the next day's direction).
- **Bonferroni correction.** Classic family-wise error rate control across 12 simultaneous
  hypothesis tests (6 streak lengths × 2 directions).

## Related desk studies

- **[Study 72 — Loaded-Dice](../../72-loaded-dice/)**: the 5-minute SMA(5/10) crossover scalp —
  also asks "is the die loaded?" at intraday frequency. Same null (random walk), same
  machinery (fair-bet baseline), same conclusion: the alleged edge is a coin in costume.
- **[Study 80 — Cold-Open](../../80-cold-open/)**: overnight gap as a directional predictor —
  another folklore claim about market "memory" across sessions.
- **[Study 48 — Groundhog](../../48-groundhog/)**: calendar superstition (Groundhog Day) as
  a market predictor — in the same "fun folklore that is testable and wrong" family.
- **[Study 76 — Rice-Paper](../../76-rice-paper/)**: Bonferroni in a multiple-comparison
  context — the same correction applied to a different set of sub-tests.
