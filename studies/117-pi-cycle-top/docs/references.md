# References — Study 117 (Pi-Cycle-Top)

## The claim's origin

1. **Philip Swift (LookIntoBitcoin)** — "The Pi Cycle Top Indicator" (2019, updated 2021).
   The original blog post and charts that popularized the indicator. Argues that 111/(2×350)
   is "encoded in Pi" because 111/35 ≈ π. The post identified two visual coincidences
   (2013, 2017) and predicted the 2021 top. Available at: lookintobitcoin.com/charts/pi-cycle-top-indicator/

2. **Benjamin Cowen (IntoTheCryptoVerse)** — multiple YouTube videos (2021–2024) discussing
   the Pi Cycle Top in the context of the 2021 BTC cycle, the 4-year halving model, and
   predictions for the 2024-25 cycle. The indicator gained mainstream crypto awareness partly
   through this channel; the 2024 non-signal was noted but not widely published.

## The underlying effect — MA crossovers as trend signals

3. **Faber, M.T.** (2007). "A Quantitative Approach to Tactical Asset Allocation."
   *Journal of Wealth Management.* The canonical paper on using moving-average crossovers
   for market timing. Establishes the risk-management framework (reduce exposure when price
   is below the 10-month SMA) that is frequently cited alongside crossover claims.

4. **Brock, W., Lakonishok, J., LeBaron, B.** (1992). "Simple Technical Trading Rules and the
   Stochastic Properties of Stock Returns." *Journal of Finance*, 47(5), 1731–1764.
   Foundational evidence that MA crossovers had statistical content in US equity markets
   pre-1988. The out-of-sample literature since then has largely not confirmed this.

5. **Park, C.-H., Irwin, S.H.** (2007). "What Do We Know About the Profitability of Technical
   Analysis?" *Journal of Economic Surveys*, 21(4), 786–826. Meta-analysis of 95 studies:
   MA-based rules show profits in some early studies but most disappear in out-of-sample and
   transaction-cost tests. The inference bar for MA claims is high.

## The overfitting problem — small-n and data-snooping

6. **White, H.** (2000). "A Reality Check for Data Snooping." *Econometrica*, 68(5), 1097–1126.
   The foundational paper on p-hacking from searching over indicator parameters. The
   Pi-Cycle multiplier (2.0) was hand-tuned over a 2-event history — a textbook example of
   the problem White formalizes. The family-wide error rate from parameter search dwarfs any
   nominal p-value.

7. **Harvey, C.R., Liu, Y., Zhu, H.** (2016). "... and the Cross-Section of Expected Returns."
   *Review of Financial Studies*, 29(1), 5–68. Argues that the minimum t-statistic for a
   publishable anomaly should be 3.0 (not 2.0) given data-mining across many studies. With
   n=2 and a hand-tuned parameter, the effective threshold is immaterial — there is no
   inference to be drawn.

8. **Lo, A.W., Mamaysky, H., Wang, J.** (2000). "Foundations of Technical Analysis: Computational
   Algorithms, Statistical Inference, and Empirical Implementation." *Journal of Finance*,
   55(4), 1705–1770. Systematic evaluation of technical patterns; the authors find some
   statistical support for certain patterns but only over large samples. n=2 events do not
   qualify.

## The Bitcoin-specific context

9. **Ciaian, P., Rajcaniova, M., Kancs, A.** (2016). "The economics of BitCoin price formation."
   *Applied Economics*, 48(19), 1799–1815. Early empirical study of BTC price drivers. The
   4-year halving cycle is a recurring theme in this literature; most studies note the
   challenge of inference given the short history of the asset.

10. **Urquhart, A.** (2016). "The Inefficiency of Bitcoin." *Economics Letters*, 148, 80–82.
    Tests for return predictability in BTC; finds significant autocorrelation and calendar
    effects in early data, suggesting some predictability, but the sample is tiny and the
    patterns have largely diminished.

11. **Baur, D.G., Dimpfl, T., Kuck, K.** (2018). "Bitcoin, gold and the US dollar — a
    replication and extension." *Finance Research Letters*, 25, 103–110. Contextualizes BTC
    as a speculative asset, not a gold equivalent; relevant to the Pi Cycle claim that BTC
    follows systematic cycles.

## Method lineage

12. **Newey, W.K., West, K.D.** (1987). "A Simple, Positive Semi-Definite, Heteroskedasticity
    and Autocorrelation Consistent Covariance Matrix." *Econometrica*, 55(3), 703–708.
    The HAC standard-error estimator used throughout this desk; particularly important here
    because BTC returns are fat-tailed and serially correlated.

## Related desk studies

- [Study 83 — Half-Life](../../83-half-life/): same asset (BTC-USD), same small-n problem
  (n=3–4 halvings), same verdict: compelling visual story, no statistical power.
- [Study 84 — Moon-Math](../../84-moon-math/): another BTC numerology claim (lunar cycle
  effects); same family of n-is-tiny arguments.
- [Study 21 — Fools-Gold](../../21-fools-gold/): the golden cross (50/200 MA) on equities;
  same MA-crossover family, same struggle to clear the inference bar.
- [Study 70 — Digital-Gold](../../70-digital-gold/): BTC as a macro hedge; the secular BTC
  trend dominates any signal.
