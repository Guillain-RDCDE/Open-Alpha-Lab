# References & literature map — Study 839 (The t > 3 Threshold)

## The claim, at full strength

The pitch every factor paper implicitly makes: *"Our anomaly has a t-stat above 2, so it is
real."* Harvey, Liu & Zhu argue this is precisely backwards. The published cross-sectional
"factor zoo" is the visible survivor set of an enormous, largely-unreported search across
hundreds of candidate predictors; against that backdrop, the conventional single-test
**t > 2** bar is far too lax and *guarantees* a paper's worth of false discoveries. The
multiple-testing-adjusted hurdle rises toward and past **t ~ 3.0**, which is why a newly
claimed factor should clear roughly a *t* of 3, not 2. This study makes the arithmetic
undeniable by running it on a synthetic zoo we *built* to contain nothing.

## The source paper and its companions

- **Harvey, Liu & Zhu (2016)**, *"… and the Cross-Section of Expected Returns."* *Review of
  Financial Studies* 29(1), 5–68. **The source of this study.** Catalogues ~300+ published
  factors, argues most cannot survive an honest multiple-testing haircut, applies Bonferroni,
  Holm and Benjamini-Hochberg-Yekutieli to the factor zoo, and derives the recommendation
  that a new factor should exhibit a *t*-ratio exceeding **~3.0**. The 3.0 hurdle and the
  publication haircut this study reproduces are theirs.
- **Harvey & Liu (2014)**, *"Backtesting."* *Journal of Portfolio Management* 41(1). The
  practitioner statement of the same haircut: how to discount an in-sample Sharpe / t-stat
  for the number of trials behind it.
- **Harvey & Liu (2020)**, *"A Census of the Factor Zoo."* SSRN. The updated tally — the zoo
  has only grown, tightening the multiple-testing bind and pushing the honest hurdle higher.
- **Harvey (2017)**, *"Presidential Address: The Scientific Outlook in Financial Economics."*
  *Journal of Finance* 72(4). The p-hacking / publication-bias critique that frames the
  haircut as a scientific-integrity problem, not just a statistics one.

## The multiple-testing machinery implemented here

- **Bonferroni (1936).** Control the family-wise error rate by testing each of N hypotheses
  at level α/N — the two-sided `|t|` cutoff is `Φ⁻¹(1 − α/2N)`, which rises with N (1.96 at
  N=1, ~3.78 at N=316). Implemented in
  [`strategy.bonferroni_t`](../tstat_threshold/strategy.py).
- **Holm (1979)**, *"A Simple Sequentially Rejective Multiple Test Procedure."* *Scandinavian
  Journal of Statistics* 6(2). A uniformly more powerful step-down FWER procedure than
  Bonferroni — [`strategy.holm_reject`](../tstat_threshold/strategy.py).
- **Benjamini & Hochberg (1995)**, *"Controlling the False Discovery Rate."* *JRSS-B* 57(1).
  The step-up procedure that controls the *expected proportion of false discoveries* rather
  than the probability of any — less conservative than FWER control and the appropriate goal
  when a few false factors among many true ones is tolerable.
- **Benjamini & Yekutieli (2001)**, *"The Control of the False Discovery Rate in Multiple
  Testing under Dependency."* *Annals of Statistics* 29(4). The BH variant valid under
  arbitrary dependence, using the harmonic factor `c(N) = Σ 1/i` — the one HLZ lean on for
  correlated factor returns. Both BH and BHY are in
  [`strategy.benjamini_hochberg`](../tstat_threshold/strategy.py).

## The Sharpe/t-stat sampling context

- **Lo (2002)**, *"The Statistics of Sharpe Ratios."* *Financial Analysts Journal* 58(4).
  The sampling distribution of a Sharpe / t-stat under realistic return moments — the ruler
  the whole haircut re-scales.
- **Bailey & López de Prado (2014)**, *"The Deflated Sharpe Ratio."* *Journal of Portfolio
  Management* 40(5). The Sharpe-ratio cousin of the t-stat haircut: correct a reported Sharpe
  for the trial count and non-normality. Same idea, different metric.

## Neighbours on this bench (the dedup map)

- **[Study 346 — Multiple-Testing](../../346-multiple-testing/)** — the *generic*
  family-wise-error demonstration (any battery of tests, any domain). Study 839 is the
  factor-zoo *specialisation*: it is framed around HLZ's specific **3.0 hurdle** and the
  **publication haircut** peculiar to return predictors, not the generic Bonferroni lesson.
- **[Study 536 — Anomaly-Decay-Post-Publication](../../536-anomaly-decay-post-publication/)**
  — what happens to a factor *after* it clears the bar and is published (McLean-Pontiff
  decay). 839 is upstream: whether the bar itself was ever high enough for the "discovery" to
  be real in the first place.
- **[Study 343 — Data-Mining-Roulette](../../343-data-mining-roulette/)** — mining a *single*
  dataset for a lucky rule (one series, many rules). 839 is the cross-sectional dual: *many
  factors*, one bar, and how high that bar must be.

## Shared method

- **The factor-zoo t-stat** — vectorised single-test `mean / (sd/√T)` per candidate, with
  two-sided p-values (normal / Student-t), in
  [`strategy.factor_tstats`](../tstat_threshold/strategy.py).
- **Seed-robust synthetic controls** — every synthetic-dependent claim (the null clearing
  fractions, the mixture FDR collapse) is averaged over ≥ 20 seeds, the house rule.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a
  synthetic control is a machinery proof, never market evidence; `REAL` needs a robust
  *t* ≥ 2 on a real tape — which a synthetic-only demo can never provide), and the three-axis
  verdict (Signal / Tradability / a descriptive myth-check axis).
