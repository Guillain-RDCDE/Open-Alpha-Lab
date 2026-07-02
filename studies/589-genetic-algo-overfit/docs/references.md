# References & literature map — Study 589 (Genetic-Algo-Overfit)

## The claim, at full strength

The seductive pitch: *"Don't hand-design a strategy — let a genetic algorithm **evolve** one. It
searches millions of rule combinations and keeps the fittest, so it finds edges a human never
would."* The pitch is half true. A GA is a strong optimiser and it will indeed find a rule with a
gorgeous in-sample backtest — but on finite data that "fitness" is largely a fit to the training
noise, and the rule dies out of sample. This study makes the trap undeniable by running the GA on a
tape we *built* to have zero timing edge.

## The overfitting / data-snooping literature

- **Bailey, Borwein, López de Prado & Zhu (2014)**, *"Pseudo-Mathematics and Financial
  Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance."* *Notices of the
  AMS* 61(5). The expected maximum Sharpe from N trials grows without bound in N; report a Sharpe
  without its trial count and it is meaningless. The theoretical spine of this study — a GA's
  effective trial count is the number of distinct genomes it evaluated.
- **Bailey & López de Prado (2014)**, *"The Deflated Sharpe Ratio: Correcting for Selection Bias,
  Backtest Overfitting, and Non-Normality."* *Journal of Portfolio Management* 40(5). The **Deflated
  Sharpe Ratio** used here: haircut the observed Sharpe for the number of trials (via the expected
  maximum Sharpe) and the return moments (Lo's skew/kurtosis correction), returning the probability
  the Sharpe is genuinely beyond luck.
- **López de Prado (2018)**, *Advances in Financial Machine Learning*, ch. 11–12. "The most common
  error in finance is to develop a strategy by backtesting until it looks good" — and the specific
  warning that flexible optimisers (including evolutionary and ML methods) overfit hardest because
  they search hardest. The **Probability of Backtest Overfitting** (PBO / CSCV) framing.
- **White (2000)**, *"A Reality Check for Data Snooping."* *Econometrica* 68(5). The formal test for
  whether the best of many rules beats a benchmark once you correct for the search — the ancestor of
  the deflated-Sharpe idea.
- **Lo (2002)**, *"The Statistics of Sharpe Ratios."* *Financial Analysts Journal* 58(4). The
  non-normal standard error of the Sharpe (skew/kurtosis correction) that the DSR uses.
- **Harvey, Liu & Zhu (2016)**, *"...and the Cross-Section of Expected Returns."* *Review of
  Financial Studies* 29(1). Why a *t* of 2 is far too low a bar once you account for how many
  factors (or rules) were tried — the multiple-testing crisis a GA embodies at industrial scale.

## Genetic algorithms in trading

- **Holland (1975)**, *Adaptation in Natural and Artificial Systems.* The genetic algorithm:
  selection, crossover, mutation over a population of genomes — the optimiser evolved here.
- **Allen & Karjalainen (1999)**, *"Using genetic algorithms to find technical trading rules."*
  *Journal of Financial Economics* 51(2). A landmark GA-evolved-trading-rule study: after realistic
  costs the evolved rules earned **no** excess return out of sample — the empirical cousin of this
  demo on real data.

## Neighbours on this bench (the dedup map)

- **[Study 344 — Backtest-Overfitting](../../344-backtest-overfitting/)** — the same trap via a
  plain **grid search** over moving-average crossovers, with the DSR and PBO on a random-walk tape.
  Study 589 swaps the grid for a **genetic algorithm** — an adaptive optimiser whose effective trial
  count is the number of distinct genomes it breeds, and which searches a much larger feature-weight
  space than a fixed grid.
- **[Study 348 — Curve-Fitting](../../348-curve-fitting/)** — optimise a crossover's two windows,
  crown the IS-best, watch it collapse OOS. Study 589 is the *evolutionary* generalisation: the
  optimiser evolves a *multi-feature* rule, not just picks a pair off a grid.

## Shared method

- **In-sample / out-of-sample split** — the honest protocol: evolve on the first half, judge the
  frozen champion on the untouched second half. The IS − OOS shrinkage is the overfitting.
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005) — the placebo null: re-run the
  whole evolve-then-validate protocol on shuffled forward returns and read the OOS Sharpe's tail
  probability.
- **The demeaned timing book** — forward returns are centred before the long/flat position is
  applied, so a long-biased rule cannot inherit the tape's drift; the Sharpe measures *timing* skill
  only. Keeps the parable about overfitting, not beta.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a synthetic
  control is a machinery proof, never market evidence; `REAL` needs a robust *t* ≥ 2 on a real
  tape — which a synthetic-only demo can never provide), one execution lag, costs one-way × NAV, and
  the ≥ 20-seed rule for any synthetic-dependent claim.
