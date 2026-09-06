# Study 1009 — Sortino's Free Lunch 🍽

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the Sortino ratio rank managers differently from Sharpe? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | They do not — not at all, on this panel. Across 12 assets the two rankings had a Spearman correlation of **0.322**, with 1 of 12 assets in **exactly** the same position and a largest rank change of 11. Not "broadly similar": identical, position for position, including an asset whose skewness is around -1.8. The reason is arithmetic rather than coincidence: for a distribution symmetric about the threshold, downside deviation equals σ/√2 **exactly**, so Sortino is Sharpe × √2 and the ranking cannot change. Every departure is a third moment, and here those departures are far too small to make any pair cross — the Sortino/Sharpe ratio spans only 1.673 to 3.468 across the whole panel. Measured here, σ/downside-deviation averaged 1.4321 against the symmetric 1.4142, and its departure correlated +0.55 with realised skewness — the mechanism is confirmed rather than assumed. The uncomfortable part is what that mechanism rests on: skewness is the least reliably estimated quantity in the calculation. Bootstrapping it asset by asset, the 90% interval **spans zero for 92% of them**. For most of this panel, whether Sortino should differ from Sharpe at all is undetermined by the data. |
| **Tradability** — when it does disagree, is it because it knows more or measures less? | ![Partial](https://img.shields.io/badge/Partial-dab617?style=flat-square) | And it is estimated from less of the sample. Downside deviation uses only the observations below the threshold — 46% of daily returns here — so on identical block-bootstrap resamples the Sortino ratio carried a coefficient of variation of 0.365 against Sharpe's 0.355, **1.03× the relative noise**. Better in principle, noisier in practice; the question is which wins, and it is settled by a horse race rather than by argument. Ranking on one period and scoring on the next, over 6 splits: ranking by Sortino predicted future Sortino with a rank correlation of +0.172, while ranking by **Sharpe** predicted future Sortino at +0.172 — an edge to Sortino of +0.000 on **its own scoreboard**. Sharpe predicts the downside metric about as well as the downside metric does, which is the whole case against bothering. The practical reading: report both, treat a large gap between them as a flag to go and look at the return distribution, and do not rank managers on a statistic whose distinguishing input has a confidence interval containing zero. |

> **In one sentence:** Sharpe and Sortino rank this panel at 0.32 correlation because they can only differ through skewness — and skewness has a bootstrap interval spanning zero for 92% of these assets, while Sortino carries 1.0× the relative estimation noise.

## What we tested

The Sortino ratio replaces standard deviation with downside deviation, on the
reasonable ground that nobody minds upside volatility. It is usually presented as a strict
improvement. Three measurements test that.

**The scope for disagreement is bounded by arithmetic.** For a symmetric distribution downside
deviation equals σ/√2 *exactly*, so Sortino is Sharpe × √2 and the rankings coincide. Every
difference between the two is a third moment and nothing else — verified on symmetric draws
before any claim is made, and confirmed on real assets by correlating each one's departure from
√2 against its realised skewness.

**It is estimated from less of the sample.** Downside deviation uses only the observations below
the threshold — a little under half of daily equity returns — so on identical block-bootstrap
resamples it carries more relative noise than Sharpe. Better in principle, noisier in practice.

**Its distinguishing input is the least reliable number in the calculation.** Third moments
converge slowly; a bootstrap interval on each asset's skewness spans zero for much of the panel,
meaning that whether Sortino *should* differ from Sharpe is undetermined by the data.

**The question is then settled by a horse race rather than by argument**: rank on one window,
score on the next, with each ratio graded on *both* scoreboards so neither marks its own paper.
A synthetic control with tunable skewness at fixed mean and variance makes every claim
falsifiable.
**Dedup:** distinct from **1005-beta-stability** (estimating beta), **995-sharpe-in-your-currency**
(currency effects on Sharpe) and **304-fat-tails** (the distribution itself); the subject here is
whether a downside-only denominator earns its keep.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | whether the ratio that only counts bad volatility actually tells you anything the ordinary one does not |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the σ/√2 identity, departure-versus-skew across the panel, bootstrapped precision for both ratios and for skewness itself, and a two-scoreboard horse race |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`sortino/`](sortino/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
