# Study 968 — Which Bootstrap 🥾

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the resampling scheme change what the interval covers? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | It matters, and it matters where you would not look. On an i.i.d. tape every scheme covers near nominal. Once volatility clusters and the tape is fat-tailed, the **Sharpe** interval degrades for all of them — the worst method covers **87%** against a promised 95% — and the spread between best and worst method reaches **9%** of coverage. With genuine AR(1) in the returns the i.i.d. resample is the one that breaks: **97%** coverage of the mean, because it destroys the dependence that inflates the true standard error. |
| **Tradability** — is there one bootstrap a desk should default to? | ![Useful](https://img.shields.io/badge/Useful-2ea44f?style=flat-square) | **Moving block (Kunsch 1989)** is the least-bad default: its worst coverage across every world tested is **100%**-ish, within 0.3% of nominal, and it costs the same as any other resample. On the real tapes the choice moves the published Sharpe interval by up to **17%** of its width (SPY: [+0.34, +0.97]) — enough to change whether a strategy 'clears zero', which is exactly the decision these intervals are used for. |

> **In one sentence:** The bootstrap you pick is not a detail: on dependent, fat-tailed returns the coverage of a nominal 95% Sharpe interval ranges from **87%** to **97%** depending only on how you resample, and the i.i.d. version — the one in every tutorial — is the one that fails first when returns are actually correlated.

## What we tested

Every backtest on this desk — and in most of the literature — ships a bootstrapped
confidence interval, and almost nobody checks whether those intervals keep their promise. A 95%
interval is a claim about *repeated sampling*: build it a thousand times and the truth should
be inside 950 of them. That is measurable, but only on data whose truth you know, so this study
simulates four worlds — i.i.d. Student-t, volatility clustering with no serial correlation,
AR(1) returns, and both together — and measures the empirical coverage of the **i.i.d.**,
**moving-block** (Kunsch 1989), **circular-block** (Politis-Romano 1992) and **stationary**
(Politis-Romano 1994) bootstraps, plus the analytic Sharpe standard errors of Lo (2002) and
Mertens (2002), for two statistics: the mean and the annualised Sharpe ratio. The block length
is swept rather than assumed.

Then the practical half: on **SPY, TLT, GLD and BTC-USD** we publish the same Sharpe ratio five
different ways and measure how much the interval moves — including on a deliberately short
three-year sample, which is the length at which the choice starts changing conclusions rather
than decimal places.
**Dedup:** distinct from **841-overlapping-returns** (the inference problem created by
overlapping windows), **838-hac-necessity** (HAC standard errors in regression),
**833-deflated-sharpe-ratio** and **834-minimum-backtest-length** (multiplicity and sample-size
corrections to a Sharpe, taking the standard error as given), **346-multiple-testing** and
**840-clustered-standard-errors** (cross-sectional dependence).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a confidence interval actually promises, the four ways to resample in pictures, and the worlds where each one quietly breaks |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | coverage by simulation for the mean and the Sharpe across four dependence regimes, block-length sweeps, analytic alternatives, and the same Sharpe published five ways on four real tapes |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`boot_choice/`](boot_choice/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
