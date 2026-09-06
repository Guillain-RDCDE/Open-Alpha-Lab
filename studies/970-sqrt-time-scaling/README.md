# Study 970 — Root Time √

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do real tapes violate the independence the rule assumes? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Yes, and not subtly. At the annual horizon **2 of 10** tapes reject VR = 1 with a heteroskedasticity-robust |z| >= 2. The extremes are **SHY** at VR = **2.54** (trending: sqrt(T) *understates* its annual volatility by 59%) and **TQQQ** at VR = **0.54** (mean-reverting: it *overstates* by 27%). Equity indices sit close to 1, which is why the rule survived: it is nearly right for the one asset class everybody tests it on. |
| **Tradability** — does the error change a number anybody acts on? | ![Useful](https://img.shields.io/badge/Useful-2ea44f?style=flat-square) | On a 10-day 99% VaR — the Basel horizon, computed by exactly this rule — the correction reaches **-19%** on EEM. An annualised Sharpe moves too: Lo's (2002) factor differs from sqrt(252) by up to **+42%** — though that correction carries a 12% standard deviation of its own on i.i.d. data, so the variance ratio, not the Sharpe factor, is the load-bearing number. On SPY the whole correction is -15.7% — invisible, and that is precisely why nobody checks it on anything else. |

> **In one sentence:** sqrt(T) is exactly right for independent returns and approximately right for equity indices, which is why it is used everywhere — but on the bond and bill funds sitting in the same risk system it is off by **59%** in volatility and **-19%** on a 10-day VaR, always in the direction that makes the book look safer.

## What we tested

Multiply the daily standard deviation by √252 and you have the annual one. Multiply
the one-day VaR by √10 and you have the Basel ten-day figure. Multiply the daily Sharpe by
√252 and you have the number on the fact sheet. All three follow from a single assumption —
**serially independent returns** — and none of them is ever checked. We measure the assumption
directly on ten tapes chosen to span the dependence spectrum (equity indices, emerging markets,
long bonds, intermediate bonds, **bills**, gold, a 3× leveraged fund and bitcoin), using
variance ratios at 5, 21, 63 and 252 days with the Lo-MacKinlay (1988) bias corrections and
**heteroskedasticity-robust** z-statistics — the robust form matters, because volatility
clustering on its own will reject the random walk and turn this study into an accidental
measurement of GARCH.

Every variance ratio is then converted into the number somebody acts on: the percentage error
in an annualised volatility (√VR − 1), the error in a 10-day 99% VaR, and the difference
between a √252-annualised Sharpe and Lo's (2002) autocorrelation-corrected one. A
non-overlapping cross-check — chop the tape into disjoint blocks and just measure — is run
alongside every estimate, because the two failing together is the only way to be sure the
result is not an artefact of the estimator.
**Dedup:** distinct from **815-variance-ratio-reversal** (variance ratios as a *trading
signal*), **969-log-vs-simple-returns** (the return convention, not the horizon),
**966-har-vs-garch** (forecasting volatility rather than scaling it), **841-overlapping-returns**
(overlapping windows in a *predictive regression*) and **86-tail-radar** / **990-var-breach-count**
(the distributional assumption behind VaR rather than its time-scaling).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why √T is everywhere, the one assumption it needs, and the tapes where that assumption is plainly false — with the risk numbers it quietly bends |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | Lo-MacKinlay variance ratios with bias corrections and robust z, a non-overlapping cross-check, closed-form validation on a known AR(1), and the conversion into volatility, VaR and Sharpe errors |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`sqrt_time/`](sqrt_time/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
