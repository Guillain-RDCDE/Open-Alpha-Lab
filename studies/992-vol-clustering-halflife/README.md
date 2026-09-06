# Study 992 — How Long Is a Storm? 🌪

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does volatility have a stable half-life, or does the number depend on how you ask? | ![Confirmed](https://img.shields.io/badge/Confirmed-2ea44f?style=flat-square) | For SPY over 7,764 sessions, the five standard ways of measuring the half-life of a volatility storm give **11 to 71 days** — a spread of 6.3×. They are not contradicting each other; they are measuring different things, because volatility is not one process. Fitting a two-component autocorrelation gives a **fast part with a 14.1-day half-life carrying 44% of the variation, and a slow part at 172 days** — a fit that beats the single-exponential version by **100%** of squared error. Every single-number estimator returns a weighted average of those two, weighted by whichever lags it happens to look at, which is why GARCH (40 days) systematically exceeds the raw autocorrelation reading (41 days) rather than differing from it at random. The model-free impulse response — after a 95th-percentile volatility day, how long until half the excess is gone — says **51 days**. |
| **Tradability** — is the half-life long enough to act on after costs? | ![Useful](https://img.shields.io/badge/Useful-2ea44f?style=flat-square) | The version that decides anything: after a top-5% volatility day, realised volatility over the next week ran **2.59×** the level following a normal day, over the next month **2.27×**, and over the next quarter 1.90×. That is a slow enough decay to reposition against — nobody needs a same-day reaction to exploit a signal that is still worth 2.27× a month later. Two caveats travel with it: the half-life is itself unstable (across the 7 assets here it ranges 32 to 97 days), and the AR(1) number is partly an artefact of the estimation window — sweeping that window from 1 to 63 days moves the answer from 0.4 to 409 days on identical data, attenuated at the short end by proxy noise and inflated at the long end by the rolling mean's own autocorrelation. |

> **In one sentence:** Volatility's half-life is 11 days or 71 days depending on how you ask, because it is genuinely two processes — a 14-day one and a 172-day one — and every single-number answer is an average of them.

## What we tested

"Volatility clusters" is the most reliably true statement in empirical finance.
The useful question is *how long a cluster lasts* — the number that decides how fast a risk
model should react and how long after a crash you should still be nervous.

This study measures that half-life five ways: an AR(1) on log realised volatility; the lag at
which the autocorrelation of absolute returns crosses half; `log(0.5)/log(α+β)` from a
hand-rolled GARCH(1,1); RiskMetrics' λ = 0.94 inverted; and a **model-free impulse response**
that simply asks how many days after a top-5% volatility day half the excess has gone. The
answers span a factor of several, and none of them is wrong.

The explanation is the study's core. Fitting a **two-component** autocorrelation —
`w·exp(−k/τ₁) + (1−w)·exp(−k/τ₂)` — shows volatility is a fast process with a half-life of days
*plus* a slow one with a half-life of months. Every single-number estimator returns a weighted
average of the two, weighted by whichever lags it happens to emphasise, which is why GARCH
systematically exceeds the raw autocorrelation reading rather than differing at random. A
synthetic control with **exactly known** half-lives confirms the diagnosis: plant one process
and the estimators cluster; plant two and they scatter the way the real tape does. A window
sweep then shows how much of the popular AR(1) number is the smoothing of its own input, and the
whole thing is reduced to the only question a trader asks — after a big day, how much wilder is
the next month?
**Dedup:** distinct from **256-volatility-clustering** (that clustering exists),
**966-garch-vs-har** (forecast accuracy, not persistence structure), **371-vix-term-structure**
(implied rather than realised), **988-bitcoin-volatility-decay** (a trend in the *level* of
volatility) and **991-aggregational-gaussianity** (clustering's effect on distributional shape
rather than its own timescale).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | how long markets stay wild after they go wild, and why five reasonable ways of measuring it give five different answers |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | five half-life estimators graded against planted truths, a two-component autocorrelation fit that explains their disagreement, the window-smoothing artefact, a model-free impulse response, and the cross-asset picture |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`storm/`](storm/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
