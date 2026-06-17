# Study 278 -- Sunshine-Effect

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**Does NYC sunshine lift the NYSE (Hirshleifer-Shumway)?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Cloud->return slope **-0.93 bps/octa** (Newey-West t = **-1.70**); sunny-cloudy gap **+4.1 bps/day** (HAC t = **1.94**). Right sign, stable across sub-periods, literature-backed -- but both HAC t below the |t| >= 2 bar on this tape. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Long-on-sunny sleeve churns ~half the days; net of a 1 bp/side cost it earns **4.4%/yr** vs **6.1%/yr** buy-and-hold, net Sharpe **0.39 ~= 0.39**. Breakeven cost is sub-basis-point. |
| **Busted?** | ![Mostly](https://img.shields.io/badge/Mostly-8b949e?style=flat-square) | Real-but-uneconomic: Hirshleifer-Shumway themselves found no after-cost edge, and our reconstruction agrees. |

> **In one sentence:** the sunshine effect is the rare anomaly that is *directionally* honest -- the cloud->return sign comes out negative in every sub-period, just as the mood story predicts -- but it is too small to clear significance on this tape and far too small to pay its transaction costs.

## What we tested

The Hirshleifer-Shumway (2003) "good day sunshine" effect (after Saunders 1993):
morning cloud cover over the exchange city is negatively related to the same-day
index return, via a mood channel. We pair S&P 500 (`^GSPC`) daily returns with a
**de-seasonalised** NYC cloud-cover series, regress returns on the cloud anomaly
with a **Newey-West HAC** standard error, contrast sunny vs cloudy days, and run
a costed long-on-sunny sleeve against buy-and-hold. A synthetic positive control
confirms the machinery detects a planted effect.

**Honesty note.** The trading-day cloud series is a *deterministic climatological
reconstruction* drawn from NYC's monthly sky-cover normals (hardcoded in
`data.py`), not a hand-collected station log -- so the real-tape result is an
**upper bound**, named on the Signal axis. By the desk rubric, a Real stamp needs
a robust HAC |t| >= 2 on the real tape; literature support alone is Weak -- and
on this tape the HAC t does not clear the bar.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, de-seasonalisation, sunny-vs-cloudy in plain English, the cost wall |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | Newey-West regression slope, HAC contrast, sub-period stability, costed sleeve, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`sunshine_effect/`](sunshine_effect/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
