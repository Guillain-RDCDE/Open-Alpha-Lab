# Study 140 — Amihud-Illiquidity

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | Hedge Q5−Q1 **+12.95%/yr**, HAC *t* = **+17.0**, **18/18 years** positive — but the panel is survivorship-biased (current S&P 500 projected back); this is an upper bound on the live effect, not a live premium. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | On a properly-constructed ex-ante large-cap universe the Amihud premium is absent (Hasbrouck 2009); the premium lives in micro-caps, not in S&P 500 names; the bias-inflated spread is not implementable. |
| **Survivorship** | ![Confirmed](https://img.shields.io/badge/Survivorship--Bias-Confirmed-8b949e?style=flat-square) | 100% hit rate and *t* = 17 on 18 observations are hallmarks of a panel-construction artefact: Q5 = smaller S&P 500 names that happened to survive and grow by 2026. |

> **In one sentence:** Amihud ILLIQ sorts S&P 500 names into quintiles that look spectacularly profitable — 18/18 years positive, *t* = 17 — but the signal is a survivorship-bias ghost: only the winners are in the current-index panel, and the "illiquid" bucket is dominated by small-cap survivors whose future returns were unknowable at the time of the sort.

## What we tested

Amihud (2002) proposed ILLIQ = mean(|r_d|/DolVol_d) as a cheap price-impact proxy and showed
that less-liquid stocks earn higher returns as compensation for liquidity risk. We steelman the
claim and test it on an S&P 500 panel: compute ILLIQ for each stock-year from Yahoo Finance
daily data, sort into quintiles, and measure Q5 (illiquid) − Q1 (liquid) forward-year returns.
Against this we place a random-quintile control and a synthetic positive control (a panel where
the premium is planted). The honest caveat is structural: the EDGAR cache contains only current
S&P 500 members projected backwards, so the panel is survivorship-biased — positive results
are upper bounds, not live-tradable edges. The Amihud premium lives in micro-caps; on large-caps
the post-cost evidence is weak to absent even without survivorship concerns.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the ILLIQ idea in plain English, the 18/18 result, the survivorship trap, why the premium lives in micro-caps |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-year quintile returns, HAC *t* on the hedge, random-quintile null, monotonicity, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`amihud_illiquidity/`](amihud_illiquidity/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
