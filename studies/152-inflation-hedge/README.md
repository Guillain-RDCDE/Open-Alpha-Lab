# Study 152 — Inflation-Hedge

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | The *negative* effect is real: Fisher beta_nom = **+0.31** (not 1.0), real return Fisher beta = **-0.72** (HAC *t* = **-3.02**); 1-year forward real return gap high vs. low inflation = **-7.1 pp** (HAC *t* = **-4.08**), robust to shuffled-label control (*t* = +0.79). The positive "hedge" claim fails; the negative Fama-Schwert finding is confirmed. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Inflation regimes are multi-year macro epochs (the 1970s, the 2021-23 spike), not actionable timing signals. The 5-year gap narrows to **-1.9 pp pa** — too small, too slow, and unobservable in real time. |
| **Short-run hedge?** | ![No](https://img.shields.io/badge/Short--run_hedge%3F-No-8b949e?style=flat-square) | The Fisher hypothesis fails: stocks absorb only **31%** of each CPI percentage point, not 100%. Real returns fall significantly with inflation — the opposite of protection. |

> **In one sentence:** stocks fail as a short-run inflation hedge — real returns fall when CPI rises (Fisher beta_real = -0.72, *t* = -3.02), high-inflation years deliver 7 pp less real return than low-inflation years (*t* = -4.08), and the regime gap only partially closes over 5 years — confirming Fama & Schwert (1977) on 150 years of Shiller data.

## What we tested

The folk wisdom: *"stocks are a real asset — hold them during inflation because factories and brands are worth more when prices rise."* The academic version (the Fisher hypothesis for stocks) requires nominal returns to rise one-for-one with CPI (beta = 1.0), leaving real returns invariant. We test this directly using the Shiller monthly S&P 500 panel (1872-2023, n = 1,818 months): OLS regressions of trailing 12-month nominal and real total returns on CPI inflation (12-lag Newey-West HAC), and a pre-specified 3% CPI-YoY regime split comparing forward 1-year and 5-year real total returns. A shuffled-label control confirms the regime gap is not a random split artefact. A deterministic synthetic panel with a tunable Fisher beta serves as the positive/negative control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the folk claim, the Fisher scatter plots, the regime gap in plain language, the 5-year horizon cross-check |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | Fisher OLS with HAC t-stats, synthetic beta sweep (positive/negative control), regime t-test with shuffled-label control, rolling inflation vs. real return scatter |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`inflation_hedge/`](inflation_hedge/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
