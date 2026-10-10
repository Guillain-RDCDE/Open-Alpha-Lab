# Study 1014 — The Perfect Macro Forecaster 🔮

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is a perfect forecast of the coming quarter's economy worth anything for timing stocks? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | A flawless forecast of the coming quarter's GDP growth, unemployment change and inflation timed the US market to a net Sharpe of **0.29** against **0.25** for buy-and-hold — ΔSharpe +0.04, block-bootstrap p = 0.40, rotation-placebo p = 0.12. The value only appears one quarter *further* out (combined oracle 0.47, p = 0.09 / 0.01; GDP alone 0.58, p = 0.02 / 0.006): the market prices the next quarter, not the current one. |
| **Tradability** — what survives once you only know what has been released? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Last quarter's prints (h = −1) give **0.48** on paper (bootstrap p = 0.08), but **0.13** — below buy-and-hold — with one more quarter of lag; the active return's Newey-West *t* is ≈ 0, and it all runs on **final-vintage** data nobody had at the time, which caps the stamp at Fragile. The perfect *market* oracle reaches 1.53; perfect coming-quarter macro captures **3%** of that gap. |
| **Stocks lead the economy?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | The quarterly excess return correlates **+0.07** with same-quarter GDP growth (t = +0.8) but **+0.27** and **+0.34** with GDP one and two quarters later (t = +3.6, +4.4). |

> **In one sentence:** Even a flawless forecast of the coming quarter's GDP, jobs and inflation would have timed US stocks to a Sharpe of 0.29 against 0.25 for buying and holding, because the market had already moved on to pricing the quarter after — only an oracle that sees that far (0.47) starts to earn its fee; the lagged rule a real investor could run looks better on paper (0.48) but loses to buy-and-hold with one more quarter of lag (0.13).

## What we tested

The premise behind every quarterly outlook: *if you knew where GDP, unemployment and inflation
were heading, you would know where stocks are heading* (the macro-to-markets link of
[Fama 1981](docs/references.md) and [Chen, Roll & Ross 1986](docs/references.md)). We grant it at
full strength — a hypothetical investor gets the **final-vintage** US macro numbers
(`statsmodels` `macrodata`, 1959Q1–2009Q3) *before* each quarter, holds the market (Fama-French
**total return**) when the oracle's number is on the good side of its past-only median and
T-bills otherwise, and we sweep how far ahead the oracle sees: the holding quarter itself, one or
two quarters beyond, or only last quarter's release. Sharpe races are excess-vs-excess, net of
10 bps a switch, tested with a block bootstrap and a rotation placebo, and scaled against an
oracle that knows the market itself. **Dedup:** [877-gdpnow-revisions](../877-gdpnow-revisions/)
trades daily nowcast *revisions*, [387-economic-surprise-index](../387-economic-surprise-index/)
trades data *surprises*, and [268-sahm-rule](../268-sahm-rule/) a recession *trigger* — all real
forecasting signals; this study assumes the forecast is **perfect** and measures the ceiling on
what any of them could ever be worth.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a perfect economic forecast arrives already priced, a dollar grown five ways, and how far the crystal ball would have to see |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | lead-lag with HAC *t*, the 25-cell oracle × horizon grid with block-bootstrap and rotation nulls, the release-lag leg with costs and sub-periods, the market-oracle ceiling, and a planted-lead synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`clairvoyance/`](clairvoyance/). Macro is **final-vintage** (revised); equity is Fama-French **total return**. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
