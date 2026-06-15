# Study 204 — Talmud-Portfolio

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Talmud Sharpe +0.31 vs 60/40 +0.46; HAC *t* on annual return diff vs 60/40 = **−1.26** (not significant); REIT leg amplifies crashes rather than hedging them. |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Tradable (low turnover, liquid ETFs), but dominated by 60/40 on all risk-adjusted metrics; forfeited ~1.2 pp/yr vs 60/40 over 19 years. |
| **vs 60/40?** | ![No Edge](https://img.shields.io/badge/vs_60/40%3F-No_Edge-8b949e?style=flat-square) | The REIT "land" leg fails its diversification mission — VNQ fell harder than SPY in the GFC (−69% vs −55%) and COVID crash (−42% vs −34%). Bonds hedge; REITs amplify. |

> **In one sentence:** the ancient 1/3–1/3–1/3 Talmud allocation is a reasonable diversified portfolio, but its distinctiveness — swapping half the bond hedge for REITs — is exactly what hurts it; in every major crash VNQ fell harder than SPY, leaving the Talmud blend with a lower Sharpe, higher drawdown, and 1.2 pp/yr less than the simpler 60/40.

## What we tested

The Babylonian Talmud (Bava Metzia 42a, ~3rd–6th century CE) prescribes a three-way split: *"a third in land, a third in business, a third in reserve."* Rendered in modern ETFs: **1/3 VNQ** (real estate), **1/3 SPY** (stocks), **1/3 BND** (bonds/cash), rebalanced annually. We run this against three honest benchmarks — 60/40 (SPY/BND), 100% SPY, and a 50/50 SPY/VNQ blend — over 19 years (BND inception 2007–2026) at 10 bps one-way rebalance cost, using HAC t-stats on annual return differences to assess significance. A deterministic synthetic tape with tunable regime cycles serves as the positive control to confirm the engine would find the diversification benefit if it existed.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the Talmud recipe, why REITs fail as a crisis hedge in plain language, the 60/40 comparison |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stats, annual return table, crash-episode analysis, cost sweep, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`talmud_portfolio/`](talmud_portfolio/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
