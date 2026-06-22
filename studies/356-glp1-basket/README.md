# Study 356 — the GLP-1 / Ozempic basket 💉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there alpha beyond the market / sector? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The basket beats SPY by +9.7%/yr but at HAC-*t* = **1.10** (fails ≥2); the market-model α vs SPY is *t* = **1.78**. The one positive read is vs healthcare (XLV: +15.1%/yr, *t* = 2.01) — and its bootstrap CI is **[+0.2%, +29.3%]**, grazing zero. Post-hoc name selection biases all of it upward. |
| **Tradability** — a harvestable, scalable edge? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A **two-name** bet (corr 0.40) with a **−49%** drawdown, riding one product cycle. Costs are immaterial (≤0.1bp/yr of excess) — what's left is concentrated beta to a story, not an edge. |
| **Two stocks and a story?** | ![Misattributed](https://img.shields.io/badge/Misattributed-8b949e?style=flat-square) | The alpha is **all LLY** (*t* = 2.66); **NVO** — the actual Ozempic maker — adds none (*t* = 0.35) and is **−69% from peak**. And the basket's excess turned **negative** (−3.5%/yr) *during* the 2023–26 GLP-1 mania it's named for. |

> **In one sentence:** the "Ozempic trade" did beat the market on paper, but the edge isn't statistically robust (HAC-*t* ~1.1 vs SPY), it's **entirely one stock** (Eli Lilly, not the Ozempic-maker Novo) wearing a theme's clothes, and it's a concentrated two-name bet that actually *underperformed during the very weight-loss-drug boom it claims to capture*.

## What we tested

The viral **"Ozempic / GLP-1 trade"**: buy **LLY + NVO** — the makers of Mounjaro/Zepbound and Ozempic/Wegovy — as a 50/50 basket to ride the weight-loss-drug boom. We take **2018→2026** daily **total-return** prices (yfinance, adjusted close) and ask two separate questions: is the basket's **excess** over **SPY** (the market) and **XLV** (the healthcare sector) statistically real under autocorrelation-robust (Newey-West) inference — and *where does it come from?* The decisive moves are a single-name **decomposition** (is it one stock?) and a **recency split** at 2023 (did it work *during* the mania, or only before it?). Names picked because they already won → survivorship/recency is named on the Signal axis.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | did the basket beat the market, is it really "the theme," and what NVO's chart says — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC-*t* on mean excess + market-model α vs SPY & XLV, the single-name decomposition, the 2023 recency split, a block-bootstrap CI, the cost sweep, and a planted-alpha synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`glp1_basket/`](glp1_basket/). **Not investment advice** — research & education; concentrated single-name risk is real. See [LICENSE](../../LICENSE).*
