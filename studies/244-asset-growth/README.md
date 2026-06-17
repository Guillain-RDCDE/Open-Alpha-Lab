# Study 244 — Asset-Growth

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

> *Do fast-growing-asset companies underperform the disciplined ones?*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | On the survivor-biased S&P 500 panel: hedge −1.0%/yr, HAC *t* = −0.45 — wrong direction, zero signal. High-AG firms slightly *outperform* on this panel (fast-growing survivors = tech giants). |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No signal to trade. The long-low-AG / short-high-AG hedge inverts direction on this panel. Even if the academic effect is real in its original small/mid-cap universe, it does not appear here. |
| **Survivorship bias?** | ![Selection_dominates](https://img.shields.io/badge/Survivorship-Selection_dominates-8b949e?style=flat-square) | Panel is current S&P 500 members projected back. Fast-growing firms that survived = tech giants. Fast-growing firms that failed are absent. The selection effect reverses the predicted sign. |

> **In one sentence:** Cooper, Gulen & Schill (2008) documented a robust asset-growth anomaly on a broad US universe, but on a survivorship-biased S&P 500 panel the effect vanishes — fast-growing large-cap survivors (Apple, Nvidia, Amazon) outperform, reversing the predicted direction.

## What we tested

Cooper, Gulen & Schill (2008) argue that firms with high total-asset YoY growth earn
low future returns because managers over-invest and markets over-extrapolate past growth.
We compute AG = (Assets_t − Assets_{t-1}) / Assets_{t-1} from the shared EDGAR cache,
sort the current S&P 500 survivors into quintiles on this signal, lag fundamentals by
one full year (fiscal year y → calendar year y+1 returns), and test whether the low-AG
quintile outperforms the high-AG quintile. A deterministic synthetic tape with tunable
AG premium serves as the positive control.

The panel is **survivorship-biased**: it covers only firms that remain in the S&P 500
as of 2026. Critically, the high-AG group on a survivor panel skews toward successful
technology and growth companies that dominated the 2010s — the opposite of the firms
the anomaly was built around (aggressive expanders that subsequently failed).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the asset-growth recipe in plain English, why the survivor panel reverses the sign, year-by-year results |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | quintile return monotonicity (absent), HAC t-stats, random-portfolio null distribution, synthetic positive control sweep |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`asset_growth/`](asset_growth/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
