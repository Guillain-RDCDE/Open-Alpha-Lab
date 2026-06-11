# Study 51 — Blue-Chip 💎

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do profitable firms out-earn? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The right direction, but modest: the long-high/short-low gross-profitability hedge earns **+3.3%/yr (hit 61%)**, high-GP firms (+19.3%) beating low-GP (+16.0%) — yet Sharpe **0.20 (t ≈ 0.8)**, not significant. |
| **Tradability** — can you confirm and bank it? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Only ~18 years of data (SEC XBRL starts ~2007), survivorship-biased and large-cap — the strong pre-2008 quality decades are invisible to a free EDGAR pull. |
| **"Quality premium is real"?** | ![Supported](https://img.shields.io/badge/Supported-8b949e?style=flat-square) | The sign and the literature agree (Novy-Marx 2013); our short window just can't *confirm* the magnitude. |

> **In one sentence:** the quality (gross-profitability) premium — the factor that survived where size and value faded — points the right way on real SEC data (profitable firms beat unprofitable by ~3 pts/year), but a Sharpe of 0.20 over a too-short XBRL sample can only glimpse it, not confirm it.

## What we tested

The **quality / gross-profitability premium** (Novy-Marx 2013): gross profit ÷ total assets predicts returns — profitable firms out-earn unprofitable ones at the same valuation, the "quality" leg of modern factor models. We build it on **real SEC EDGAR balance-sheet data** (GrossProfit and Assets from 10-K filings) for current S&P 500 members, sort by gross profitability, go long the top quintile and short the bottom, and measure the annual hedge against the universe. We flag the honest limits openly: the XBRL window (~2007+) is short and the universe is survivorship-biased large caps. The offline control is a synthetic firm panel with a known quality premium (and a null).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the "factor that survived" only glimmers in the data we can freely see |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the long-short, the t-stat, the two legs vs the market, and why ~18 XBRL years under-power a real premium |

The fingerprinted real-data run (~150 names, ~2007–2025, fp `184218d2df5c`) is in [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (`--fetch` crawls SEC EDGAR — slow); the offline machinery proof runs on the synthetic panel in [blue_chip/data.py](blue_chip/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
