# Study 52 — Smoke-Screen 💨

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do cash-backed earners beat accrual-heavy ones? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Yes. The long-low/short-high accruals hedge earns **+5.9%/yr at Sharpe 0.64 (t ≈ 2.7), 72% of years** — and the long leg (cash-backed, +25.0%) *beats the market*. Significant even on a short sample. |
| **Tradability** — can you bank it? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | The edge is on the costly short side, the data lags filings, and the anomaly is documented to have weakened post-2000 (Green-Hand-Soliman 2011) — a fade ~18 XBRL years can't show. |
| **"Accrual anomaly replicates"?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | A clean replication of Sloan (1996): the quality of earnings genuinely predicts returns. |

> **In one sentence:** Sloan's accruals anomaly — firms whose profits are backed by *cash* beat those whose profits are *accounting accruals* — replicates clearly on real SEC data (+5.9%/yr, t 2.7, 72% hit, the long leg beating the market), a genuine quality-of-earnings signal, though fragile to trade given short-side frictions and a documented post-2000 fade.

## What we tested

The **accruals anomaly** (Sloan 1996): accruals = (net income − operating cash flow) / total assets. High accruals flag earnings the cash doesn't support; those firms subsequently disappoint. The trade is **long low-accruals (cash-backed), short high-accruals**. We build it on **real SEC EDGAR data** (net income, operating cash flow, total assets from 10-K filings) for current S&P 500 members, sort by the accruals ratio, run the annual hedge, and measure its significance. We flag the honest limits: the XBRL window (~2007+) is short and the universe is survivorship-biased large caps. The offline control is a synthetic firm panel with a known accruals premium (and a null).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "cash beats accounting" is a real, sensible edge — and why the long leg beats the market |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the hedge with its t-stat, the two legs vs the market, the short-side and decay caveats |

The fingerprinted real-data run (~190 names, ~2007–2025, fp `b2c8bc41ffd9`) is in [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (`--fetch` crawls SEC EDGAR — slow); the offline machinery proof runs on the synthetic panel in [smoke_screen/data.py](smoke_screen/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
