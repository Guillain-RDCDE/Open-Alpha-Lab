# Study 243 — Graham NCAV

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

> *Do Graham net-nets (price below net current asset value) still exist and still pay?*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | 0–7 net-nets per year; **zero in 2020-2023**. Sample too thin and contaminated by survivorship to support inference. The stocks "selected" are NVDA, AAPL, LRCX — ex-post winners, not cheap cigar-butts. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The screen finds **zero investable stocks** in 4 of the last 4 years in the S&P 500. Uninvestable by construction in large-cap. |
| **Graham's NCAV rule valid in his era?** | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Academic evidence (Oppenheimer 1986, Vu 1988) confirms NCAV worked in small-cap value stocks 1970–1985. It does **not** translate to modern large-cap indices. |

> **In one sentence:** Graham net-nets essentially vanished from the S&P 500 by the 2020s, and the handful that did appear were asset-light tech winners (NVDA, AAPL, LRCX), not the distressed small-cap bargains Graham intended — a classic case of survivorship bias masquerading as a signal.

## What we tested

Benjamin Graham's NCAV criterion: buy stocks where price is below two-thirds of
(current assets minus total liabilities) per share. We apply this screen to the
S&P 500 large-cap universe (2008–2023, 241 tickers with all required EDGAR concepts),
using December month-end market cap from yfinance and 10-K balance-sheet data.

The panel reveals a structural problem: in large-cap, current assets are typically
dwarfed by total liabilities (leverage) and intangible value. NCAV is negative for
most firms. The handful of "net-nets" that pass the screen are asset-light technology
firms where intangible value dominates — the opposite of Graham's intended target.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what NCAV means, why large-cap net-nets aren't what Graham meant, the scarcity problem, and why the apparent "outperformance" is a survivorship artifact |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | annual portfolio table, NCAV ratio distribution, scarcity analysis, survivorship anatomy, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

## Sibling studies

- [122 — Cash-Flow Yield](../122-cash-flow-yield/) — another balance-sheet value screen from the same EDGAR universe.
- [123 — Altman-Z](../123-altman-z/) — distress scoring from the same EDGAR balance-sheet data.
- [138 — Magic Formula](../138-magic-formula/) — Greenblatt's earnings yield + ROIC, also in the Graham tradition.

---

*Engine: [`quantlab/`](../../quantlab/) + [`graham_ncav/`](graham_ncav/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
