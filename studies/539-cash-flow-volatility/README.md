# Study 539 -- Cash-Flow-Volatility

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

> Do firms with steadier cash flows quietly out-earn the cash-flow lottery tickets?

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Long-low / short-high CF-vol earns **-3.43%/yr**, HAC *t* = **-1.03** (wrong sign), placebo *p* = **0.36**. Robust across quintile/tercile/half splits. The Huang (2009) anomaly does not reproduce here. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Negative gross (-3.43%/yr) and net (-4.61%/yr), Sharpe **-0.35**, max DD **-34.7%**. No edge for costs or borrow to erode. |
| **Huang (2009) direction?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | High-CF-vol leg *won* (**+18.67%** vs **+15.24%/yr**). Survivorship-biased basket (names still trading 2026) + thin ~5-quarter yfinance fundamentals -- both **named**. |

> **In one sentence:** the cash-flow-uncertainty anomaly -- stable cash generators should out-earn volatile ones -- is plausible in theory and documented by Huang (2009), but on a 40-name large-cap survivor basket with yfinance's thin quarterly fundamentals it shows up with the **wrong sign** and no significance (*t* = -1.03, placebo *p* = 0.36) -- None signal, Mirage tradability.

## What we tested

Huang (2009): rank names by trailing cash-flow volatility (std of quarterly operating cash
flow / total assets), go **long the low-CF-vol tercile** and **short the high-CF-vol tercile**,
equal-weight, one execution-lag day, monthly returns. Basket: 40 large-cap names, yfinance
daily prices 2018-2025 (95 monthly observations) + per-ticker quarterly cash-flow statements.
A 20-seed synthetic positive control confirms the engine recovers a planted premium; the real
tape is silent. Universe is survivorship-biased and the CF-vol signal rests on only ~5 quarters
of yfinance fundamentals -- both named, results treated as upper bounds.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the cash-flow-uncertainty idea in plain language, synthetic positive control, real long-short, honest verdict |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | 20-seed control sweep, CF-vol cross-section, year-by-year, equity curve and drawdown, split-fraction robustness, placebo null |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`cash_flow_volatility/`](cash_flow_volatility/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
