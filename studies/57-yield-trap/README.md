# Study 57 — Yield-Trap 🪤

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do high-dividend stocks out-earn? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | No. On total return, VYM (high dividend) trailed SPY (market): CAGR **+9.3% vs +10.8%**, Sharpe **0.67 vs 0.76**. The spread is **−1.5%/yr (Sharpe −0.26)**. |
| **Tradability** — is high-yield a better way to own equities? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Lower return, lower Sharpe, *deeper* drawdown — the "income" edge is accounting, not money. |
| **"High-dividend tilt beats the market"?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Dividends are payout form, not a premium (Miller-Modigliani); the tilt is a disguised value/sector bet that lagged. |

> **In one sentence:** the belief that high-dividend stocks are a superior, income-rich way to own equities is a yield trap — on a total-return basis a high-dividend ETF trailed the plain market on both return and Sharpe, because dividends are a payout choice, not a premium, and the high-yield screen is really an old-economy sector bet.

## What we tested

The popular conviction that **high-dividend-yield stocks** beat the market — beloved of income investors and "dividend aristocrat" lore. We test it the only honest way, on **total return** (dividends reinvested), comparing the high-dividend ETF **VYM** to the plain-market **SPY**, 2007–2026: each leg's CAGR, Sharpe and drawdown, and the spread with its t-stat. The offline control is a synthetic two-ETF world with a tunable (counterfactual) dividend premium and a null.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "high dividends = better returns" is an accounting illusion |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the leg-by-leg total-return comparison, the spread t-stat, the value/sector explanation |

The fingerprinted real-data run (VYM vs SPY, 2007–2026, fp `d9a9ba515c71`) is in [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (`--fetch` to download); the offline machinery proof runs on the synthetic world in [yield_trap/data.py](yield_trap/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
