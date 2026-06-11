# Study 30 — House-Edge 🎰

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the risk management actually work? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | The vol-targeted, trend-gated dip-buyer **more than halves the drawdown**: max-DD **−24.7%** vs buy-and-hold's **−55.7%** on the S&P 500 (1990–2026), Calmar **doubled** (0.38 vs 0.19). Real, robust risk control. |
| **Tradability** — does the *levered* book beat buy-and-hold once financed honestly? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No — and the line where the dream dies is the **retail markup**. Funded flat at the bill it nearly ties the index (CAGR **10.1%** vs **10.6%**, *better* excess Sharpe 0.54 vs 0.50); at the 1.5–3% markup a CFD actually charges, it makes **7.4%** at excess Sharpe **0.38**. The edge never goes positive at any markup. |
| **Free leverage?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | A CFD charges its markup on the **whole notional**, not the borrowed slice: **2.65 pts/yr** vs a margin account's **0.66** at the same 2.5% — **~2 pts/yr of rent on money you never borrowed**. That house edge, not financing itself, is what kills the pitch. |

> **In one sentence:** a vol-targeted contrarian dip-buyer genuinely halves your drawdown and, funded at the fair bill rate, nearly matches the index — but it never *beats* it, and the retail CFD pitch is a mirage because the broker's 1.5–3% markup is charged on your entire position, borrowed or not.

## What we tested

Every levered-ETF and CFD timing product makes the same pitch: *take a smart timing model — buy the dips, target volatility, gate on the trend — add leverage, and you beat buy-and-hold.* Leverage, the story goes, just amplifies a good signal. We build a faithful version of that strategy (RSI(2) contrarian entries, `target_vol / realised_vol` sizing, a 200-day trend gate) and run it on the S&P 500 back to 1990 under two **honest** financing models that charge every dollar exactly once: a *margin account* (borrow only the slice above 100% at T-bill + markup, idle cash earns the bill) and a *futures/CFD account* (the whole notional carries T-bill + markup, capital stays in bills). The two coincide exactly at a zero markup — what separates them is where the broker's markup lands. The control is a synthetic GARCH-with-bear-regimes index where the same machinery is exercised offline; the headline run is the real tape.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why levered timing *feels* like a free lunch, the drawdown protection it really buys, and the markup line where the brochure's promise dies |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the two account types and their zero-markup identity, the markup sweep by account, excess-Sharpe accounting, the house-edge identity |

The fingerprinted real-data run (S&P 500, 1990–2026) is in [docs/results.md](docs/results.md); reproduce it offline on the synthetic control via [examples/run_synthetic_demo.py](examples/run_synthetic_demo.py), or on the real tape via [examples/verify.py](examples/verify.py) (`--fetch` to download ^GSPC + ^IRX).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
