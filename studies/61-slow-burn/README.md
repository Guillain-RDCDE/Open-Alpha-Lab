# Study 61 — Slow-Burn 🔥

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the volatility drag real? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Yes, and it matches theory: a 3× daily-rebalanced QQQ realizes **~8–13%/yr below** 3×(QQQ's CAGR), in line with **0.5·L·(L−1)·σ² ≈ 12.8%/yr**. |
| **Tradability** — should you hold a 3× ETF? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | TQQQ turned QQQ's +19.8%/yr into +43.7% — but at **no risk-adjusted gain** (Sharpe **0.90 vs 0.98**), an **−82% drawdown**, and **−79% in 2022 alone**. |
| **"Leverage is a free return amplifier"?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | It amplifies *volatility and the tail*, not return-per-risk; "decays to zero" is too glib (it 50×'d in the bull), but "free multiplier" is the real myth. |

> **In one sentence:** the volatility drag of leveraged ETFs is real and matches the textbook formula (~13%/yr for 3× QQQ), but the bigger headline return (+44% vs +20%) bought *no* risk-adjusted benefit — same-to-worse Sharpe, an −82% drawdown and −79% in 2022 — so leverage is a path-dependent bull-regime bet with ruinous tails, not a free amplifier.

## What we tested

The folk belief that **3× leveraged ETFs (TQQQ)** decay to nothing through "volatility drag" — and its flip side, that they're a free way to triple your returns. We compare TQQQ to QQQ over 2010–2026: CAGR, Sharpe, volatility and drawdown; the realized decay (3× the underlying's CAGR minus a self-replicated 3×-daily series) against the textbook **0.5·L·(L−1)·σ²**; and the regime split (2010–21 bull vs 2022 bear). The offline control is a synthetic underlying with tunable volatility, so the drag is provable without the network.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a +44%/yr ETF was still a worse deal than the plain index |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the decay-vs-theory match, the Sharpe/drawdown comparison, the regime path-dependence |

The fingerprinted real-data run (TQQQ vs QQQ, 2010–2026, fp `a35f4f6cebe4`) is in [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (`--fetch` to download); the offline machinery proof runs on the synthetic world in [slow_burn/data.py](slow_burn/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
