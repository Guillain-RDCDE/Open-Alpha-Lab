# Study 62 — Premium-Seller 💸

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does covered-call "income" beat the index? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | No. QYLD trailed its *own* underlying (QQQ) by **−10.9%/yr (Sharpe −1.05, t −3.6)**: **+8.1% vs +19.3%/yr**, and below SPY too. |
| **Tradability** — is it a better way to own equities? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A brutal asymmetry: it keeps only **50% of up months** but takes **58% of down months** — capping the upside more than the premium cushions the downside. |
| **"Covered-call income beats the index"?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The fat distribution feels like income but is gains given back (Israelov-Nielsen 2015); on total return it's a big shortfall. |

> **In one sentence:** the covered-call "income" ETF (QYLD) trailed the very index it holds by ~11%/yr at a lower Sharpe, because writing calls keeps only half the upside while still taking ~58% of the downside — the high monthly distribution is your own gains handed back, not a premium.

## What we tested

The popular belief that **covered-call income funds** (QYLD, JEPI, "sell premium for steady income") are a superior, lower-risk way to own equities. We compare QYLD to its underlying (**QQQ**) and to **SPY** over 2014–2026 on total return — CAGR, Sharpe, drawdown — and decompose the upside/downside *capture* that explains the result. The offline control is a synthetic world that caps the fund's up-moves (and a null).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a fund that *holds* QQQ lost to QQQ by 11%/yr, and what "income" really means |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the total-return spread with its t-stat, the 50%/58% upside-vs-downside capture asymmetry |

The fingerprinted real-data run (QYLD/QQQ/SPY, 2014–2026, fp `48f465a73194`) is in [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (`--fetch` to download); the offline machinery proof runs on the synthetic world in [premium_seller/data.py](premium_seller/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
