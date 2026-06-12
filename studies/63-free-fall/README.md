# Study 63 — Free-Fall 🪂

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there a short-vol carry? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Yes. Selling volatility earns the variance risk premium: SVXY compounded **+11.8%/yr** (post-Volmageddon, de-levered, **+10%/yr**). |
| **Tradability** — can you hold it? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Skew **−4.8**, a **−95% drawdown**, an **−83% day** (Volmageddon, 6 Feb 2018) — five crash days wiped 95% between them. Survivable only sized and hedged. |
| **"Naive short-vol survives"?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The 1× inverse-VIX ETPs were *liquidated* in Feb 2018. Bought-and-held, it blows up. |

> **In one sentence:** selling volatility is a real, persistent carry (+10–12%/yr) and the textbook "picking up pennies in front of a steamroller" — skew −4.8, a −95% drawdown, an −83% single day — so the premium is harvestable only when sized small and tail-hedged; a naive short-vol hold is not survivable.

## What we tested

The **short-volatility carry** — selling volatility to harvest the variance risk premium (the inverse-VIX ETF SVXY). We compare SVXY to SPY over 2018–2026 (SVXY's free history, which centres on the **Volmageddon** crash): CAGR, Sharpe, volatility, drawdown and **skew**; the single worst day; and the share of the total loss concentrated in a handful of crash days. The offline control is a synthetic world that generates a steady carry plus rare catastrophic crashes (and a no-crash null).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a +12%/yr strategy with a −95% drawdown is a trap, and what "Volmageddon" was |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the carry-vs-crash decomposition, the −4.8 skew, the −83% day, the post-2018 resumption |

The fingerprinted real-data run (SVXY vs SPY, 2018–2026, fp `893091a77c52`) is in [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (`--fetch` to download); the offline machinery proof runs on the synthetic world in [free_fall/data.py](free_fall/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
