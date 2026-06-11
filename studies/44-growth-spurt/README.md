# Study 44 — Growth-Spurt 📈

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do low-asset-growth firms beat high-growth ones? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Not on tradable large caps. The long-low/short-high hedge earns **−1.6%/yr (Sharpe −0.12)** over 2005–2025 — the sign is **backwards**: high-growth firms *out*-earned low-growth (+21.4% vs +19.8%). |
| **Tradability** — can you trade the highest headline Sharpe on the list? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No. The vendor's **0.835** doesn't replicate where you can hold names at scale; the effect hides in micro-caps and is subsumed by the Fama-French investment factor. |
| **"Sharpe 0.835 replicates"?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Even a *survivorship-biased* large-cap test — which should flatter the strategy — can't find it. |

> **In one sentence:** the juiciest number on the vendor list (Sharpe 0.835) is a mirage for anyone trading liquid stocks — on large-cap survivors the asset-growth hedge is *slightly negative* — because the effect, to the extent it's real, lives in micro-caps and is already captured by a standard factor, i.e. exactly the illiquid, survivorship-fragile corner where headline Sharpes are manufactured.

## What we tested

The **asset-growth effect** (Cooper, Gulen & Schill 2008): firms that grow their total assets fast subsequently underperform. It carries the **highest headline Sharpe (0.835) on [paperswithbacktest](https://github.com/paperswithbacktest/awesome-systematic-trading)'s open list** — the most tempting thing in the backlog. We rebuild it on **real balance-sheet data** — fiscal-year total assets from **SEC EDGAR** for ~399 current S&P 500 members, annual returns from Yahoo, 2005–2025 — sort firms by asset growth, go long the slow growers and short the fast ones, and ask the one question that matters: **does the famous Sharpe survive on names you could actually trade?** We flag both caveats openly (the universe is large-cap and survivorship-biased). The offline control is a synthetic firm panel with a tunable growth penalty (and a null).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the best-looking number on the list vanishes on real, tradable stocks, and where the effect actually hides |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the long-short construction, the backwards sign on large caps, the micro-cap / investment-factor / survivorship explanations |

The fingerprinted real-data run (~399 S&P 500 names, 2005–2025, fp `16e4328571d8`) is in [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (`--fetch` crawls SEC EDGAR — slow); the offline machinery proof runs on the synthetic panel in [growth_spurt/data.py](growth_spurt/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
