# Study 23 — Broken-Tether 🔗

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the spread tradably mean-reverting? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The half-life engine cleanly flags a *genuinely* cointegrated pair (synthetic spread half-life ~11 days, reverting), and the trade makes money there — but only thinly, and real ETF pairs are weakly, unstably cointegrated. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Of **45** real ETF pairs, only **1** clears a 0.3 Sharpe out of sample; the first-/second-half Sharpe rank correlation is **+0.17** (in-sample winners don't repeat). Picking pairs on past performance is picking noise. |
| **Stays tethered?** | ![Breaks](https://img.shields.io/badge/Breaks-8b949e?style=flat-square) | The best in-sample pair (**QQQ/EWJ**, Sharpe **+0.61**) collapses to **+0.18** out of sample, its hedge ratio drifting **87%** of its level; and **3%** of *independent random walks* look cointegrated by chance — the search itself is a trap. |

> **In one sentence:** pairs trading is real on a stable cointegrated pair, but stable cointegration among liquid, already-arbitraged ETFs is rare, drifting, and mostly an artefact of selection — so the scanned-universe edge breaks out of sample.

## What we tested

The desk's sixth idea from Kakushadze & Serur, *151 Trading Strategies* (strategy **§3.8**, pairs trading). The steelman, at full strength (Gatev, Goetzmann & Rouwenhorst, *"Pairs Trading: Performance of a Relative-Value Arbitrage Rule"*, **Review of Financial Studies** 2006): find two cointegrated assets, and when the spread `log A − β·log B` stretches far from its mean, bet on reversion — short the rich leg, long the cheap one, close when it converges. We prove the apparatus on a synthetic **cointegrated** pair (and a **spurious** one — two independent random walks — that must, and does, fail), then run the causal z-score book across real ETF pairs and split "is the spread stationary?" from "does the relationship survive out of sample?".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story in plain language: a real rubber band vs a coincidence, the selection trap, and the tether that snaps in the second half |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the machinery: the spread half-life, the in/out-of-sample split, the spurious-pair false-positive rate, and the hedge-ratio drift |

The real run — every fingerprinted, as-of'd ETF number — is in [docs/results.md](docs/results.md); the **beat-7 worked complement** (the *hedge-ratio drift* and the ~zero in/out-of-sample rank correlation) is in [docs/extension.md](docs/extension.md). Reproduce offline via [examples/run_synthetic_demo.py](examples/run_synthetic_demo.py); on the real tape via [examples/verify.py](examples/verify.py) and [examples/extension.py](examples/extension.py) (`--fetch` once to populate the close cache).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
