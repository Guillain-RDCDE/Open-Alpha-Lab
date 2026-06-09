# Study 03 — Fear-Gauge 🌡️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) on the level · ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) on the spike | VIX≥30 beats a random day by **+1.0% at 1wk** (p≈0.00) and **+1.3% at 1mo** (p≈0.01); but the famous **+30% spike** earns **−0.02% at 1mo** (p≈0.51) — its whole edge is the 2016–2026 window. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The level's excess **does not significantly beat just buying a −3% price day** (gap p≈0.13–0.20), is borderline once clustering is respected (bootstrap p≈0.05, CI touches 0), and traded it sits in cash ~88% of the time and **underperforms buy-and-hold**. |
| **The "double down at 50"?** — is the martingale survivable? | ![Ruin--prone](https://img.shields.io/badge/Ruin--prone-8b949e?style=flat-square) | Held a quarter, the worst episode draws down **−33%** (−40% over six months); the 2016–2026 window that sells the rule caps the worst *terminal* loss at −3.6% and hides all of it. |

> **In one sentence:** the fear gauge genuinely carries information — a high VIX really is followed by a real rebound — but it's the **variance risk premium** (you're paid to hold the tail), it barely beats the price drop we already studied in [02](../02-falling-knife/), and the "double-down" martingale is a risk-of-ruin generator the cherry-picked chart window conveniently hides.

## What we tested

Two claims that get routinely conflated. The **"VIX rule"** (a *level*): *"Buy stocks when VIX hits 30. Double down when VIX hits 50."* — panic as a buy signal with a martingale bolted on. And the **Altucher chart** (a *spike*): *"S&P after every VIX +30% single-day spike, 2016–2026: avg +2.66% next month, 21/23 positive."* ([@jaltucher](https://twitter.com/jaltucher)). A level and a one-day jump are not the same trigger, so we test the whole family — level ≥ K, +30% spike, spike-by-base-level, and the martingale — through the same event study, random-day benchmark and backtest as Study 02, apples-to-apples.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story + the stakes, plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the full teardown: inference, confounds, capacity |

Both render inline on GitHub (pre-executed). Also here: reproducible headline tables via [`examples/verify_real.py`](examples/verify_real.py) (and an offline [`run_synthetic_demo.py`](examples/run_synthetic_demo.py)).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
