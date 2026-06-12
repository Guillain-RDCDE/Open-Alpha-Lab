# Study 31 — Trade-Winds 🌬️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the time-series-momentum premium real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) on this tape | It pays across **every** lookback (1-to-12-month Sharpe 0.06→0.65) with the trend-follower's tell-tale **positive skew (+1.5)** — but the headline blend's net Sharpe **0.30** is only Lo/HAC *t* **≈ 1.6** over ~26 years, below the desk's robust-inference bar. The classic **12-month leg clears it** (*t* ≈ 3.3), and three decades of literature argue the premium exists; this sample alone can't certify the blend. (The synthetic control, Sharpe 2.4 vs null 0.2, is a *machinery* proof — the premium is wired in by construction.) |
| **Tradability** — does the trend book beat just owning the basket? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Standalone, no. Net Sharpe **0.30** (2 bp) trails the always-long diversified basket (**0.51**) and 60/40 (**0.48**), it endured a decade-long drought (sub-period Sharpe **0.83 → −0.28 → 0.29**), and it breaks even around **5 bp**. Not a standalone return engine. |
| **Crisis alpha?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | This is the real prize. The book earned **+23.6%/yr in the 31 worst equity months** vs +1.0% the rest of the time, at a **−0.07** correlation to stocks (an in-sample split at n = 31 — descriptive, no significance test possible). Blend a 30% trend sleeve into a 60/40 and its Sharpe rises **0.48 → 0.56** while its drawdown is cut **−34% → −20%**. |

> **In one sentence:** cross-asset trend-following is a premium with a century of literature behind it that our own 26-year tape stamps `WEAK` (blend *t* ≈ 1.6; only the 12-month leg clears at *t* ≈ 3.3), and on its own it won't beat simply holding a diversified basket — but it's genuinely uncorrelated insurance that **pays you to hold it**, lifting a normal portfolio's Sharpe and roughly halving its drawdown; the edge isn't prediction, it's diversification.

## What we tested

The one strategy a quant desk keeps coming back to is **time-series momentum** (Moskowitz, Ooi & Pedersen 2012; Hurst, Ooi & Pedersen's *"A Century of Evidence on Trend-Following"* 2017): ask each market on its own *"has it been going up or down lately?"*, go long the risers and short the fallers, size every market to equal risk, and hold a diversified basket across equities, bonds, commodities and FX. It's the engine of the managed-futures industry, and its reputation is **crisis alpha** — making money precisely when stocks crash, because crashes are trends. We build a faithful version (blended 1/3/12-month signal, per-market vol-scaling, portfolio vol target) and run it on **18 liquid continuous futures back to 2000**, charging real (small) transaction costs — then ask the only question that matters: does it earn its place *in a portfolio*? The control is a synthetic regime-switching trend panel (and a random-walk null) that exercises the machinery offline.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "the trend is your friend" is the cliché that comes closest to surviving, the crisis-alpha it really buys, and why it only shines *inside* a portfolio |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | TSMOM vs the long-only basket, the cost & lookback sweeps, the sub-period decay, the crisis-alpha decomposition, the blend that lifts portfolio Sharpe |

The fingerprinted real-data run (18 futures, 2000–2026) is in [docs/results.md](docs/results.md); the **beat-7 worked complement** — *breadth is the lever*, where widening to a 27-market universe lifts the standalone Sharpe 0.30 → 0.55, past the benchmarks — is in [docs/extension.md](docs/extension.md). Reproduce offline on the synthetic control via [examples/run_synthetic_demo.py](examples/run_synthetic_demo.py); on the real tape via [examples/verify.py](examples/verify.py) and [examples/extension.py](examples/extension.py) (`--fetch` to download the baskets).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
