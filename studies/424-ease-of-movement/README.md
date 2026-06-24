# Study 424 — Ease of Movement 🪶

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does "effortless movement" predict returns? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | EOM(14) long/flat on SPY earns net Sharpe **+0.684**, **HAC *t* = +3.18** (clears *t* ≥ 2), sign-shuffle permutation *p* = **0.0005**, robust across smoothing windows 14–40 and confirmed on QQQ/DIA. An advance on light volume genuinely precedes positive forward returns — EOM is a real (if weak) trend detector. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | It **loses the race against buy-and-hold** (Δ *t* = −1.71; +7.7% vs +12.8%/yr) and is **statistically indistinguishable from a price-only SMA(50/200) cross** (Δ *t* = −0.61). Its only win is a halved drawdown — *de-risked beta*, not alpha. You'd get the same from a two-line moving average. |
| **Better than a moving-average cross?** | ![Not_supported](https://img.shields.io/badge/Beats_an_SMA_cross%3F-Not_supported-8b949e?style=flat-square) | The volume weighting that is EOM's whole selling point adds **no measurable advantage** over a plain SMA cross (Δ Sharpe −0.005). "Ease of movement" is real momentum wearing a volume costume. |

> **In one sentence:** Richard Arms' Ease of Movement does carry a real, statistically-significant trend signal (SPY net Sharpe +0.68 at HAC *t* = +3.18, permutation *p* = 0.0005) — but it never beats buy-and-hold, it is indistinguishable from a price-only SMA(50/200) cross, and its only genuine edge is the lower drawdown any trend filter buys you, so the famous volume weighting earns nothing.

## What we tested

We build EOM(14) on 20 years of daily ETF bars (SPY headline, plus QQQ/IWM/DIA/EFA/EEM), cache-first via `yfinance`, and turn the folk rule into a **daily long/flat timing series** — long when EOM > 0, flat otherwise — entered with **one documented execution lag** (signal at the close of *t*, position held over *t+1*). We race its **NET** (1 bp/leg, shorts pay borrow) **excess-of-cash** Sharpe against buy-and-hold *and* against the two simpler rules an EOM advocate must beat — an SMA(50/200) cross and MACD(12/26/9) — using a HAC *t*-stat on the strategy and on each strategy-minus-benchmark difference, a 2,000-draw sign-shuffle permutation placebo, and cost/period sweeps. A deterministic synthetic positive control with a *planted* volume-coupled edge confirms the engine recovers a signal when one exists (and reads ≈ 0 when it doesn't).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "ease of movement" means, why an effortless advance *does* tend to continue, and the trap that the lower drawdown isn't free money — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the EOM kernel, the long/flat book net of costs, HAC *t* on the rule and on EOM−B&H / EOM−SMA / EOM−MACD differences, the sign-shuffle null, cost/period sweeps, and the synthetic planted-edge control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`ease_of_movement/`](ease_of_movement/). Tapes are total-return-adjusted Yahoo daily bars (`auto_adjust=True`), as-of 2026-05-31. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
