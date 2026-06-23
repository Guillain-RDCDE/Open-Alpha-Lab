# Study 386 — NFCI-Conditions 🌡️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does tightness predict equities? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The eye-catching number — a **−0.73** correlation between the conditions index and stocks — is **contemporaneous**: the index *embeds* equity vol, so "conditions tightened" and "stocks fell" are one event. The *forward* link is correctly signed but tiny and **fails t ≥ 2** at every horizon (13-week Welch *t* = **−0.70**, placebo *p* = **0.13**) and **flips sign** at the tightest cut. Direction-right, significance-absent. |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | "Step out when tight" *does* lift Sharpe (**0.69** vs **0.53** net of costs) — but at the **same CAGR (~7.7%)**, purely by cutting vol **17.2% → 11.7%**. It's a volatility overlay a leverage-matched benchmark erases; the gap **halves** ex-2008 and its bootstrap CI **straddles zero**. |
| **"Step out of stocks?"** | ![Misattributed](https://img.shields.io/badge/Step_out_of_stocks%3F-Misattributed-8b949e?style=flat-square) | The signal's apparent power is the index *being built from the thing it claims to predict*. A conditions index tells you it's **raining now**, not that it'll rain next week. |

> **In one sentence:** a financial-conditions index looks like a market-timing oracle because it is *made of* equity volatility — so tight weeks **are** down weeks (−0.73, contemporaneous) — but once you isolate the **forward** link with a one-week lag the predictive edge is statistically indistinguishable from noise (13-week *t* = −0.70), and the only thing a "step out when tight" rule actually buys you is lower volatility at the same return, which is a risk overlay, not alpha.

## What we tested

The Chicago Fed's **NFCI** (positive = tighter financial conditions) is sold as an equity regime switch: *tight ⇒ sell, loose ⇒ buy*. The true NFCI lives only on FRED, which isn't reachable here, so we **build a transparent conditions proxy** from yfinance instruments — equity vol (`^VIX`), rates vol (`^MOVE`), an IG-vs-Treasury credit spread (`LQD`/`IEF`) and broad-dollar momentum (`UUP`) — z-scored to NFCI's sign (**high = tight**), labelled a proxy throughout. Over **24.5 years** (2002–2026, **1,277** weeks) we split the contemporaneous link (mechanical, the index embeds vol) from the **forward** link (the only tradable one), test tight-week forward S&P returns vs the base rate with a Welch *t* and a placebo null, and run a "step out when tight" timing rule with a 1-week lag and one-way costs. A deterministic synthetic control with a *planted* forward edge confirms the engine is faithful **and** that a strong contemporaneous correlation manufactures **no** forward signal.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a conditions index *looks* like a crystal ball when it's really a thermometer, the difference between "raining now" and "will rain," and why lower volatility isn't the same as more money — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the contemporaneous-vs-forward decomposition, tight-week conditional returns, a Welch *t* + placebo null, the timing backtest with a vol/CAGR split and a block-bootstrap Sharpe-gap CI, and a synthetic faithful-engine / contemporaneous-isn't-predictive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`nfci_conditions/`](nfci_conditions/). The conditions index here is an explicit **proxy** (four yfinance legs), not the Chicago Fed NFCI. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
