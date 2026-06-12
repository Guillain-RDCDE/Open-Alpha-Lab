# Study 75 — Knee-Jerk

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Pooled gross **+90.9 bps/trade**, HAC *t* = **+10.70**; beats a random-direction control by **+57.4 bps**; every instrument |*t*| ≥ 2.4 (weakest: JPM). |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Signal decayed −35% post-publication (SPY: +75.8 → +49.0 bps); still *t* = +2.75 post-2009, ~10.8%/yr gross at ~24 trades/yr — but long-only in a bull market and structurally thin at scale. |
| **Decayed since publication?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | Pre-2009 *t* = +4.73 → post-2009 *t* = +2.75 on SPY; pooled post-2009 still *t* = +7.36 — real, but ~35% thinner. |

> **In one sentence:** the Connors RSI(2) mean-reversion signal is genuinely real — at t=+10.70 pooled one of the desk's clearest edges — but it has decayed by a third since publication, is structurally long-only in a bull market, and the canonical 200-SMA filter *hurts* rather than helps.

## What we tested

Larry Connors and Cesar Alvarez documented the RSI(2) system in *Short Term Trading Strategies That Work* (2008/2009): buy when the 2-period RSI falls below 10 (an extreme oversold reading), exit when RSI(2) rises above 60 or after 10 trading days, optionally only trade above the 200-day SMA. We take that literally: run it on SPY and four large caps (QQQ, AAPL, MSFT, JPM) with daily bars since 1993, pin it against a **random-direction control** on the same entry bars (the only fair measure of information content), and split pre/post 2009 to check whether the strategy was arbitraged away after publication. A deterministic synthetic tape with tunable mean-reversion serves as the positive control — confirming the engine finds the edge exactly when anti-persistence exists.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the recipe, why RSI(2) bounces really do work in plain language, the 200-SMA filter trap, the publication-decay story |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-instrument HAC *t*, pre/post-2009 decay bootstrap, cost sweep, vs random control, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`knee_jerk/`](knee_jerk/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
