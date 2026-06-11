# Study 19 — Rubber-Band 🪀

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a low close really bounce? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Yes, unmistakably. Bucket days by Internal Bar Strength and a close near the low earns far more next session than one near the high — real-ETF basket gross Sharpe **+1.56**, Newey–West *t* = **+8.6**, with the synthetic random-walk null flat. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | It trades **every day** (~170×/yr turnover), so the whole edge lives inside the bid-ask: basket break-even **8.1 bps**, and net of a 3 bp round-trip the **last-five-year** Sharpe is **−0.37**. The biggest bounce sits in the thinnest, widest-spread country ETFs — the names you can't trade cheaply. |
| **Still alive?** | ![Decayed](https://img.shields.io/badge/Decayed-8b949e?style=flat-square) | A once-strong, now-famous microstructure edge, competed away: gross Sharpe **+2.09 → +0.88** (first → second half), just **+0.40** over the last five years. |

> **In one sentence:** a genuine one-day mean-reversion bounce — real at *t* > 8 gross — that turns over daily, lives entirely inside the spread, and has decayed to negative net returns in the modern sample.

## What we tested

The desk's second idea from Kakushadze & Serur, *151 Trading Strategies* (strategy **§4.4**, ETF mean reversion). The steelman, at full strength (Connors & Alvarez, *short-term ETF reversal*): **Internal Bar Strength**, `IBS = (Close − Low) / (High − Low)`, is a beloved short-horizon signal — a bar that closes near its low (IBS ≈ 0) tends to *bounce* the next day, a liquidity-provision / mean-reversion effect. We prove the apparatus on a synthetic OHLC tape with a *baked-in* IBS→next-day reversal (and a random-walk null that must — and does — kill every leg), then run the single-asset IBS timing overlay (`w = 1 − 2·IBS`, long after a low close) across a basket of 14 liquid ETFs (split-only daily OHLC).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story in plain language: why a low close snaps back, why daily trading meets the spread, and the edge that's quietly fading |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the machinery: IBS-bucketed next-day returns, the Newey–West *t*, the break-even cost, the decay curve, and the random-walk null |

The real run — every fingerprinted, as-of'd ETF number — is in [docs/results.md](docs/results.md); the **beat-7 worked complement** (the *realistic-spread test* — charge each ETF its own spread, and almost none clears) is in [docs/extension.md](docs/extension.md). Reproduce offline via [examples/run_synthetic_demo.py](examples/run_synthetic_demo.py); on the real tape via [examples/verify.py](examples/verify.py) and [examples/extension.py](examples/extension.py) (`--fetch` once to populate the ETF cache).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
