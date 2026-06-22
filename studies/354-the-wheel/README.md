# Study 354 — The Wheel 🎡

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the Wheel beat just holding SPY ("income")? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | The monthly edge over buy-and-hold **flips sign by regime**: +6.2%/yr (*t* = 3.1) in 1993–2009, **−1.2%/yr (*t* = −0.6)** since 2010; full-sample *t* = **1.73**, under the bar. At a *fair* implied vol the control shows the wheel mechanic **loses** to SPY — the only edge is the variance risk premium, not the wheel. |
| **Tradability** — is it a harvestable money machine? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Gorgeous gross Sharpe (**1.32** vs SPY 0.60) but the edge dies at **~25 bp/option** of cost (*t* → −0.35), with skew **−1.87** and worst-5 months summing to **−51%**. One crash hurts; survivable only sized and tail-aware. |
| **"Free / market-neutral income"?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | It keeps **69%** of up months and rides **36%** of down months — a directional short-vol payoff with the right tail sold off, not income, not neutral. |

> **In one sentence:** the retail "Wheel" — sell ATM puts, then ATM covered calls, forever for "income" — is **short volatility in a costume**: priced transparently from the VIX it looks great gross (11.6%/yr, Sharpe 1.32), but that edge is just the variance risk premium, it flips negative in the modern bull market and dies to ~25 bp of option cost, and its 69%-up / 36%-down capture with −1.87 skew is a packaged covered-call exposure, not free money.

## What we tested

The viral options-income "Wheel": **sell a cash-secured at-the-money put** on SPY each month for premium; if assigned, **sell at-the-money covered calls** until called away; repeat "forever" for income, supposedly market-neutral. We model each month's one-month ATM option with **Black-Scholes**, using **VIX/100 as the implied vol** (a transparent, clearly-labelled model — no live option chain, and one that *flatters the seller* since implied runs above realised), over **401 months** of real SPY (1993–2026). We race the Wheel against **buy-and-hold SPY** and a **cash** baseline, test the per-month return edge with a paired *t*, and decompose the up/down **capture** — which exposes the short-vol payoff (truncated upside, near-full downside). A synthetic world priced at its *own* vol proves the mechanic has no edge of its own. (Same short-vol family as [Study 62 — Premium-Seller](../../62-premium-seller/) and [Study 63 — Free-Fall](../../63-free-fall/).)

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why steady premium isn't free income, the up/down capture that unmasks the costume, and how the edge dies to costs — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the Black-Scholes ATM premium, the payoff identity (wheel = B&H − upside sold + premium), the paired-*t* regime flip, the −1.87 skew, and a fair-price positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run (as-of 2026-06-18): [docs/results.md](docs/results.md) — recompute with [examples/verify.py](examples/verify.py) (`--fetch` to download SPY+VIX).

---

*Engine: [`the_wheel/`](the_wheel/). **Not investment advice** — research & education; VIX-as-IV is a transparent proxy that flatters the seller, so a real Wheel is worse than modelled. See [LICENSE](../../LICENSE).*
