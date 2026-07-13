# Study 768 — Charm-Decay ⏳

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Pre-OpEx charm-window drift −0.32 bps/day vs baseline (HAC *t* = −0.13); rally-then-fade asymmetry +1.05 bps (*t* = +0.31). Nothing clears \|t\| = 2. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Long-the-charm-week overlay Sharpe 0.62 = buy-and-hold 0.65 (pure beta); the full long/short "rally then fade" makes −0.4%/yr — below cash, before a basis point of cost. |
| **Charm rally?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | A calendar-randomisation placebo puts the real OpEx anchor at the **94th percentile** of arbitrary "fake OpEx" weeks (empirical p = 0.94) — the week is not special. |

> **In one sentence:** charm is a real Greek and dealers really re-hedge it, but the "rally into monthly OpEx, fade after" drift is statistically indistinguishable from a randomly chosen week of SPY — no signal, no trade.

## What we tested

Options-flow desks (SpotGamma, Menthor Q, vol-Twitter) popularised the *charm* story: in the last week before monthly expiration the delta of the dealer options book bleeds with time (charm = ∂Δ/∂t), forcing systematic hedging that pushes SPY **up into OpEx** and lets it **give back after**. We turned that into a directional window/event study on SPY daily bars 1993–2026 (n = 8,418 days, 408 monthly expiries) — pre- and post-OpEx drift, the rally-then-fade asymmetry, a quarterly-only restriction, a pre/post-2012 break, and a **calendar-randomisation placebo** that slides the anchor to fake OpEx dates. HAC *t*-stats throughout, costs charged, capacity noted.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story, the null drift, the placebo that buries it, the tradability check in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stats, the calendar-randomisation null, quarterly/pre-post-2012 splits, synthetic positive control, cost sweep |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`charm_decay/`](charm_decay/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
