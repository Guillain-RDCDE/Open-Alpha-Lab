# Study 102 — Free-Rebalance ♻️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The realised bonus over a drift portfolio is **−0.59 pts/yr** (annual) on 50/50 SPY/TLT, and the daily bonus doesn't clear the bar (HAC *t* = **−1.19**, bootstrap 95% CI **[−0.66, +0.13] bps/day** straddles zero). It flips sign by regime: **+1.27 pts/yr** (2002–13) → **−1.74 pts/yr** (2014–26). |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | As a *"free lunch that adds return"* it's a mirage: rebalancing **trailed buy-and-hold by ~0.6 pts/yr** (8.22% vs 8.82% CAGR, net of 10 bps/rebalance) and Sharpe was a dead heat (**0.875 vs 0.874**). The textbook **+0.90 pts/yr** "diversification return" is measured vs the *weighted average of assets*, not vs drift. |
| **Controls risk?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | Real risk control: volatility **9.6% vs 10.3%**, and the book stays pinned at the target mix instead of drifting stock-heavy. That — not a return bonus — is what rebalancing actually buys. |

> **In one sentence:** rebalancing genuinely **controls your risk** (lower vol, a disciplined weight mix), but the famed "rebalancing bonus" is **not** a guaranteed free lunch — on real SPY/TLT it earned ~0.6 points/year *less* than just holding the basket, the Sharpe was a wash, and the bonus flips negative whenever one asset trends.

## What we tested

The most-loved "free lunch" in portfolio construction, stated at full strength: *"rebalancing a fixed-weight portfolio mechanically **adds** return on top of the risk control — the **rebalancing bonus** (volatility harvesting / diversification return / Shannon's Demon)."* We take it literally — a 50/50 SPY/TLT book (total return) **rebalanced** annually and quarterly (paying **10 bps one-way** per rebalance) vs the same initial weights left to **drift** — and decompose the gap with the **Booth-Fama / Willenbrock** diversification-return identity to separate the real variance benefit from the artefact. We pin it against the **drift** portfolio (the honest benchmark), put a **block-bootstrap CI** on the daily bonus, and split it by sub-period. Two synthetic controls check the harness reads the *sign* right: a mean-reverting equal-drift pair (bonus **positive**, Shannon's Demon) and a pair where one asset **trends** (bonus **negative**).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the "free lunch" pitch, rebalanced-vs-drift equity curves, the bonus that came out *negative*, why it flips by regime, what rebalancing really buys you |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the Booth-Fama diversification-return identity, HAC *t* + block-bootstrap CI on the daily bonus, the sub-period sign flip, the synthetic sign controls |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`free_rebalance/`](free_rebalance/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
