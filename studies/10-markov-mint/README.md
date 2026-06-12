# Study 10 — Markov-Mint 🎲

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | On a provably-fair market the machine's realized directional edge is **−0.45 pp** (HAC *t* = **−0.51**) — a coin flip; the *most* any method could capture is **exactly 0**. The raw Monte-Carlo "edge" is zero-mean noise whose spread **collapses from ~20 pp to ~2 pp as history grows** — the fingerprint of estimation error, not information. Delete the chain and the bet count falls **3×**: the Markov stage manufactures trades, not signal. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Quarter-Kelly on the fair null ends at **0.0003×** the bankroll net of a 1¢ entry half-spread — and **0.0017× even at zero cost**: Kelly-compounding noise is pure variance drag. Given a *real* planted favorite-longshot wedge the pipeline does detect it gross (**+1.83 pp**, *t* = 2.09 — via its borrowed calibration table, not the chain) and **still nets −22% per trade** (bankroll **0.15×**): it trades cost-blind, and the ~1.7¢ wedge is thinner than the 1¢ toll on a large share of its trades. |
| **"Win every single trade"?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Realized win rate **51.5%** — a coin flip. And the bound is brutal: even a *cost-aware oracle that knows the true probabilities* and refuses every toll-dominated market only nets **+2.3% per trade** (HAC *t* = 1.97 — under the desk's bar even with perfect information, on 2,000 markets). |

> **In one sentence:** run the five-step "Markov chain that wins every trade" on markets whose price is *provably* fair and it finds nothing but Monte-Carlo noise — Kelly-sizing that noise destroys the bankroll even before costs; hand it a market with a *real* favorite-longshot wedge planted in it and it detects the edge only through its hard-coded calibration table (the chain still contributes nothing), then loses it all back to the bid/ask, because the genuine wedge is about the size of the toll — so thin that even perfect information barely certifies a profit.

## What we tested

A viral X/Twitter thread by Alex (@de1lymoon), *How To Use Markov Chains To Win Every Single Trade + [Quant Framework]* (26 May 2026; ~1.2 M views — see [docs/references.md](docs/references.md)), sells a concrete five-step pipeline to print money on Polymarket: build a price-history transition matrix → Monte-Carlo the resolution → calibrate against the favorite-longshot bias → size with quarter-Kelly → execute as a maker. We port the author's code **verbatim** and feed it two controlled markets: 2,000 synthetic binaries whose price is the exact Bayesian posterior — a **martingale**, so the price is provably the best estimate and **no edge exists by construction** — and 2,000 with a **planted Thaler-Ziemba wedge** (longshots priced *above* their true probability, favorites *below*; a fair 5.0¢ contract trades at 7.6¢) so we can also measure what a real edge is worth once the spread is charged. Every bet is scored against the true resolution.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the teardown |

Reproduce the headline run with [examples/verify.py](examples/verify.py) → [docs/results.md](docs/results.md) (as-of + fingerprint `6b3d94fff8f1`).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
