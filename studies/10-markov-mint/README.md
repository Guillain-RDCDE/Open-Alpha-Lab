# Study 10 — Markov-Mint 🎲

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | 🔴 `NONE` | On a provably-fair market the machine's realized directional edge is **−0.68 pp** (HAC *t* = **−0.77**) — a coin flip; the *most* any method could capture is **exactly 0**. The raw Monte-Carlo "edge" is zero-mean noise whose spread **collapses from ~20 pp to ~2 pp as history grows** — the fingerprint of estimation error, not information. |
| **Tradability** — does it survive costs, capacity, scale? | 🔴 `MIRAGE` | Kelly-sized and scored against truth, the bankroll is **incinerated** — **0.0003×** after a 2¢ spread, and still **0.002× at *zero* cost**. The calibration table tops out at **0.958**, handing every richer contract a probability *below its own price* — a mechanical **BUY NO** on **568 / 2,000** markets. Shorting a fair favorite at 98¢ loses ~98% of the time. |
| **"Win every single trade"?** | ⚪ `BUSTED` | Realized win rate **51.6%** — a coin flip. And the only real effect in the whole pipeline, the favorite-longshot bias, nets **−13.6% per trade *even for an oracle with the true probabilities*** once a 2¢ spread is charged. |

> **In one sentence:** run the five-step "Markov chain that wins every trade" on markets whose price is *provably* fair and it finds nothing but Monte-Carlo noise — worse, a hard-coded calibration ceiling makes it reflexively short every strong favorite, so Kelly-sizing the "edge" doesn't break even, it **destroys the bankroll**; and the one genuine effect it leans on, the longshot bias, is too thin to beat a two-cent spread even with perfect information.

## What we tested

A viral X/Twitter thread by Alex (@de1lymoon), *How To Use Markov Chains To Win Every Single Trade + [Quant Framework]* (26 May 2026; ~1.2 M views — see [docs/references.md](docs/references.md)), sells a concrete five-step pipeline to print money on Polymarket: build a price-history transition matrix → Monte-Carlo the resolution → calibrate against the favorite-longshot bias → size with quarter-Kelly → execute as a maker. We port the author's code **verbatim** and feed it 2,000 synthetic binary markets whose price is the exact Bayesian posterior — a **martingale**, so the price is provably the best estimate and **no edge exists by construction** — then score every bet against the true resolution.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the teardown |

Reproduce the headline run with [examples/verify.py](examples/verify.py) → [docs/results.md](docs/results.md) (as-of + fingerprint `585b80af7d53`).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
