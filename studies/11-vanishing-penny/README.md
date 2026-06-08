# Study 11 — Vanishing-Penny 🪙

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the arbitrage actually real? | 🟢 `REAL` | It is risk-free by construction and well-documented: the paper (Saguillo et al. 2025) prices ~\$40M extracted in a year, and on our own real tape **161** genuine ≥3¢ `YES+NO` gaps open across 13 markets (median penny **4¢**). The free money exists. |
| **Tradability** — can a non-co-located trader catch it? | 🔴 `MIRAGE` | Every one of those 161 episodes closes **inside our 1-minute measurement floor** (`frac_below_floor = 100%`, median episode duration **1 min**). Even against a *generous* 1-minute upper bound, a human reacting in 5 min keeps **3%** of the penny, in 30 min **~0%**. The edge belongs to the block, not the browser. |
| **Execution moat?** — is the edge structurally reserved? | ⚪ `CONFIRMED` | The gap's true half-life is **below every resolution we can sample**: the median episode is exactly *one tick* long at 1, 2, 5, 15, 30 *and* 60-minute fidelity. A timescale that hides under any tape you bring is the definition of a moat — won in the ~30 ms before the next block, where retail is the exit liquidity. |

> **In one sentence:** the guaranteed Polymarket penny the viral thread sells you is **real and already gone** — it closes faster than the public tape can even sample, so the \$40M is a same-block infrastructure prize, and the retail "roadmap" is a slower way to provide the fast wallets their exit liquidity (and, at the bottom of the thread, an airdrop funnel).

## What we tested

A [viral thread](https://x.com/robrtcode) (2.9M views) lays out "the exact maths that pulled **\$40,000,000** out of Polymarket — complete roadmap": when `YES` is \$0.62 and `NO` is \$0.33 that is \$0.95, a guaranteed \$0.05 per pair, risk-free. The core is true and checkable — the arbitrage is documented in a real 2025 paper, *Unravelling the Probabilistic Forest* ([arXiv:2508.03474](https://arxiv.org/abs/2508.03474)), and the wallet P&Ls are public on-chain. So the only honest question left is not *whether* the penny is real but **how long it lives** — which we test by measuring the half-life of the gap `g = 1 − (p_yes + p_no)` across 13 real markets.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious.ipynb](notebooks/01_for_the_curious.ipynb)** | the curious | the story + the stakes, plain language |
| **[02_for_the_quants.ipynb](notebooks/02_for_the_quants.ipynb)** | quants | the full method: episodes, two half-life estimators, sweep, capture |

The real run — every fingerprinted, as-of'd table — is in [docs/results.md](docs/results.md); reproduce offline via [examples/run_synthetic_demo.py](examples/run_synthetic_demo.py) and on the real tape via [examples/verify_real.py](examples/verify_real.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
