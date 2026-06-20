# Study 318 — Election-Volatility 🗳️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Realized vol barely budges into elections (ratio **1.05**, t = **+0.54**, 66th placebo pctile). Implied vol *does* over-price it (mean VRP **+5.2** vol pts, 7/9 positive) but t = **+1.28**, CI **[−2.8, +11.8]** — below the bar. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The headline short-vol carry (gross t = **+4.51**) is the **always-on** variance premium: random dates earn **+3.80** of the +6.15. Election *excess* = **+2.35** vol pts, t = **+1.72** — and it owns the 2008 tail (**−23** vol pts in one trade). |
| **An election premium, or just the VRP?** | ![MISATTRIBUTED](https://img.shields.io/badge/MISATTRIBUTED-8b949e?style=flat-square) | Strip out the volatility-risk premium you'd earn any random Tuesday and the election-*specific* component doesn't clear significance. |

> **In one sentence:** US elections don't spike *realized* vol, *implied* vol over-prices them only weakly (VRP +5.2, t = +1.28), and a short-vol carry's gross t = +4.51 is almost entirely the everyday variance-risk premium — the election-specific excess (+2.35 vol pts, t = +1.72) can't be told from noise and is paid for with a −23-vol-point 2008 tail.

## What we tested

Believers — backed by [Kelly, Pástor & Veronesi (2016)](https://doi.org/10.1111/jofi.12406) — say political-event uncertainty raises volatility and the option market charges a premium for it, so you can **sell vol into the election and collect the post-vote crush**. We test the three links on the real tape: (1) does *realized* vol spike in the run-up (^GSPC, all 25 elections since 1928), (2) does *implied* vol (^VIX, 9 elections since 1992) over-price the realized vol that follows — a variance-risk premium, and (3) does a short-vol carry timed to the election beat the **always-on** VRP you'd harvest any time. This is the **volatility/event** angle, deliberately distinct from the desk's *return-cycle* election studies [81 (Four-Year-Itch)](../81-four-year-itch/) and [248 (Presidential-Honeymoon)](../248-presidential-honeymoon/).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the "election VIX bump" story in plain language, why realized vol barely moves, the trap of a 90%-win short-vol trade, the 2008 steamroller |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | vol-ratio event study + placebo, the VRP per election, the excess-vs-always-on race, block bootstrap, the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`election_volatility/`](election_volatility/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
