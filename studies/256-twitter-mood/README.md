# Study 256 -- Twitter-Mood

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Lag-1 lead-lag slope HAC *t* = **+0.90**; no lag in the 1-5 sweep clears *t* >= 2 (Bonferroni bar ~2.58); permutation *p* = **0.35**; directional hit-rate **51.3%** vs a 50.8% unconditional up-rate (+0.5 pts). n = 207 business days is far too small to detect a 1-3 day effect, and the mood tape is a curated reconstruction, not the proprietary GPOMS feed. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The long-only Calm rule is flat after costs (*t* = 0.14). The long-short variant (+48%/yr net) only "works" because the 2008 window is the crash -- it goes net-short into a -43%/yr panic, does not clear *t* >= 2 (*t* = 1.25), and would not generalize. |
| **Curated proxy** | ![Curated-proxy](https://img.shields.io/badge/Curated--proxy-8b949e?style=flat-square) | The "Calm" series is a stylized hardcoded reconstruction pinned to documented 2008 public-mood moments -- the original GPOMS/OpinionFinder feed was never released. |

> **In one sentence:** Bollen's "Twitter mood predicts the stock market" (87% accuracy!) does not survive a clean out-of-sample lead-lag test -- on the real S&P tape over the original window the slope is *t* = 0.90, the permutation null swallows it, and the only profitable trading variant is a 2008-crash artefact, not a mood signal.

## The claim

> *Does aggregate Twitter mood predict tomorrow's market (Bollen)?*

## What we tested

The Bollen, Mao & Zeng (2011) recipe: a daily aggregate "Calm" mood index leads
the market 3-4 days out. We join a curated daily Calm index (a stylized
reconstruction of the GPOMS feed -- the original was never public) with real
^GSPC daily returns over the original 2008 window, then (a) regress next-day
return on lagged Calm with a HAC *t*-stat, (b) sweep lags 1-5 -- the Granger-lag
multiple-comparisons trap -- (c) run a 2,000-draw time-shuffle permutation null,
(d) compute the directional hit-rate against the *unconditional* up-rate (the
honest base rate), and (e) trade a gross/net long-short rule with a one-day
execution lag and costs on NAV (shorts pay borrow). A deterministic synthetic
positive control confirms the engine recovers a planted lead-lag (*t* = 3.71)
and reads ~zero on the null.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the "87% accuracy" claim, why the 2008 window flatters any net-short rule, the base-rate trap, the lead-lag in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC lead-lag regression, the lag-sweep / Bonferroni trap, the permutation null, gross/net trading rule, the n=207 power problem, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`twitter_mood/`](twitter_mood/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
