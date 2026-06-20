# Study 323 — BTC-Halving

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The post-halving quarter *does* carry the strongest return (**+31.9 bps/day**, naive HAC *t* = **+3.45**) — but Yahoo covers only **three** halvings, and the per-cycle means **+36.5 / +52.1 / +7.7 bps/day** are *decaying*. A daily *t* on three clustered cycles overstates the case: suggestive, not certifiable. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A halving-timing rule (long the bull half of each cycle) earns **+35%/yr** vs buy-and-hold's **+52%/yr** — excess HAC *t* = **−0.92**. It *underperforms doing nothing*: the "cycle edge" was beta minus the days it sat in cash. |
| **Does the halving print the top & bottom?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Realised highs land at **+1296 / +525 / +1402 / +534** days after each halving; lows at **+777 / +24 / +0 / +139**. No cluster — each cycle peaks and troughs at a different phase. The clean cycle chart is a line drawn through four points in hindsight. |

> **In one sentence:** the halving cycle is the grandest chart in crypto and the post-halving glow is faintly real on three cycles — but it is fading, it does not reliably mark the top or the bottom, and a calendar-timed version *loses to simply holding BTC*.

## What we tested

Crypto's most durable folklore: Bitcoin moves in a clean **four-year cycle locked to the mining halving** — bottoming a little before each halving, topping ~12-18 months after, then crashing into the next bottom. Because the halving schedule is fixed *years in advance* (PlanB's stock-to-flow essays and a thousand "cycle" charts lean on it), the calendar alone is supposed to have timed the tops and bottoms. We hardcode the four halving dates (2012-11-28, 2016-07-09, 2020-05-11, 2024-04-20), measure where each cycle's high and low actually fell relative to its halving, slice forward returns by cycle phase, and race a halving-timing rule against buy-and-hold on Yahoo `BTC-USD` daily — flagging the central limit loudly: Yahoo's history *starts 2014*, so a "four-cycle" study is really **two-to-three cycles**. A deterministic synthetic tape with a tunable halving-locked cycle is the positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the cycle story, where the tops/bottoms *actually* fell, and why timing it loses to holding |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | cycle-extrema scatter, phase-bucket HAC *t*, the three-cluster problem, timing vs B&H, synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`btc_halving/`](btc_halving/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
