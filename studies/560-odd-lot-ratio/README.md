# Study 560 — Odd-Lot-Ratio 🪙

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does fading odd-lot buying pay? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | On the modern null tape the fade slope of forward return on the prior-week odd-lot ratio is **+0.032%/z-unit**, Newey–West *t* **+0.50** (the *wrong* sign, nowhere near ≥ 2); placebo *p* **0.62**; panic **+8.1%/yr** *under*-earned euphoria **+9.7%/yr** (spread −1.6%/yr, *t* −0.20). The slope flips sign across windows and only clears \|*t*\|=2 the *wrong* way. **Synthetic-only** (no free odd-lot series) → capped at `WEAK`/`NONE` by house rule. |
| **Tradability** — does the overlay pay? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A weekly contrarian overlay earns **−2.0%/yr gross**, **−2.7%/yr net** (Sharpe −0.11 → −0.15) at **25.9× annual turnover** — wrong sign before costs, worse after the short-euphoria borrow. Nothing to harvest. |
| **Alive post-decimalization?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Decimalization (2001) + algorithmic order-slicing turned most 'odd lots' into machine child-orders, not retail; odd lots weren't even on the SIP until 2013–14. The premise — odd lot = dumb retail money — is dead. |

> **In one sentence:** the odd-lot ratio was a real 'dumb-money' fade when odd lots were mom-and-pop retail, but decimalization and algorithmic order-slicing turned most odd-lot flow into machine child-orders (and the series itself is proprietary, unmeasurable on a retail stack) — so on the modern null tape the fade is statistically zero (HAC *t* +0.50, placebo *p* 0.62), sign-unstable, and loses money net, while the seed-robust synthetic control proves the engine *would* bank the fade if it still existed.

## What we tested

The **odd-lot theory** (Garfield Drew, 1950s): the small odd-lot (< 100 share) trader is chronically
wrong — buying tops, selling bottoms — so **fading** odd-lot buying should pay. A clean,
point-in-time odd-lot ratio series is proprietary (TAQ/consolidated-tape odd-lot flags; odd lots
weren't on the SIP until 2013–14), so — like the desk's [whisky-cask](../275-whisky-cask/),
[lego-returns](../273-lego-returns/) and [sneaker-resale](../276-sneaker-resale/) studies — the real
free data does not exist and the study is **synthetic-only** (capped at `WEAK`/`NONE` on the SIGNAL
axis). We build the textbook contrarian signal (position = −z of the *prior*-week odd-lot ratio, one
execution lag), test the **HAC slope** of next-week return on the ratio, a **regime sort** (panic vs
euphoria terciles), a **label-shuffle placebo** null, a weekly overlay net of costs and a short-leg
borrow, a **four-window** sign-stability sweep, and a **seed-robust synthetic positive control**
(25 seeds) that plants the fade (`fade_alpha > 0`, the old dumb-money world), watches the slope-*t*
cross −2, and stays flat at the modern null (`fade_alpha = 0`). *Distinct from the survey-sentiment
contrarian [257 AAII-Sentiment](../257-aaii-sentiment/): this is the odd-lot **order-flow** gauge and
its story is structural **death**, not weak-but-alive.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what an odd lot is, why it used to mark the wrong side, and why decimalization + machines killed the tell |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the HAC fade slope, the placebo null, the panic-vs-euphoria sort, the four-window sign-flip, costs + borrow, and the seed-robust synthetic positive control |

The reproducible headline run (1,039-week synthetic modern-null tape, `fade_alpha = 0`, fingerprint
`2659c7bf0fd0`, as-of 2026-06-30) is in [docs/results.md](docs/results.md); the offline machinery
proof runs on the deterministic synthetic world in [`odd_lot_ratio/data.py`](odd_lot_ratio/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`odd_lot_ratio/`](odd_lot_ratio/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
