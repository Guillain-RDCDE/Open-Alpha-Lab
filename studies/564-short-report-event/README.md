# Study 564 — Short-Report-Event 🐻

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the stock fall and *stay* down after a short report? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | Directionally **yes** — the post-report excess drift is **negative at 1/3/6m** (−1.6% / −11.6% / −14.6% vs SPY), the **median** target is crushed (**−42.7%** at 6m) and the short is right **64–77%** of the time. But the **mean** *t* **never clears −2** (best 3m **t = −1.55**, placebo *p* 0.068) because a few names **squeeze** (Carvana **+318%** excess) and eat the mean. Right sign, high hit-rate, no robust *t* ≥ 2. |
| **Tradability** — can you short it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | The **median** short pays, but the payoff is **asymmetric** — bounded gains, unbounded squeeze losses — which no sizing fixes. A **punitive borrow** on a crashed name, a rare event, and the biggest wins (Nikola/Lordstown/Mullen) **delisted** and are missing from the feed. Net short is a positive *mean* wrapped around an ugly left tail. |
| **"Free short?"** | ![Busted](https://img.shields.io/badge/Free_short%3F-Busted-8b949e?style=flat-square) | The take-downs you *remember* are exactly what **survivorship** looks like — and the ones that worked best went to zero and *delisted*, so a surviving basket both flatters the roster and starves the tape. On the leg you can actually short, one Carvana squeeze undoes eighteen winners. |

> **In one sentence:** the short-side cousin of the [activist-13d](../390-activist-13d/) study — on a hardcoded basket of 32 famous activist short reports (Hindenburg, Muddy Waters, Citron; 22 priced after 10 delisted) the stock *does* fall and stay down (median −42.7% under SPY at 6m, 64–77% hit-rate), but the **mean** short never clears a robust *t* ≥ 2 (best −1.55) because activist shorting is a high-hit-rate bet with an unbounded left tail — one +318% Carvana squeeze eats the whole basket.

## What we tested

There's no free, complete panel of *every* activist short report, so we work from a **transparent,
hardcoded table of 32 famous campaigns** (Hindenburg, Muddy Waters, Citron, Kerrisdale, Spruce
Point, …, 2015–2023) — the kind of headline hit piece the folklore is built on, a basket
**selected on outcome** (bias *for* the claim) whose biggest winners then **delisted** (bias
*against* it) — both named on the Signal axis. For each report we pull the target's and SPY's daily
adjusted closes and split the effect into the **report-day crater** (the day-0 return — real news,
but uncapturable unless you were already short) and the **post-report drift** you can actually
short: enter the close **one day after** the report (no look-ahead), hold 1 / 3 / 6 months, measured
in **excess of SPY**, with a Welch *t*, a 20,000-draw same-target placebo null, a 10-bps round-trip
and a **punitive 800 bps/yr short borrow**. A deterministic, seed-robust synthetic control (25
seeds) with an *injected* negative drift confirms the engine catches a real short edge (*t* → −2.9)
and stays flat at the null. *Distinct from the **long-side** [390 Activist-13D](../390-activist-13d/)
— same event-study shape, opposite sign, plus the short-only pathology: the **squeeze**.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what an activist short report is, why the median stock really does crater, and why one Carvana squeeze turns a great hit-rate into a mediocre average — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the crater/drift split, excess-of-SPY drift at 1/3/6m, mean-vs-median skew, a Welch *t* + same-target placebo null, costs + borrow, lag robustness, and a seed-robust synthetic short-edge control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run (price fp `946a53954ab8`, as-of 2026-06-30): [docs/results.md](docs/results.md).

---

*Engine: [`short_report_event/`](short_report_event/). The event set is an explicit **hardcoded, outcome-selected** basket of famous campaigns, not a full short-report panel. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
