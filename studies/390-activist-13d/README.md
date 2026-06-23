# Study 390 — Activist-13D 🦈

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a stock drift up after a 13D? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The folklore points the right way — the announcement **pop is +1.2%** and the post-filing drift is **positive at 1/3/6 months** (+1.7% / +0.9% / +3.8% excess of SPY) — but **nothing clears t ≥ 2** (best **Welch *t* = 1.11**, placebo *p* = 0.18), it's **non-monotone** across horizons, and **dropping three mega-caps collapses it** (*t* = 0.60). A positive whisper on an **outcome-selected** basket, not an edge. |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Costs are negligible (10 bps shaves ~0.1pp off a multi-month hold), but the only buy-able leg — enter **one day after** the filing — returns an excess **inside the noise**, carried by a handful of names, on a rare event. **Not a NAV-scale strategy.** |
| **Free lunch?** | ![Busted](https://img.shields.io/badge/Free_lunch%3F-Busted-8b949e?style=flat-square) | A **selection illusion**: remember Icahn/Apple and Elliott/Salesforce, forget the duds, and a basket of legends *looks* like a free ride. On the leg you can actually trade, the average drift is **statistically indistinguishable from buying the same names on random dates**. |

> **In one sentence:** the "buy the 13D and ride the activist" trade is what survivorship looks like — on a hardcoded basket of 25 *famous* activist campaigns (the easiest possible test, tilted toward memorable wins) the announcement pop and the post-filing drift are both positive but never significant (best t = 1.11, placebo p ≈ 0.18), the modest edge is carried by three mega-caps, and the only leg you can buy after the news is public is indistinguishable from random dates on the same names.

## What we tested

True EDGAR coverage of *every* 13D isn't clean on a free feed, so we work from a **transparent, hardcoded table of 25 famous activist campaigns** (Icahn, Elliott, Third Point, Pershing Square, Trian, ValueAct, Starboard) — the kind of headline filing the folklore is built on, and a basket **selected on outcome**, which we name on the Signal axis as the bias that runs *in favour* of the claim. For each campaign we pull the target's and SPY's daily adjusted closes and split the effect into the **announcement pop** (the day-0 return — real news, but uncapturable unless you knew the filing was coming) and the **post-announcement drift** you can actually buy: enter the close **one day after** the announcement (no look-ahead), hold 1 / 3 / 6 months, measured in **excess of SPY**, with a Welch *t*, a 20,000-draw placebo null drawn from random dates **on the same targets**, and a 10-bps round-trip cost. A deterministic synthetic control with *injected* events confirms the engine is faithful **and** that ~25 events can't reach significance unless the planted drift is implausibly large.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "buy the 13D" feels like free money, why the pop isn't yours to take, and why a basket of famous wins can't be a strategy — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the pop/drift split, excess-of-SPY drift at 1/3/6m, a Welch *t* + a same-target placebo null, costs, lag and mega-cap robustness, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`activist_13d/`](activist_13d/). The event set is an explicit **hardcoded, outcome-selected** basket of famous campaigns, not a full 13D panel. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
