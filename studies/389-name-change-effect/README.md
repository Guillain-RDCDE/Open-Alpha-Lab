# Study 389 — Name-Change-Effect 🪧

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a theme-chasing rename pop, then fade? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The dot-com literature makes the *pop* real in 1998–99, but on a ~21-event survivor tape the pop is just **+1.2%** (Welch *t* = **0.51**, placebo *p* = **0.50**) and the **fade runs the wrong way — +16%, not negative** (*t* = **1.16**). A faint, insignificant, sign-wrong point estimate, not an edge. **Survivorship** named here: the worst give-backs *delisted*. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The believers' buy-the-pop / short-the-fade trade is **−15% gross** per event (you lose shorting a fade that never comes) and **−16% net** of four small-cap crossings. Nothing to harvest, at any size. |
| **"Pop then dump"?** | ![Busted](https://img.shields.io/badge/Pop_then_dump%3F-Busted-8b949e?style=flat-square) | A **selection-on-anecdotes** illusion: Long Blockchain and KodakCoin are remembered *because* they were extreme — and they **delisted**, so they never enter a survivor table. Across a representative sample the pop is a coin-flip and the give-back goes the wrong way. |

> **In one sentence:** the legend that a company rebranding toward the hot theme (`.com`, `Blockchain`, `AI`) pops and then gives it all back is built on the few that delisted (Long Blockchain, KodakCoin) — across a transparent table of ~25 real rebrands, the surviving ~21 show only a faint insignificant **+1.2%** pop (*t* = 0.51) and, instead of a give-back, a **+16%** drift *up* over the next quarter (*t* = 1.16), so the believers' pop-and-dump trade loses ~15% before costs and the whole thing is real-as-anecdote, absent-as-law.

## What we tested

We hardcode a **transparent table of ~25 documented theme-chasing rebrands** — dot-com renames of 1998–2000, the 2017–2018 "blockchain" pivots (Riot, Marathon, Long Blockchain), and the 2020–2024 "AI" wave (C3.ai, BigBear.ai, SoundHound). Around each rename we measure the **abnormal (excess-of-SPY) return** on a short **pop** leg (`[+1…+5d]`) and a longer **fade** leg (`[+6…+65d]`), with a one-day entry lag, then judge each leg with a Welch *t*, a placebo null sized to the event count, and a base-rate win-rate. We name the central honesty problem loudly: the loudest give-backs **delisted** and leave no price series, so the surviving tape is biased *against* the believers' fade — a survivor-only fade that comes out *positive* is therefore a conservative refutation. A deterministic synthetic control with an injected pop+give-back confirms the engine is faithful **and** that ~25 events can't reach significance unless the planted effect is implausibly large.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why we only remember Long Blockchain, what a "pop and fade" really is, and why the survivors went *up* instead of giving it back — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | abnormal-return event windows on a rebrand table, pop/fade legs vs zero, a Welch *t* + placebo randomization null, the costed believers' trade, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`name_change_effect/`](name_change_effect/). The rebrand table is hardcoded & transparent; the priced tape is **survivor-biased** (the worst give-backs delisted), named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
