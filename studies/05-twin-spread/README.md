# Study 05 — Twin-Spread 👯

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the spread actually revert into a profit? | 🔴 `NONE` | The minimum-distance pairs **don't reconverge enough to pay**: modern-era gross is **−0.48%/mo** (Sharpe **−0.44**, bootstrap CI **[−0.84, −0.03]**, 98% of resamples negative) — *negative even at a literal zero spread*. The full sample is statistically zero (CI [−0.34, +0.15]). |
| **Tradability** — does it survive costs, capacity, scale? | 🔴 `MIRAGE` | There's no edge for costs to kill — and they deepen the loss anyway (monotone in the spread), to **−0.54%/mo net**, **Sharpe −0.44**, a **−85% max drawdown**, with **β≈0** so there isn't even market beta to fall back on. Liquidity is *not* the binding constraint (capacity ~\$52k/leg); the **missing edge** is. |
| **Decay since GGR?** — has the famous edge faded? | ⚪ `CONFIRMED` | Positive years cluster in **1983–2004**; the well-populated modern era is **mostly red** (worst: 2020 −2.3%, 2022 −3.9%, 2023 −2.0% monthly). The only modern green is in **dislocations** (2008 +0.9%/mo, Sharpe 1.30; 2019). **And the obvious modern fixes don't rescue it**: a stop-loss tames the −85% drawdown to −24% but leaves it ~flat-negative; a cointegration gate doesn't help at all. |

> **In one sentence:** run honestly on a tradeable liquid basket, the parameter-free
> pairs-trading rule the tweet celebrates has **no convergence edge left** in the modern
> era — it's significantly negative *before* costs, market-neutral so there's nowhere to
> hide, saddled with an −85% drawdown, and **not revived by the obvious fixes**: a textbook
> the world arbitraged past, leaving the naive follower holding the tail of pairs that
> diverge and never come back.

## What we tested

A [viral thread](https://x.com/MatiasScalbi/status/2063042609816252666) resurfaces the most respectable anomaly in the book: **pairs trading**, the relative-value rule Nunzio Tartaglia's Morgan Stanley desk ran in the 1980s and Gatev, Goetzmann & Rouwenhorst (GGR) published in 1999. Stated at full strength: *find two stocks that move together; when they diverge, short the winner and buy the loser and wait — it made ~1.4% a month at a Sharpe near 0.6 and near-zero beta, and kept paying even after the paper told the world how to do it.* It's the steelman to beat because the rule is **parameter-free** — rank pairs by a single distance, trade a fixed 2σ trigger, close on the cross. We ran exactly that textbook GGR rule over a cached, liquid 174-name US universe (1962–2026), headlining the modern era (2005–2026) where there are finally enough names for genuinely tight pairs.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story + the stakes, plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the full method: formation, the trade, decay, neutrality, costs |

Every headline number is fingerprinted in [docs/results.md](docs/results.md); the three worked rescues (stop-loss, cointegration gate, breadth) are tabled in [docs/extensions.md](docs/extensions.md) and reproducible via [examples/](examples/).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
