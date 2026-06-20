# Study 324 — Bitcoin-Treasury 🟧

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | In the treasury era (≥ 2020-08), MSTR is **~1.1× Bitcoin** with **50% of its daily variance explained by BTC**; the residual alpha is **+39%/yr** on paper but **HAC *t* = +1.08** — below the inference bar. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | MSTR's edge over its *own levered-BTC replica* is **+11 bps/day, bootstrap CI [−9, +31]** (straddles zero) — and it took a **worse drawdown (−89% vs −81%)**. The NAV premium can't be timed (*t* = −0.88). |
| **Just leveraged Bitcoin?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | Beta ≈ 1.1, R² = 50%, no robust residual alpha, no harvestable premium — the "treasury" story is a leverage story. |

> **In one sentence:** MicroStrategy really is a leveraged Bitcoin bet — and that's *all* it is on the tape: the extra return over plain leverage isn't statistically distinguishable from zero, the premium you pay for it just adds risk you could rent more cheaply elsewhere.

## What we tested

Michael Saylor reframed MicroStrategy as a **Bitcoin treasury** on 2020-08-11, and the
market loves it: MSTR is marketed as "intelligent leverage" on Bitcoin *plus* an
operating-company premium, with a whole cottage industry trading the mNAV premium as if it
were a coupon. We take that literally — decompose MSTR's daily returns onto BTC-USD (the
slope is the leverage, the intercept the alpha, with a Newey-West *t*), race holding MSTR
against a **synthetic levered-BTC replica** at MSTR's own beta (net of borrow,
excess-of-cash Sharpe), and test whether a contrarian **NAV-premium-timing** overlay beats
just holding. A deterministic synthetic tape — BTC random walk plus an MSTR built *as*
levered beta with a tunable planted alpha and a return-free premium wobble — is the
positive control that proves the harness can find an edge when one is really there.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the treasury pivot, why "MSTR beat Bitcoin" is a leverage trick, and what the premium really costs you |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the BTC beta + HAC-*t* alpha, the levered-replica race with bootstrap CI, the premium-timing null, the synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`bitcoin_treasury/`](bitcoin_treasury/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
