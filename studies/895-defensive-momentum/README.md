# Study 895 — Defensive Momentum 🛟

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — a higher Sharpe *and* shallower momentum-crash drawdowns than MTUM alone? | ![Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square) | The vol / full-window-drawdown reduction is **real & mechanical** (13.1% vs 16.3% vol; −22.2% vs −30.2% maxDD). But the headline **higher Sharpe is refuted**: 50/50 excess-Sharpe advantage over MTUM = **−0.016**, bootstrap CI **[−0.14, +0.16]** straddles zero, sign **flips across eras**, and the only robust component is a **significant return give-up** (−26 bps/mo, *t* −2.21). Crash protection is inconsistent — min-vol **failed in the 2020 COVID crash** (blend −18.5% vs MTUM −17.9%). The 2009 momentum crash is **out of sample** (MTUM ETF born 2013). |
| **Tradability** — is any of it bankable? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Nothing to bank. Costs are trivial (≈1%/mo turnover, 3 bps/side leaves the Sharpe advantage at −0.017) — so this is a **no-edge** story, not a cost story. The blend's whole "benefit" is a **beta / volatility dial**: hold less momentum → crash less *and* earn less → same Sharpe. "Momentum without the crashes" is really "**less momentum — less crash, less return**." |

> **In one sentence:** blending MTUM with USMV buys a genuinely calmer ride at an almost
> exactly matching cut in return — the excess-of-cash Sharpe is a wash (**−0.016**, CI
> straddles zero), the crash protection is unreliable (min-vol was useless in the fastest
> crash on the tape, 2020), and the free lunch the "defensive momentum" pitch promises simply
> isn't on the menu — **Signal Mixed, Tradability Mirage**.

## What we tested

A **50/50** monthly-rebalanced blend and an **inverse-trailing-vol** blend of **MTUM**
(momentum) and **USMV** (min-vol), on monthly **total returns** (net of each fund's fee)
since MTUM's 2013-05 inception (**158 months**, as-of 2026-06-30), against each sleeve, QUAL
and SPY. Everything is **excess-of-cash** (minus BIL). Per blend: the excess-vs-excess Sharpe
race vs MTUM, the **Newey-West** *t* on the monthly return difference, a moving-block
bootstrap CI on the Sharpe advantage, max drawdown + a calendar-year table + the drawdown
suffered in named crash windows (2020 COVID, 2022 bear, 2018 Q4), an era cut, and a costed
net series (one-way cost × realized turnover; the blend is long-only, no borrow). Inverse-vol
weights carry one documented month of lag. A deterministic synthetic world with a planted
crash/diversification edge proves the machinery. **Short-history caveat named on the Signal
axis** (MTUM's 2013 inception drops the canonical 2008-09 momentum crash). **Dedup:**
[508-momentum-crashes](../508-momentum-crashes/) grades the crash *mechanism*,
[330-low-volatility-anomaly](../330-low-volatility-anomaly/) the low-vol anomaly,
[601-factor-etf-live-test](../601-factor-etf-live-test/) the *single* live wrappers, and
[237-residual-momentum](../237-residual-momentum/) a *different* crash fix — this study is
the **blend** question those leave open.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "momentum without the crashes" promises, why the blend rides the straight line between its two sleeves, why lower drawdowns came with a matching cut in return, and why min-vol was no help in the 2020 crash — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the excess-vs-excess Sharpe race with NW *t* and a bootstrap CI, the crash-window drawdown grid, the era cut, the costed net series, and the planted-edge synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`def_momentum/`](def_momentum/). The audited unit is a long-only blend of two live
ETFs, net of their own fees; the only lagged signal (inverse-vol weights) carries one
documented month of lag. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
