# Study 594 — Leverage-Rotation-200SMA 🎢

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the 200SMA timing real? | ![Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square) | *Real on the vol channel, none on the return channel.* Below the 200SMA, next-day QQQ vol runs **2.05×** the above-line vol (Welch *t* = **+13.79** on the real tape) — the SMA is a genuine **volatility switch**. But it certifies **no return timing**: HAC *t* = **+1.66** vs QQQ B&H over 26.5 years, and vs 40 exposure-and-switch-matched random timers the seed-averaged Welch *t* = **+0.05**. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Costs are irrelevant (7 switches/yr; 0→10 bps trims CAGR 11.9%→11.1%) — the **path** is the killer: **−95.14%** max drawdown, **18.2 years** underwater for a 2000-top entrant, COVID's V-crash whipsawed QQQ's −6.5% into **−41.5%**, and since 2010 the filter **significantly lagged** just holding TQQQ (HAC *t* = **−2.02**). Regime-conditional leveraged beta whose viral backtest starts after its own funeral. |
| **Does the 2000 dot-com cohort survive it?** | ![Busted](https://img.shields.io/badge/2000_cohort_survives%3F-Busted-8b949e?style=flat-square) | $10,000 at the 2000-03-24 peak → trough **$507** (−95.14%), made whole in **2018**. Better than 3x-forever ($2 trough, never whole) — but a −95% crash is exactly the "3x crash" the plan promised to remove. |

> **In one sentence:** the Reddit-famous "hold TQQQ above the 200-day line" plan rests on one real fact — below the SMA, vol doubles (*t* = +13.8) — but that vol switch buys **no certified return edge** (*t* = +1.66 vs QQQ; +0.05 vs matched random timers), and restoring the 2000-02 regime its 2010+ backtests always omit shows the promise failing at full strength: **−95% drawdown, 18 years underwater**, while post-2010 the filter only *lagged* raw TQQQ (*t* = −2.02).

## What we tested

The specific retail escalation that combines two desk verdicts — [110-faber-timing](../110-faber-timing/)'s 200SMA switch (Real/Fragile) driving [100-melting-ice](../100-melting-ice/)'s 3x daily-reset instrument (Real/Mirage): hold **TQQQ** while QQQ closes above its **200-day SMA**, T-bills (^IRX accrual) below, signal at the close, position next day — one documented lag, costs one-way × NAV per switch (0/2/5/10 bps). TQQQ only exists since 2010, so we synthesise the 3x fund from QQQ's daily total return (`3r − 2·rf − 2.5%/yr`, calibrated once; daily corr **0.9989** vs real TQQQ, residual drag −0.04%/yr) and run 1999→2026 including the dot-com crash the community backtests never show. Inference: HAC *t* on daily return differences, a **40-seed** exposure-and-switch-matched random-timer baseline (averaged Welch *t*), above/below-SMA mean and vol conditioning, SMA-window and cost sweeps, sub-periods (2000-02, GFC, COVID, 2022), and a planted-regime synthetic control proving the harness detects real timing skill when it exists. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the Reddit pitch, what the 200-day line actually switches (volatility, not returns), the 2000 parabolic-top trap and the COVID whipsaw, and why the famous backtests all start in 2010 |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | sim-3x validation vs real TQQQ, HAC *t*s, the matched random-timer test, mean-vs-vol channel decomposition, cost & window sweeps, sub-periods, the 2000-cohort ledger, synthetic null + planted-regime control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`leverage_rotation_200sma/`](leverage_rotation_200sma/). Signal on QQQ's close vs its 200-day SMA; instrument is real TQQQ (2010+) and an audited synthesised 3x before that. Siblings: [110-faber-timing](../110-faber-timing/), [100-melting-ice](../100-melting-ice/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
