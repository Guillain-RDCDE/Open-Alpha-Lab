# Study 599 — Tax-Loss-Harvesting 🧾

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does systematic TLH add after-tax alpha? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | Across **94** overlapping 10-year lump-sum cohorts on SPY (total return, lot-level engine, 35/15 rates), the mechanical after-tax alpha is **+46.0 bps/yr** with **HAC *t* = +2.88** — robust to the harvest threshold (*t* = 2.88–2.94) and the rate pair (*t* = 2.42–2.78). The 0/0-rate myth-check run banks **−0.4 bps/yr** (pure costs): the delta is **tax arithmetic**, exactly as claimed. But the advertised **0.3–1%/yr** only holds at the **top** bracket on the *mean* cohort — the **median** cohort gets **+23.5 bps/yr** and a 33-year contributor nets **+13 bps/yr**. |
| **Tradability** — can you pocket it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | The arithmetic survives DIY costs trivially (2 bps ETF legs) with unbounded capacity — but it is **front-loaded** (year-1 harvest **6.6%** of the stake → **0.17%/yr** by years 7–10 as basis locks up), **path-dependent** (**24 of 94** cohorts never harvest at all; bear years yield **2.2%** vs **0.16%** calm, Welch *t* = +2.72), needs external **short-term gains** to absorb losses at 35%, and a typical 0.25%/yr robo fee **exceeds the median cohort's 23 bps**. Real, but decayed and regime-dependent — not INVESTABLE. |
| **Who gets ZERO?** | ![Confirmed](https://img.shields.io/badge/Who_gets_zero%3F-Confirmed-8b949e?style=flat-square) | The 0%-bracket retiree/donor banks **−0.4 bps/yr** (pure trading costs) — step-up at death does **not** rescue it (still −0.4). At equal rates 15/15 (deferral only) the alpha shrinks to **+5.4 bps/yr**; the step-up **helps only the high bracket** (+13.9 vs +13.1). The "alpha" is a **private tax-rate arbitrage** — it scales with *your* marginal rates, and at rate zero it is exactly nothing minus costs. |

> **In one sentence:** tax-loss harvesting is real arithmetic, not market alpha — a lot-level SPY↔twin engine banks **+46 bps/yr** on the mean 10-year cohort at top-bracket rates (HAC *t* = +2.88) and exactly **−0.4 bps/yr** when rates are zero — but the harvest is front-loaded and bear-market-fed (a quarter of cohorts get *nothing*), the median cohort's 23 bps is **less than a robo fee**, and the low-bracket investor's share is **zero**.

## What we tested

The advisor/robo claim at full strength: *"systematically harvesting losses into a twin fund adds **0.3–1%/yr** of after-tax alpha."* We run a daily-check, **lot-level** TLH programme on SPY total-return (1993→2026): any lot below basis by >1% and wash-sale-safe (≥31 days held) is sold at the **next close** (one execution lag) and swapped into a twin fund (SPY↔IVV, daily-return correlation 0.993, mean gap ~0 bps/yr — exposure economically unchanged, so the **entire** delta is tax timing), losses classified ST/LT, tax savings **reinvested**, both books liquidated at horizon paying per-lot capital-gains tax, 2 bps one-way costs per leg. The Signal axis is the mean after-tax alpha over 94 overlapping 10-year quarterly cohorts (Newey-West lag 40); the rate grid (35/15, 24/15, 15/15, 0/0, ± step-up) and the harvest-yield decay curve do the rest. A 20-seed synthetic GBM control proves the machinery: alpha rises with the planted volatility knob and sits at ~0 when rates are 0/0. Assumptions named in [docs/results.md](docs/results.md): losses fully usable (no $3k cap), twins never wash. As-of **2026-06-30**.

**Named siblings** (household-finance folklore family): [101 — Slow-and-Steady](../101-slow-and-steady/) (DCA timing), [102 — Free-Rebalance](../102-free-rebalance/) (rebalancing bonus), [172 — Hundred-Minus-Age](../172-hundred-minus-age/) (glidepaths), [173 — Four-Percent-Rule](../173-four-percent-rule/) (withdrawals). This one is the family's **pure tax-arithmetic** member — no timing claim at all.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "harvesting a loss" actually is, why swapping SPY for its twin keeps you invested, where the money really comes from (your own tax rates), why the machine goes quiet after a few years — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the lot-level engine, cohort HAC *t*, the rate grid and step-up variants, harvest-yield path dependence (Welch *t*), basis-lock-up decay, threshold robustness, and the 20-seed synthetic volatility-knob control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`tax_loss_harvesting/`](tax_loss_harvesting/). The tested rule is daily-check, lot-level loss harvesting into an economically identical twin (wash-sale-safe by two-twin routing — the SPY↔IVV "not substantially identical" reading is industry practice, never formally blessed by the IRS, a named implementation risk). **Not investment advice, not tax advice** — research & education. See [LICENSE](../../LICENSE).*
