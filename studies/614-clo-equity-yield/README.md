# Study 614 — CLO-Equity-Yield 🏭

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the 15% machine actually pay? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | Yes — more than advertised: ECC distributed **+17.22%/yr (HAC *t* = +15.33)** and OXLC **+19.19%/yr (*t* = +15.35)** of hard cash for 11.7 / 15.4 years, and the erosion financing it is on the same tape (price legs **−13.5% / −14.4%/yr**). What does *not* exist is a surplus: the total-return spread vs plain HYG is **HAC *t* = +0.13 / +0.34** — zero. Survivorship named: ECC/OXLC are the category's two survivors (no index fund exists). |
| **Tradability** — is it the income sleeve it's sold as? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No. Harvesting the 17-19% coupon left TR CAGR at **+1.75% / +2.52%/yr** — excess over bills *t* = +0.43 / +0.81, i.e. *cash* — at 25-32% vol and **−59% / −61%** drawdowns, with no alpha vs a passive HYG+SPY mix (*t* = −0.33 / −0.66) whose copy compounded 5-8 pp/yr faster at half the drawdown. The funds' own 9-13%/yr fee-and-leverage drag is where the CLO cash flows went. |
| **Return OF capital, or ON capital?** | ![Confirmed](https://img.shields.io/badge/Return_OF_capital%3F-Confirmed-8b949e?style=flat-square) | Mostly OF: **78.1%** (ECC) / **74.9%** (OXLC) of the distribution stream was offset one-for-one by price erosion. $100 of capital → **$18.5 / $9.1** price-only while the funds paid "income" the whole way down — the market-price mirror of their own 19(a) return-of-capital notices. |

> **In one sentence:** the CLO-equity 15% machine truly pays — 17-19%/yr of cash at HAC *t* ≈ 15, the biggest measured coupon in the desk's packaged-carry family — but ~three quarters of it was your own capital handed back (price legs −13 to −14%/yr, $100 → $9-19), the total return over a decade-plus is statistically T-bills at −60%-drawdown risk, and the surplus over one-click HYG is exactly zero (*t* = 0.13 / 0.34).

## What we tested

The pitch, steelmanned: *"ECC/OXLC pay ~15% from CLO equity — the levered first-loss tranche of leveraged-loan pools throws off huge cash flows and the funds hand them to you monthly; the income is real."* We decompose both listed CLO-equity CEFs on a double yfinance tape — total-return *and* price-only — so the monthly difference isolates the **distribution return** from **price/NAV erosion**, each with a Newey-West HAC *t*. The headline test is the **total-return spread vs plain HYG** (the one-click credit alternative in the same account); the decisive tradability test regresses each fund's excess return on **HYG** and **SPY** excess returns (NW-HAC alpha vs a passive credit/equity mix). The third axis is the classic: pure returns arithmetic splits the payout into return *ON* vs return *OF* capital. Crisis autopsies (2015-16, Q4-2018, COVID, 2022) show the first-loss profile; a deterministic synthetic world with planted `carry`/`alpha` knobs proves the machinery is faithful. **Sibling of [340 — Bank-Loans](../340-bank-loans/), [341 — MLP-Pipelines](../341-mlp-pipelines/), [342 — BDC-Yield](../342-bdc-yield/) and [611 — mREIT-Carry](../611-mreit-carry/)** — same packaged-carry family, and its most extreme member: the coupon here is the *residual of a first-loss tranche*, double-levered by the wrapper, with the family's biggest yield and starkest capital-consumption share. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a CLO equity tranche actually is, the $100 → $9.10 arithmetic, why a fund can pay 19% a year while making you nothing, and what "return of capital" means in plain words |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the TR-vs-price-only distribution decomposition (HAC *t*), the TR spread vs HYG, the NW-HAC alpha vs the HYG+SPY benchmark, lag/subperiod robustness, crisis table, synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run (ECC 2014-11 → 2026-06, fp `366d9cf7d356`): [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py).

---

*Engine: [`clo_equity_yield/`](clo_equity_yield/). The signal is the distribution-return component (TR minus price-only) and the HAC t of the TR spread vs HYG; the myth-check is the return-OF-capital split. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
