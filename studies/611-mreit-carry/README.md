# Study 611 — mREIT-Carry 🏚️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the leveraged-MBS carry real? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | Yes, spectacularly: the sector fund (REM) harvested **+84.7 bps/mo (+10.65%/yr)** of pure dividend return at **HAC *t* = +18.1** (NLY 13.1%/yr, *t* = 23.0; AGNC 15.6%/yr, *t* = 17.8) — and the erosion financing it is just as real: REM's carry premium vs a duration-matched levered-IEF benchmark is **−87.6 bps/mo at HAC *t* = −2.50**. REM is an index fund, so the headline leg is **not survivor-biased**; the single names are survivors, quoted as colour. |
| **Tradability** — is it the income sleeve it's sold as? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No. 19.1 years of harvesting a 10.65% coupon left $100 at **$80.5** (TR CAGR **−1.13%/yr** — excess over T-bills *t* = +0.18, i.e. *cash*), at **24% vol**, a **−67% drawdown**, and **−10.0%/yr (t = −2.50)** behind a passive levered IEF+SPY mix. A worse bond, a worse stock, a worse carry trade than DIY. |
| **"10-14% free lunch"?** | ![Busted](https://img.shields.io/badge/Free_lunch%3F-Busted-8b949e?style=flat-square) | The coupon is fully financed by NAV erosion: REM's price leg fell **−10.96%/yr** ($100 → **$10.9** price-only). Spending the dividend is spending your principal — with an equity beta of **1.05** and a repo margin call attached (−74.7% GFC, −68.5% COVID). |

> **In one sentence:** the mortgage-REIT 10-14% is a *real*, harvestable carry stream (HAC *t* ≈ 18-23) bolted onto a price leg that eroded −11%/yr — nineteen years of coupons compounded to *less than T-bills* at equity-crash risk, and the packaged carry lost significantly (*t* = −2.50) to a duration-matched levered-Treasuries benchmark anyone could hold instead.

## What we tested

The pitch, steelmanned: *"Mortgage REITs pay 10-14% — leveraged MBS carry. The income is real; harvest the dividend stream and it compounds."* We decompose the sector bellwether **REM** (2007→, GFC included) plus flagships **NLY**/**AGNC** on a double yfinance tape — total-return *and* price-only — so the monthly difference isolates the **dividend return** (the carry) from **NAV erosion**, each with a Newey-West HAC *t*. The decisive test regresses each name's excess return on **IEF** and **SPY** excess returns (the duration-matched levered benchmark a DIY investor could hold): the intercept is the carry *premium*, and for REM it is significantly **negative**. Crisis autopsies (fixed windows: GFC, 2013 taper, COVID, 2022) show the "income sleeve" falling more than equities in every stress. A deterministic synthetic world with planted `carry` / `alpha` knobs proves the machinery is faithful. **Sibling of [341 — MLP-Pipelines](../341-mlp-pipelines/) and [342 — BDC-Yield](../342-bdc-yield/)** (both Real × Mirage): same packaged-carry family, new asset — here the coupon genuinely *is* a carry trade (levered repo net-interest-margin, short liquidity/convexity) and the benchmark is levered Treasuries, not an equity sector. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | where the 10-14% really comes from, the $100 → $80.5 arithmetic, why the "bond substitute" crashed harder than stocks in 2008 *and* 2020 |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the TR-vs-price-only carry decomposition (HAC *t*), the NW-HAC alpha vs the levered-IEF benchmark, lag/subperiod robustness, crisis table, synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run (2007-06 → 2026-06, fp `3b2ec2fc7dd3`): [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py).

---

*Engine: [`mreit_carry/`](mreit_carry/). The signal is the dividend-return component (TR minus price-only) and the NW-HAC carry premium vs an IEF+SPY excess-return benchmark; the myth-check is the free-lunch claim. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
