# Study 600 — Asset-Location 🏦

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does bonds-in-the-IRA add real after-tax return? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | Across **122** overlapping 30-year cohorts (Shiller, 1872–2022), a 60/40 household that fills the IRA with **bonds** beats the one that fills it with **stocks** by **+17.8 bps/yr** of after-tax return — **HAC *t* (lag 29) = +3.55**, win rate **86.9%** — and beats the pro-rata default by **+8.6 bps/yr** (*t* = +3.11). Compounds to **+5.2% terminal wealth** at identical pre-tax risk; positive at **every realistic 2026 US bracket pair** (+11 to +67 bps/yr). Index-level tape — no survivorship. |
| **Tradability** — can you actually pocket it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | The choice is genuinely **free** (differential rebalancing cost **+0.013 bps/yr**, unbounded capacity) — **but only if your tax-advantaged account is a *traditional* IRA**, the vehicle the study (and Dammon-Spatt-Zhang) models. Point the *identical* rule at a **Roth** IRA — a mainstream retail wrapper — and it **reverses**: bonds-in-Roth *loses* to stocks-in-Roth by **−28.6 bps/yr** (HAC *t* = **−3.15**, win **10.7%**), because a tax-free wrapper should shelter your highest-growth asset, not your bonds. The reversal deepens with equity turnover (−43.9 bps/yr at 100%) and holds at low brackets (−30.8 bps/yr at 22/15). So "can a *plain retail household* pocket it?" depends on **which account they hold** — the sign flips for a very common one — and the study never tested Roth. Real and free for the traditional-IRA household it models; a trap if mis-applied to a Roth. |
| **"When does it flip?"** | ![Confirmed](https://img.shields.io/badge/Does_it_flip%3F-Confirmed-8b949e?style=flat-square) | It flips on **two** axes. (1) **Equity turnover**: at **100%** (an active fund distributing its gains) the long-tape delta crosses zero (**−0.3 bps/yr**, *t* = −0.07; **−3.7 bps/yr** in the lowest-yield cohorts), statistically zero already at ≥50% — Shoven-Sialm's reversal, confirmed on the tape. (2) **Account type**: swap the traditional IRA for a **Roth** and the whole rule inverts (**−28.6 bps/yr**, *t* = −3.15) — the bigger, more common flip, and the one the headline scope quietly assumes away. Low yields only shrink the prize (Q1 +12.4 vs Q4 +34.3 bps/yr); they don't flip it alone. |

> **In one sentence:** the textbook asset-location rule survives an honest 150-year test **for a traditional IRA** — bonds in the tax-*deferred* account adds **+17.8 bps/yr** after tax (HAC *t* = 3.55) for literally zero cost, compounding to ~5% extra terminal wealth — a real, if small, free lunch; but the *tradability* is **Fragile**, because the identical rule pointed at a **Roth** IRA reverses to a **−28.6 bps/yr** penalty (*t* = −3.15), so "put bonds in the IRA" is only safe advice once you say *which* IRA.

## What we tested

The claim (Dammon-Spatt-Zhang 2004; Bogleheads folklore): *"putting bonds in the tax-deferred account and stocks in taxable adds real after-tax return for free."* We simulate a 60/40 household with a 50/50 taxable/traditional-IRA split, one calendar year at a time, under three location policies holding the pre-tax 60/40 fixed — **bonds-in-IRA**, **stocks-in-IRA**, **pro-rata** — with a full tax-drag model: qualified dividends at the LTCG rate, bond coupons at the ordinary rate, 10% equity-fund turnover realised at LTCG (average-cost basis, loss carryforward), IRA taxed once at withdrawal, taxable settling terminal gains. One documented lag: year *t*'s coupon is the **prior-December** 10y yield. The Signal axis runs every overlapping 30-year cohort of the Shiller tape (1872–2022) and puts a **Newey-West *t* (lag 29)** on the mean after-tax delta; sensitivity runs an 8-cell tax-rate grid and a delta-on-yield regression (+5.6 bps/yr per 1pp of yield). The third axis sweeps equity turnover until the rule flips, and swaps the traditional IRA for a **Roth** — where the rule reverses outright (−28.6 bps/yr, *t* = −3.15), the study's honest scope limit. A zero-tax-rate null (delta must be exactly 0) and a 20-seed planted-coupon control prove the machinery. As-of **2026-07-03**, complete calendar years only.

**Named siblings** (personal-finance arithmetic family): [101 — Slow-and-Steady](../101-slow-and-steady/) (DCA timing), [156 — Martingale](../156-martingale/) (sizing arithmetic), [172 — Hundred-Minus-Age](../172-hundred-minus-age/) (glidepaths), [173 — Four-Percent-Rule](../173-four-percent-rule/) (withdrawals), [599 — Tax-Loss-Harvesting](../599-tax-loss-harvesting/) (the other tax-alpha claim). This study is the family's **placement** question — same funds, same allocation, different account.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the taxman treats a bond coupon worse than a stock gain, what "asset location" means, how much the free lunch is actually worth — and the one household it backfires on — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the household simulator, HAC *t* on 122 overlapping cohorts, the tax-rate grid, the delta-on-yield regression, the turnover flip, the modern SPY/IEF cohort, and the zero-tax null / planted-coupon control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`asset_location/`](asset_location/). The tested rule is the account-placement policy (`bonds_ira` vs `stocks_ira` vs `pro_rata`) at fixed pre-tax 60/40; the myth-check knob is equity-fund turnover. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
