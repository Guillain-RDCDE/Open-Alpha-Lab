# Study 620 — A-H Share Premium 🇨🇳

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the same company really cost ~30% more in Shanghai? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | Across 14 dual-listed pairs the FX-adjusted premium averaged **+29.69%** over 2010–2026 (HAC(12) *t* = **5.12**, tape-calibrated Monte-Carlo null **p = 0.018**), positive **139 months in a row** since Stock Connect opened, in **13/14 pairs**, moving as **one factor** (PC1 39.2%) — and the cross-section predicts relative returns at **HAC *t* = +5.09** *on returns* (skip-month 4.84; both halves ≥ 2.36; price-only 5.38; 20-seed random baseline −0.26). **Survivorship** named (still-dual-listed mega-cap panel). Honest caveat: under a pure random-walk null the 16-yr *average alone* can't be told from luck — non-convergence is the claim itself. |
| **Tradability** — can anyone arbitrage it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | There is **no convergence channel**: A/H lines are non-fungible and A-shares are unborrowable for anyone who sees the spread. The profitable half of the fade *requires shorting A*; the executable half (long A via Connect / short H) reads gross *t* = 2.09 but **nets *t* = 1.19** at 25 bps + 1.5% borrow and *t* = **0.95** gross since 2018 — and it bets the premium *widens*. Paper alpha, locked door. |
| **Is the cheap H line at least the better buy?** | ![Busted](https://img.shields.io/badge/Cheap_H_the_better_buy%3F-Busted-8b949e?style=flat-square) | 16.4 years of the same cash flows at ~30% off: H ×4.35 vs A ×3.61 total-return in CNY = **+11.6 bps/mo at *t* = 0.42** — statistically nothing. The discount never closes; the dividend edge is mostly eaten by the premium widening. |

> **In one sentence:** the folklore is the tape — Shanghai really pays ~30% more (our 14-pair
> average is +29.69%) for the identical cash flows and no force pushes the two prices together
> (Dickey-Fuller can't reject a random walk; 139 straight positive months post-Connect) — yet
> the premium is *not* a trade: the leg that would harvest its strong cross-sectional reversion
> (HAC *t* = +5.09 on paper) needs to short unborrowable A-shares, the executable half dies on
> costs, and even just buying the cheap H line earned nothing certifiable in 16 years.

## What we tested

We rebuild the Hang Seng AH-Premium construction pair by pair: 14 dual-listed Chinese companies
(Ping An, PetroChina, the big-four banks, China Life…), premium = raw A close ÷ (raw H close ×
HKD→CNY) − 1, monthly, 2010-01 → 2026-06. The Signal axis measures the level (HAC *t* **plus** a
Monte-Carlo placebo calibrated to the tape's own near-unit-root persistence — our synthetic
control shows 26/40 zero-mean worlds faking |HAC *t*| ≥ 2, so we refuse to certify a level by
HAC alone), non-convergence (AR(1) half-life 30 months, DF *t* = −1.49), one-factor co-movement,
and whether the premium predicts relative H-vs-A total returns (it does, cross-sectionally, at
*t* ≈ 5 with one documented lag). Tradability walks the arb channel door by door: non-fungible
lines, no A-share borrow, and the only executable half-trade netted to noise. The third axis
races the H basket against the A basket, total-return, both in CNY. Synthetic worlds with a
planted premium and a random-walk premium prove the machinery faithful. As-of **2026-06-30**;
fingerprint `f54aaa8986a9`.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what A- and H-shares are, why the same company has two prices, why nobody can close the gap (non-fungible tickers, no shorting, capital controls), and why "buy the cheap one" earned nothing — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the level + the near-unit-root MC placebo, DF/AR(1) non-convergence, PC1 co-movement, the fade with skip-month/sub-period/price-only/20-seed-baseline robustness, the executable-half cost table, the H-vs-A race, and the synthetic control suites |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`a_h_premium/`](a_h_premium/). Sibling studies: [05-twin-spread](../05-twin-spread/)
is Gatev distance pairs (different companies, reversion is the trade); this is the structural
dual-listing premium (same company, no convergence mechanism);
[618-gbtc-premium-cycle](../618-gbtc-premium-cycle/) shows what happens when an arb channel
finally opens. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
