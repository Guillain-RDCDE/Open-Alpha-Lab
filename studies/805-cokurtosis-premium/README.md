# Study 805 — Cokurtosis Premium 🪁

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do high-market-cokurtosis names earn a positive premium (Fang-Lai)? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The four-moment-CAPM systematic-kurtosis premium **fails to appear** on 50 liquid US mega-caps. The specified long-high-cokurt / short-low-cokurt spread is **−0.15 bps/day** (Newey-West *t* = **−0.11**) — a flat **zero**: the high- and low-cokurtosis books earn essentially the same +7.4 bps/day (Welch *t* = −0.06), the observed spread sits **−0.14 sd** from a 1,000-permutation placebo null, and it even **flips sign** between eras (*t* = −1.21 / +0.61). A 20-seed synthetic control recovers a *planted* premium cleanly (*t* = +7.20, fires on **0/20** nulls), so this is a genuine **absence** of the effect, not a broken sort. Higher co-moments are notoriously fragile out of sample; the fourth one adds nothing here. *Survivorship: current-membership mega-caps — magnitudes are an upper bound.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | There is no gross edge to monetise; costs simply turn the zero into a loss (**−2.29 bps/day** at 1 bp one-way, −10.29 at 5 bps). A paycheck from thin air is a Mirage. |

> **In one sentence:** the celebrated systematic-kurtosis premium — high-cokurtosis names
> should be paid to hold — **does not exist on liquid US mega-caps**; the long-short spread is
> a statistical zero (NW *t* = −0.11), and nothing survives costs, so the honest read is
> **claimed signal absent, paycheck a mirage**.

## What we tested

Fang & Lai (1997), **"Co-Kurtosis and Capital Asset Pricing"** (four-moment CAPM): a
security's **systematic kurtosis** — its **cokurtosis** with the market,
`E[(r_i−μ_i)(r_m−μ_m)^3]/(σ_i·σ_m^3)` — is a priced risk, so high-cokurtosis names (those
that amplify the market's fat-tailed moves) should earn a *positive* premium and a long
high-cokurt / short low-cokurt book should earn a positive spread. We take the self-contained
daily version on a **liquid 50-name US cross-section (yfinance daily OHLC, total-return,
2010-01-04 → 2026-06-30)**: each name's **trailing-252-day cokurtosis** with the equal-weight
market (a standardised fourth co-moment, vectorised via rolling raw moments), sorted
point-in-time (signal known at the close of `t−1`, one shift, zero look-ahead), with a
Newey-West *t* on the daily spread, a 1,000-permutation placebo, a two-era robustness cut, a
costed long-short timer, and a 20-seed synthetic positive control. The universe is a
**current-membership** survivor set (`quantlab.universe` opt-in guard) — named on the
**Signal** axis. **Dedup:** [504-coskewness](../504-coskewness/) tests **systematic
co-skewness** — the *third*-order co-moment with the market **squared**, not cubed;
[238-betting-against-beta](../238-betting-against-beta/) tests the *second*-order co-moment
(co-variance / beta), not a higher moment; [803-realized-skewness-reversal](../803-realized-skewness-reversal/)
tests a name's **own** total skewness, not its co-movement with the market. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "co-kurtosis with the market" means, why four-moment CAPM says it should pay — and why on mega-caps it pays nothing |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the spread Newey-West *t*, the pooled Welch book test, the 1,000-permutation placebo, the two-era cut, the cost math, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`cokurtosis/`](cokurtosis/). Cross-section pulled through the `quantlab.universe`
survivorship guard (current membership → magnitudes are an upper bound). **Not investment
advice** — research & education. See [LICENSE](../../LICENSE).*
