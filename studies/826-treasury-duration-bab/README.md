# Study 826 — Treasury Duration BAB 🏦📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does betting-against-beta earn a low-risk alpha inside the Treasury curve (Frazzini-Pedersen)? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The BAB book prints **+1.31 bps/day** (Newey-West *t* = **+2.50**, right sign) and is genuinely beta-neutral (residual β = −0.007) — but it does **not** hold up. The permutation placebo shows the beta sort earns *less* than a random assignment into the same `1/β` leverage cage (observed +1.31 vs placebo mean **+2.57 bps**, ≈**1.74σ into the left tail**): the beta **signal adds no value**, the positive number is mechanical *levered carry* off the low-vol leg. And it is entirely a **2018–2026** phenomenon (2010–2017 *t* = +0.49). A 20-seed synthetic control recovers a *planted* Frazzini-Pedersen alpha cleanly (fires on **1/20** nulls, residual β ≈ 0), so the machinery is sound — the claimed low-risk edge is simply absent as a signal. *Survivorship: a fixed five-ETF maturity ladder — a design choice named on the Signal axis.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | What remains is ~3%/yr net (net *t* ≈ 2.1 at 1–5 bps costs) but it rests on ~**5× gross leverage** financed at `rf≈0`, is beaten by random assignment, and lives in one era. A realistic financing charge on the levered long — unmodelled beyond short-leg borrow — erodes the levered carry that produces it. A **Mirage**, not a harvestable low-risk premium. |

> **In one sentence:** betting-against-beta across the Treasury maturity curve *looks* like it
> works (NW *t* = +2.50, beta-neutral) but the beta signal itself adds nothing — a random leg
> assignment into the same leverage cage does **better** — so the celebrated low-risk alpha is
> here just levered carry off the short-duration end, confined to one era and financed for free:
> **claimed signal absent, paycheck a mirage**.

## What we tested

Frazzini & Pedersen (2014), **"Betting Against Beta"**: low-beta assets earn higher
*risk-adjusted* returns, so a **BAB** book — long the low-beta legs levered to unit beta, short
the high-beta legs, beta-neutral — earns a positive alpha, documented across asset classes
**including US Treasuries by maturity**. We rebuild that Treasury-curve version from **five
iShares Treasury ETFs laddering the curve** (SHY 1-3y → IEI → IEF → TLH → TLT 20y+; yfinance
daily total-return closes, 2010-01-04 → 2026-06-30): an equal-weight **duration factor**, each
ETF's **trailing-252-day beta** to it (known at the close of `t−1`, one shift, zero look-ahead),
the Frazzini-Pedersen rank-weighted book levered to unit beta, a Newey-West *t*, a
factor-regression alpha + residual beta, a 1,000-permutation placebo, a two-era robustness cut,
a costed leveraged timer, and a 20-seed synthetic positive control. The five-ETF ladder is a
**design selection** — named on the **Signal** axis. **Dedup:**
[796-corporate-bond-low-risk](../796-corporate-bond-low-risk/) tests the low-risk tilt in
**corporate/credit** bonds (credit risk), not duration inside the government curve;
[238-betting-against-beta](../238-betting-against-beta/) runs BAB across the **equity**
cross-section (market beta), not Treasury maturities; [581-term-premium](../581-term-premium/) is
a **time-series** *when-to-own-duration* timer on TLT, not a beta-neutral cross-sectional book.
As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why levering up the boring short-duration end *should* pay — and why on this curve the beta signal added nothing |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the beta ladder, the BAB Newey-West *t*, the factor-regression alpha & residual beta, the 1,000-permutation placebo, the two-era cut, the leveraged cost math, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`duration_bab/`](duration_bab/). Five Treasury-ETF total-return closes pulled via
yfinance into this study's own `_cache/`; the reproducible core runs offline.
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
