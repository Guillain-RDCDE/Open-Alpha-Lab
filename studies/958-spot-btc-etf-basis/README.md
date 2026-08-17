# Study 958 — Spot ETF Basis 🧊

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the carry still there, and did the launch compress it? | ![Mixed](https://img.shields.io/badge/Signal-Mixed-8b949e?style=flat-square) | **Still there, loudly:** BITO bleeds **−5.72%/yr against IBIT on matched closes**, certified on the estimator that assumes least — 29 **non-overlapping** monthly gaps, plain *t* = **−9.83**, no HAC anywhere (endpoint *t* = −7.24 agrees). The trend slope's *t* = −30.7 is quoted but **discounted**: that fit's residual is near a unit root (DF *t* = −3.0), so it flatters. Against the coin the drag is −5.97%/yr (trend *t* = −29.3, −17.7 at 250 lags) but only **−2.17** on the blunt monthly ruler — sixteen hours of bitcoin sits in every endpoint. An implied front basis of **+9.4%/yr**, **+5.0%/yr above cash**, on a ruler calibrated to read IBIT's 25 bp fee **exactly**. **Compression: rejected — at every window width.** Matched ±12-month windows either side of 2024-01-11 read **−7.32% vs −7.35%/yr** (change −0.03 pp, *t* = **−0.08**), and that is the *flattest* member of its family, so the sweep is published: ±6m −2.20 pp, ±9m −0.89, ±12m −0.03, ±18m −4.50, ±24m −4.45 — **not one width compresses**, several are significantly wrong-signed. The full-sample "significant" break (*t* = −9.50) is wrong-signed too and ranks only **7 of 44** placebo split dates. No survivorship (three continuously listed instruments). |
| **Tradability** — can you harvest the residual? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Long IBIT / short BITO nets **+3.28%/yr at 2% borrow and 5 bps**, excess of cash on both legs (Sharpe 1.72, HAC *t* = +4.47 and +3.74 at 250 lags, non-overlapping monthly *t* = +4.35 with 28/30 months positive, bootstrap CI [+0.95, +2.48], worst drawdown 0.67%) — and **dies at 5% borrow** (+0.30%/yr, *t* = +0.41). It is also **shrinking**: net +4.50% (2024) → +3.22% (2025) → +0.99% (2026 H1). Capacity is BITO's borrow pool; the risk-free version — just own the spot wrapper — is cost avoidance, not alpha. |

> **In one sentence:** the spot ETFs closed the *trust* discount but left the *futures* basis untouched — twelve months either side of the January-2024 launch the futures wrapper bled an identical ~7.3%/yr against bitcoin (*t* = −0.08 on the difference), and no other window width makes it cheaper either; the carry is still worth **+5%/yr above cash** two and a half years later, and the gradual fade since 2025 tracks the crypto cycle rather than the plumbing.

## What we tested

We measure each wrapper's **tracking difference** — the cumulative log gap to a bitcoin
reference — and read its annualised drag as the **HAC-robust trend slope on calendar
time**, because `BTC-USD` is stamped at 00:00 UTC while the ETFs mark at 16:00 New York
and the usual mean-of-daily-differences estimator is exactly the two-endpoint one. The
spot wrappers calibrate that ruler against a *known* cost: it reads IBIT at −0.250%/yr
against a 0.25% fee, and FBTC a shade cheaper, as its launch waiver implies. Because that
slope's HAC *t* is the most generous number available, every headline is repeated on a
deliberately blunt **non-overlapping monthly** estimator that needs no HAC. Then a
broken-trend era test at **2024-01-11**, a **matched-window sweep** (±6 to ±24 months, so
no single width is the answer), a **44-date placebo
sweep**, a calendar-year table, a bandwidth sweep to 250 lags, and the long-spot /
short-futures harvest across borrow and cost grids. All total-return
(`auto_adjust=True` — BITO's distributions are reinvested), one execution lag on the
pair's monthly reset, the pair dollar-neutral and self-financing (so its return is
already excess of cash on both legs), fees and borrow labelled **assumptions** and
swept. **Dedup:**
[619-bito-roll-drag](../619-bito-roll-drag/) measured the *level* of the toll and where in
the month it is paid; 958 asks the event question it leaves open — did the spot ETFs
*change* it — with a different estimator, a different ruler and no `BTC=F`.
[618-gbtc-premium-cycle](../618-gbtc-premium-cycle/) covers the *other* pre-2024 wedge,
the trust discount, which the same event genuinely did extinguish. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why an ETF that buys next month's bitcoin still costs you 6%/yr, why the spot ETFs fixed one wrapper wedge and not the other, and what "the basis follows the cycle" looks like year by year |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the trend-slope estimator, its fee calibration and the residual diagnostics that say how far to trust its *t*, the non-overlapping monthly cross-check, the broken-trend era test with its ±6-to-±24-month window sweep, the placebo date sweep, HAC bandwidths to 250 lags, the implied-basis decomposition and fee sweep, the borrow-swept harvest, and the planted-compression synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`etf_basis/`](etf_basis/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
