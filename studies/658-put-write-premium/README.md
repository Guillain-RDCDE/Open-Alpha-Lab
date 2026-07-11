# Study 658 — Put-Write-Premium 🎯

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does put-writing beat buy&hold, risk-adjusted? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | PUTW's entire live history (2016-02-24 → 2026-06-30): excess-return gap vs SPY **−7.48%/yr**, HAC *t* = **−3.21**; Sharpe **0.53 vs 0.80** (bootstrap 95% CI **[−0.48, −0.00]**); CAPM alpha vs SPY **−1.78%/yr at *t* = −0.95** — not distinguishable from zero. |
| **Tradability** — is it a better way to own equities? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No residual edge to deploy: a lower-beta (0.60) equity substitute with a *worse* Sharpe than the index it's priced off, whose "protection" widens from beta 0.55 to **0.93** exactly on the 33 worst days in the sample (*t* = 2.34) — and whose single worst month beat SPY's own worst month despite 29% less volatility. |
| **"Is it just truncated equity beta?"** | ![Confirmed](https://img.shields.io/badge/Truncated_beta%3F-Confirmed-8b949e?style=flat-square) | Beta 0.60 (*t* = 10.26) explains the average day; alpha is statistically zero; and the beta itself is unstable — it converges toward full equity exposure exactly when a genuine diversifier is supposed to prove its worth. |

> **In one sentence:** writing cash-secured S&P 500 puts through the only liquid ETF that does
> it (PUTW) delivered a smaller, less stable slice of SPY's own beta — not a distinct
> risk-adjusted premium — trailing the index it's compared against by 7.5%/yr (*t* = −3.21) over
> its entire ~9.4-year live history, with the "lower risk" evaporating toward full beta on
> exactly the worst days.

## What we tested

The pitch: sell a rolling one-month at-the-money S&P 500 put every month, cash-secured in
T-bills, and harvest the **variance risk premium** — implied vol runs richer than realized vol
on average — for a smoother, better risk-adjusted ride than buy-and-hold (the CBOE PUT index's
own marketing, and the standard "premium income" retail case). We test it on **PUTW**, the only
liquid tradable fund that implements it, against **SPY**, with **BIL** as the cash leg: a CAPM
regression (is the return beyond beta?), a crash-day interaction (does the lower beta survive a
crash?), and a block-bootstrapped Sharpe race — not the untradeable 1986→ CBOE index, whose
longer, friendlier sample would flatter a fund nobody could have actually bought.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "selling insurance on the S&P" sounds like free money, what the fund's own decade actually did, and why the smaller ride gets rough exactly when it matters |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the CAPM alpha/beta split, the crash-conditional-beta interaction test, the Sharpe bootstrap, the monthly capture and tail decomposition, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`put_write_premium/`](put_write_premium/). PUTW and SPY are both live, currently-listed
funds over the entire window tested — no survivorship basket involved. **Not investment advice**
— research & education. See [LICENSE](../../LICENSE).*
