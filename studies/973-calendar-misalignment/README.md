# Study 973 — Different Holidays 🗓

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the calendar mismatch bias the estimates? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Correlation with the US market rises from **0.75** at daily frequency to **0.77** at monthly on EWJ — a lift of **+0.02** — and **0 of 4** foreign tapes lift by more than 0.10. The same-market control (IWM) lifts by **+0.01**, which is the machinery's own noise floor. Dimson's correction recovers most of it without changing frequency: beta on EWJ goes from 0.81 to **0.78**, and the lagged US coefficient — the smoking gun — is -0.03. |
| **Tradability** — is the fix worth applying to a real portfolio? | ![Useful](https://img.shields.io/badge/Useful-2ea44f?style=flat-square) | A minimum-variance book built on the **daily** covariance matrix promises 17.63% annualised and delivers **15.35%** at the monthly horizon — it understates its own risk by **-12.9%** — and its weights differ from the ones the unbiased matrix would choose by up to 18%. The fix costs nothing: measure at a lower frequency, or add one lead and one lag. |

> **In one sentence:** Nothing is wrong with the data — the tapes are all New-York-listed and close at the same minute — but the *markets underneath* keep different hours, and that alone drags a daily correlation down by up to 0.02, biases beta by -0.04 and lets a minimum-variance optimiser understate its own volatility by -13%.

## What we tested

When New York is open and Tokyo has been shut for fourteen hours, a US-listed Japan
ETF is pricing yesterday's Japanese session plus today's American news. Same-day correlations
and betas across such markets are biased toward zero — a result that has been in the literature
since **Scholes & Williams (1977)** and **Dimson (1979)** and is ignored by every risk system
that builds a daily correlation matrix. This study measures the bias on **SPY, EWJ, EWU, EWG
and FXI**, with **IWM** as a same-market control that runs through every table.

The design choice that makes the result hard to dismiss: all five tickers are **New-York-listed
ETFs**, quoting on the same exchange and closing at the same minute. There is no data-alignment
problem to fix — the mismatch that remains is in the *underlying* markets' trading hours, which
no amount of careful joining can repair. We measure correlations at daily, weekly, biweekly and
monthly frequencies (the model-free fix), apply Dimson's aggregated-coefficients beta and the
Scholes-Williams estimator (the model-based fixes), and then price the consequence: a
minimum-variance portfolio built on the daily covariance matrix, evaluated at the monthly
horizon, understates its own volatility.
**Dedup:** distinct from **917-nav-staleness-timezone** (an ETF's NAV versus its price, an
arbitrage question), **578-cross-asset-correlation-regime** and **579-equity-bond-corr-flip**
(how correlations *move*, not how they are mismeasured), **634-us-leads-the-world** and
**981-asia-tech-canary** (lead-lag as a *tradable signal* rather than an estimation bias),
**146-country-momentum** (cross-country returns) and **613-currency-hedged-etf-carry** (the FX
leg).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a closed market still has a price, the correlation that appears when you stop measuring daily, and the control that proves it is not the arithmetic |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | lead-lag correlation profiles, Dimson and Scholes-Williams betas, frequency aggregation, a same-market control through every test, minimum-variance promised-versus-delivered volatility, and a planted-delay simulation |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`calendar_gap/`](calendar_gap/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
