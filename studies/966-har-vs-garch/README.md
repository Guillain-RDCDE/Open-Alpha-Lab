# Study 966 — Forecasting Tomorrow's Vol 🔮

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do the models actually differ out of sample? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Out of sample at the one-day horizon, **GARCH(1,1) QML** is the best model on **5 of 6** tapes, and something beats the 21-day rolling baseline on **6 of 6** — pooled Diebold-Mariano **+4.65** against a baseline that costs nothing to compute. The ordering is stable across horizons (1 / 5 / 21 days) and reproduced on simulated data where the true variance is observable. |
| **Tradability** — is the winner worth the extra machinery? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | The pooled QLIKE improvement over the rolling window is **3.5%**. EWMA — one parameter, never fitted — captures 89% of what the fitted GARCH achieves, and the fitted models cost a maximum-likelihood optimisation every quarter. The gap between the best and worst *sensible* model is smaller than the gap between using any of them and using none. |

> **In one sentence:** Volatility is forecastable and all four models know it — the interesting result is how little separates them: **GARCH(1,1) QML** wins by **3.5%** of QLIKE over a 21-day rolling average, and a one-line EWMA gets most of the way there for free.

## What we tested

Volatility clusters — that has been known since Mandelbrot (1963) and is the least
controversial fact in empirical finance. The live question is whether the *models* built on it
earn their complexity. Four forecasters read the same daily total-return series on **SPY, QQQ,
IWM, EEM, TLT and GLD**: a 21-day rolling standard deviation, RiskMetrics **EWMA** (one fixed
parameter), a **GARCH(1,1)** fitted by quasi-maximum likelihood (implemented here directly, no
external library), and Corsi's **HAR-RV**. Every model is refitted on an expanding window every
quarter and every forecast is strictly out of sample; targets are realised variance at 1, 5 and
21 days; scores are QLIKE and MSE; comparisons use a HAC-corrected Diebold-Mariano test against
the free baseline.

Two design decisions do the heavy lifting. First, the target on real data is a **squared daily
return** — unbiased but very noisy — so the whole tournament is re-run on simulated data where
the conditional variance is *observable*, and the two rankings are compared: that is how you
find out what the noisy proxy is costing you. Second, the same tournament is run on a
**constant-volatility null**, where any model that beats the rolling window is revealing an
artefact of the scoring rather than a fact about markets.
**Dedup:** distinct from **965-range-vol-estimators** (measuring *today's* volatility from a
richer bar, not forecasting tomorrow's from the close), **817-realized-volatility-trend**
(volatility as a *signal* for returns), **130-vol-risk-premium** and **374-vol-of-vol**
(implied versus realised), **898-managed-vol-equity** and **633-btc-vol-targeting** (what to do
with a forecast, not how to make one).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why tomorrow's calm is predictable when tomorrow's return is not, the four models in plain language, and the uncomfortable smallness of the gap between them |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | QML GARCH from scratch, HAR design and coefficients, strictly out-of-sample expanding-window forecasts, QLIKE/MSE with HAC Diebold-Mariano, horizon robustness, the proxy-versus-truth comparison and the constant-vol null |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`vol_forecast/`](vol_forecast/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
