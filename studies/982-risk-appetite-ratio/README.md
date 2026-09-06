# Study 982 — The Appetite Gauge 🌡

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the high-beta / low-volatility ratio say anything the market's own trend does not? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | The gauge is **35% explained by the market itself** — the high-beta minus low-volatility spread carries a beta of **+0.74** to the index, so a rising gauge is very largely a rising market. On its own the raw spread's trailing 63-day average predicts the next 21 days with *t* = **-0.68**; once it is beta-neutralised that falls to **+2.21**, and in the decomposition where it competes with the market component it is **+3.55** against the market's -3.71. Across the 27 lookback × horizon × signal cells, 14 clear |*t*| = 2 against 1.4 expected by luck — and the sample is only 15 years long, which caps what any of these numbers can mean. |
| **Tradability** — does a risk-on/risk-off rule built on it beat holding the index? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | The rule — own the index while the gauge is positive, hold bills otherwise — was invested 57% of the time, switched 12.4 times a year and returned **+10.63%/yr** against **+15.10%** for holding the index (-4.46%/yr; Sharpe +1.11 vs +0.92, *t* = -1.36). Its worst drawdown was -19.4% against -33.7% — the familiar trade of return for a shallower hole, available from any trend filter. |

> **In one sentence:** The high-beta / low-volatility ratio is **35% the market**, and once that is projected out the residual gauge predicts the index with *t* = +3.55 in the regression where the market component sits beside it — so what the desks are watching is, mostly, the market.

## What we tested

When the racy half of the S&P beats the boring half, desks read it as risk appetite
— and when the ratio rolls over, as a warning. The ratio of **SPHB** (high beta) to **SPLV**
(low volatility) is the most-watched version of that gauge that needs no options data. It also
has a problem nobody mentions: SPHB's beta is around 1.3 and SPLV's around 0.7, so the spread is
roughly a **0.6-beta position in the index**, and its trend is correlated with the index's trend
almost by construction. A predictive regression that ignores this is measuring market momentum
with extra steps.

So the study runs three signals side by side: the **raw** spread everybody quotes, the same
spread with a **rolling backward-looking beta removed**, and the **market's own trailing
return** as the control — univariate at three lookbacks and three horizons with horizon-lag HAC
errors, and then in a multiple regression where all three compete for the same variance. The
risk-on/risk-off rule each implies is priced against buy-and-hold with costs, and the gauge's
behaviour is examined in each of the sample's four actual drawdowns, labelled as the anecdote it
is. A synthetic world with a *planted* appetite factor and a null where the spread is nothing
but leverage establishes that the apparatus can tell the two apart.
**Dedup:** distinct from **330-low-volatility-anomaly** and **903-sector-neutral-lowvol** (the
low-volatility factor's own returns), **238-betting-against-beta** (the beta factor as a
long-short strategy), **131-utilities-canary** (a defensive *sector* as a signal),
**115-credit-spreads** (credit as a risk gauge — which appears here only as a control) and
**980-semis-lead-the-market** (a cyclical sector's lead, not an intra-market risk sort).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the gauge is actually made of, the beta confound in one chart, and what survives once the market is taken out of it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | rolling-beta neutralisation, a full lookback × horizon grid with HAC errors, the multiple regression that settles the confound, the rule and its sweep, the drawdown anecdotes and a planted-factor control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`appetite/`](appetite/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
