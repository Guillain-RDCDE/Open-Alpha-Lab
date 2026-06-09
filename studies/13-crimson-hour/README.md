# Study 13 — Crimson-Hour 🩸

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a red first hour actually *forecast* the close? | 🟡 `WEAK` | A red opening hour really does tilt the day red — on real SPY/QQQ the close is red **69–72%** of the time vs a **~45%** baseline (matching edgeful's own 46.1% / 44.5%). But strip the head-start and the *rest of the day* (10:30→close) is red only **49–51%** — a genuine continuation lift of just **+6 pp**. **~75% of the headline is mechanical**, and the "IB-high rejection" leg adds **nothing** over the candle's sign (Fisher *p* = 0.62 / 1.00). |
| **Tradability** — could you get paid? | 🔴 `MIRAGE` | We ran the only positive-expectancy expression — short 10:30→close on red-hour days. It grosses **+2.2 bps** (SPY) / **+3.7 bps** (QQQ) per trade at a Sharpe of just **0.40 / 0.54**, with a *t*-stat of only **0.68 / 0.91** — *indistinguishable from zero*. The **break-even round-trip cost (2.2 / 3.7 bps)** sits inside a realistic two-leg intraday round-trip, so net Sharpe is **+0.03 / +0.25 at 2 bps**. A real *signal*, an untradable *trade*. The newsletter's own hedge — *"bias, not a trade"* — is the honest read. |
| **"88% predictive?"** — where does the headline come from? | ⚪ `INFLATED` | Small samples + selection. A true ~**70.5%** edge, taken as the *best* of a dozen "confluences" of 25 sessions ("one prompt, combine the reports"), **expects** a top score of **84.6%** (95th pct 92%) and clears 88% **36%** of the time. The 88% is the expected output of the search, not evidence. |

> **In one sentence:** the red-opening-hour "88% the day closes red" is a **real ~6-point intraday-continuation lean wearing a mechanical head-start and a cherry-picked headline** — a red first hour leaves the day *already* in the red (that's ~75% of the effect), the IB-rejection second signal is redundant, and an AI dashboard that mines reports until one prints 88% is just rediscovering the garden of forking paths.

## What we tested

A trading newsletter ([edgeful](https://www.edgeful.com)) reports that when the **09:30–10:30 ET opening candle closes red** *and* the first hour's **high prints before its low** ("IB-high rejection"), the full session closes red **22/25 = 88%** on ES and **28/31 = 90%** on NQ — a dashboard built *"in 5 minutes, one prompt, no code"* against an API by combining two reports. We steelman it (there *is* a documented intraday-continuation effect under it), then reduce real Yahoo intraday bars to a one-row-per-session panel — **ES=F / NQ=F at 5-minute** fidelity for the faithful confluence (fine enough to order the IB high vs low), **SPY / QQQ at 1-hour** for the ~700-session high-power test — and split the headline into a *mechanical head-start*, a *genuine forecast* (tested, and then cost-swept as an afternoon short), an *IB increment*, and *selection inflation*.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story + the stakes in plain language, the head-start that masquerades as a forecast, and how mining manufactures an 88% |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the full machinery: the mechanical/forecast decomposition, the continuation significance test, the two-proportion + Fisher test of the IB increment, Wilson + beta-binomial on 22/25, the forking-paths Monte-Carlo, and the cost-swept afternoon-short backtest |

The real run — every fingerprinted, as-of'd table — is in [docs/results.md](docs/results.md); reproduce offline via [examples/run_synthetic_demo.py](examples/run_synthetic_demo.py) and on the real tape via [examples/verify.py](examples/verify.py) (`--fetch` once to populate the bar cache).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
