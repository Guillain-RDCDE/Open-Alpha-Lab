# Study 180 — TRIX

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Zero-line cross: gross **+8 bps/trade**, HAC *t* = **+0.34**; signal-line cross: *t* = **−1.55**. No variant, period, or holding window clears \|*t*\| ≥ 2. Bonferroni threshold (4 periods) = 2.57 — not approached. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Gross expectancy statistically zero at every parameter setting. Low turnover (~6 zero-line crosses/yr per instrument) means costs are not the killer — the signal simply does not exist. |
| **Triple-smoothing lag?** | ![Confirmed_lag,_no_edge](https://img.shields.io/badge/Confirmed_lag,_no_edge-8b949e?style=flat-square) | Three sequential EMAs accumulate ~22 bars of effective lag; zero-line crosses confirm moves after the trend has largely played out. The triple smoothing filters *both* noise *and* timely signal. |

> **In one sentence:** TRIX's triple smoothing delays every entry by roughly a calendar month — by the time the zero-line fires, the trend it "confirms" has already been priced in, and neither the zero-line nor the signal-line cross adds any directional information over a coin (HAC *t* ≤ 0.34) on 15 years of daily data.

## What we tested

Hutson's TRIX indicator (1983): the 1-day rate-of-change of a triple-smoothed EMA, designed
to "filter out short-term cycles" and signal only major trend changes.  Two signals:
(1) **zero-line cross** — TRIX crosses zero → buy/sell; (2) **signal-line cross** — TRIX
crosses its 9-day EMA, MACD-style.  Tested long/short flips vs a random-direction control
on identical entries, across SPY, QQQ, IWM, AAPL, and MSFT, ~15 years of daily bars,
multiple holding windows and TRIX periods.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what TRIX is, the lag trap in plain language, the fair bet vs a coin, why triple smoothing is not triple filtering |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-instrument HAC *t*, hold-period sweep, period sensitivity with Bonferroni, the synthetic positive control, cost anatomy |

Sources & literature map: [docs/references.md](docs/references.md).  Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`trix/`](trix/).  **Not investment advice** — research & education.  See [LICENSE](../../LICENSE).*
