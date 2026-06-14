# Study 143 — Dividend-Capture

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Ex-date drop/div ratio = **+1.11** (price drops *more* than the dividend), HAC *t* vs 1.0 = +1.45; gross capture mean = **−12.9 bps/trade**, HAC *t* = −1.85. No systematic under-adjustment anywhere in the 8-ticker basket. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Gross is already negative; at 5 bps round-trip cost net *t* = **−2.59** (a statistically certified loser). No break-even cost exists. |
| **Drop < Dividend?** | ![No](https://img.shields.io/badge/Drop_<_Dividend%3F-No-8b949e?style=flat-square) | Only 48.7% of events show a drop smaller than the dividend — a coin flip, not a systematic edge. |

> **In one sentence:** the dividend-capture strategy earns nothing — the market prices the ex-dividend drop to fully offset the cash payment, and once round-trip costs are added, the trade is a certified loser.

## What we tested

The folk recipe: a few days before the ex-dividend date, buy the stock; on the ex-date,
collect the dividend; sell immediately after. The steelman: stock prices *systematically
under-adjust* on the ex-date (price drops less than the dividend), leaving a net gain.
We test this using **unadjusted** daily prices (auto-adjust=False) for a basket of 8
dividend-paying US large-caps (SPY, VYM, T, MO, VZ, XOM, JNJ, KO) spanning 2000–2026,
823 events in total. We measure the ex-date price drop vs the dividend amount, run the
2-day capture trade, and sweep transaction costs. A deterministic synthetic tape with a
tunable market-efficiency knob provides the positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the "free dividend" is not free, the ex-date price mechanics in plain language, the win-rate trap, and what costs do |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | drop-ratio t-stat (HAC), per-ticker breakdown, cost sweep, the shuffled-date control, and the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`dividend_capture/`](dividend_capture/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
