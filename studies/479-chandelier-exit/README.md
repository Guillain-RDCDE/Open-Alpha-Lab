# Study 479 — Chandelier Exit 🕯️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the chandelier entry forecast? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The breakout **entry** beats a drift-matched **random-entry** baseline — but **only at 5 days** (Welch *t* = **+3.31**, *p* = 0.001). It decays straight to noise: Welch *t* = **+1.77 / +1.59 / +1.32** at 10/20/60 days (all *p* > 0.05). A short-lived momentum pop, and it's the **breakout, not the trailing stop**. |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | The lone edge is a 5-day breakout tilt that costs eat into and that does **nothing** for the thing you'd actually run: the chandelier-managed equity curve **Sharpe-loses to buy-and-hold in all 5 names** and gives up ~half the CAGR. Nothing scales. |
| **"Does the ATR trail beat holding?"** | ![Busted](https://img.shields.io/badge/ATR_trail_beats_holding%3F-Busted-8b949e?style=flat-square) | Chandelier-managed long: CAGR **4.3/7.5/2.7/4.7/5.5%** vs B&H **11.0/15.5/8.8/10.0/11.2%**; Sharpe lower in **5/5**. Scramble the ATR widths and the entry result is untouched (**p = 0.87**). The volatility-scaled trail carries no information; its only "win" is a smaller drawdown — pure time-in-cash. |

> **In one sentence:** Chuck LeBeau's chandelier exit (`HH − 3·ATR(22)`) looks clever because it cuts your drawdown — but encode it mechanically across 5 indices over 21 years and the truth is plain: its *entry* is a 5-day breakout pop (gone by day 10), its *trailing stop* adds nothing the marginal ATR doesn't (placebo *p* = 0.87), and its *managed long loses to buy-and-hold on both CAGR and Sharpe in every name* — a trailing stop on an up-drifting tape just trades off your free beta.

## What we tested

We encode the canonical mechanical chandelier a proponent would deploy: Wilder **ATR(22)**, a long that re-arms on a fresh **22-day breakout high**, a trailing stop hung **3 × ATR below the highest high since entry**, exiting on the first close below it. State is read on the close of *t* and traded at the **next close** (one documented lag), measuring the forward 5/10/20/60-day return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The **Signal** axis is the breakout entry vs a **drift-matched random-entry** baseline (a Welch *t*) — the only honest test on a drifting tape. The **thesis** axis prices the whole strategy: the chandelier-managed long's CAGR/Sharpe/drawdown vs **buy-and-hold**, with switch costs. A **scrambled-ATR placebo** permutes the ATR widths to ask whether the *trail itself* is load-bearing. A deterministic synthetic control with **planted momentum** proves the detector is live (edge 0 → *t* ≈ −1.1, managed long ties B&H; planted trend → *t* = +4.07, managed long Sharpe +4.94 vs −0.15), so the muted real-tape result is a genuine "the trail can't pay because indices don't trend enough".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a chandelier exit is, why a trailing stop can't add return on a rising market, the entry-vs-random race, the strategy-vs-hold race, and the ATR scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | mechanical ATR(22)-3× trail, one-sample HAC *t* vs the beta trap, the random-entry Welch test, the equity-curve vs buy-and-hold, the scrambled-ATR placebo, per-ticker deltas, costs, and a synthetic planted-momentum control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`chandelier_exit/`](chandelier_exit/). Wilder ATR(22), stop = HH − 3·ATR, 22-day breakout re-entry; state read on close *t*, traded the next close (one lag). Basket is surviving liquid ETFs — but this is a single-instrument trend study, so the random-entry baseline and the buy-and-hold comparison neutralize the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
