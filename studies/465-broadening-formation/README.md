# Study 465 — Broadening Formation 📣

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the megaphone call the turn? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The "short the lower-boundary break" rule does **not** beat a drift-matched **random-short** baseline: break − random = **−197.4 / −119.4 / −287.9 / −819.8 bps** at 5/10/20/60 days — the megaphone short is *worse* than a random short at every horizon, and the break-vs-random Welch *t* is **negative** (−3.13 / −1.70 / −2.33 / −3.91). The big negative one-sample *t*'s (20d **−3.07**, 60d **−3.73**) are **pure negative beta** — a short on an up-drifting index. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No reversal edge once the index's drift is accounted for — long or short. The per-name sample is paper-thin (**25** megaphones in 21 years), and costs only deepen the hole. Nothing to scale. |
| **"Does expanding volatility forecast a turn?"** | ![Busted](https://img.shields.io/badge/Expanding_vol_forecasts_a_turn%3F-Busted-8b949e?style=flat-square) | Scramble the megaphone's diverging geometry into nonsense (shuffled-pivot placebo) and the result barely moves — the real one even sits at the *bad* end: **97%** of nonsense megaphones do at least as well (*p* = **0.974**). The expanding range carries no forecasting information. |

> **In one sentence:** The broadening "megaphone" top looks like a blow-off because indices are choppy and drift up — encode it mechanically (confirmed-fractal pivots, rising highs **and** falling lows, no eyeballing) and fire the "short the lower-boundary break" rule, and across 5 indices over 21 years it fires only **25 times**, *loses more than shorting on random days* at every horizon, and the geometry placebo leaves the result untouched (*p* = 0.97): a rare, seductive shape with no signal.

## What we tested

We encode the tightest mechanical version a proponent would accept. Swing pivots are **confirmed fractals** (a local extremum with *k* = 10 strictly-beaten bars each side, usable only 10 bars later — no look-ahead); at every bar we take the last two confirmed swing highs and lows and call it a **megaphone** only when the highs are *rising* **and** the lows *falling* (the boundaries diverge); a **short** fires on the first close **below the lower boundary**, entered at the **next close** (one documented lag), and we measure the forward 5/10/20/60-day return of the short on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The Signal axis is **break vs a drift-matched random-short baseline** (a Welch *t*) — the only honest test for a short on an upward-drifting tape — plus a **shuffled-pivot geometry placebo** that destroys the diverging boundaries while keeping the price marginal. Tradability charges costs on every break. A deterministic synthetic control with a *planted* expanding-range reversal proves the detector is live (edge 0 → *t* = −0.93; planted reversal → short **+515 bps**, win **100%**, *t* = +6.88), so the flat/negative real-tape result is a genuine "nothing there".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a broadening/megaphone top is, why a short on a rising market always loses, the break-vs-random-short race, and the geometry scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | mechanical megaphones, one-sample HAC *t* vs the beta trap, the random-short Welch test, the shuffled-pivot placebo, per-ticker deltas, costs, and a synthetic planted-reversal control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`broadening_formation/`](broadening_formation/). Pivots are confirmed fractals (k = 10) with a 10-bar confirmation lag; entry is the next close (one lag); returns are signed for a short. Basket is surviving liquid ETFs — but this is a single-instrument pattern study, so the random-short baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
