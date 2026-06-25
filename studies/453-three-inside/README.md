# Study 453 — Three-Inside-Up / Down 🕯️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the pattern flip the trend? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The confirmed three-inside-up does **not** beat a drift-matched **random-entry** baseline: confirmed − random = **+20.1 / −43.2 / −139.9 / −132.3 bps** at 5/10/20/60 days, and the Welch *t* **never clears 2** (min **−1.75** at 20d). Unlike most chart rules there isn't even a beta mirage — the rule's own one-sample *t* never clears 2 and turns **negative at 20 days** (the "bullish reversal" is, on average, followed by a *fall*). |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | A rare pattern — **64** signals in 21 years across 5 tapes — that loses to a coin-flip entry in **5 of 5** names (20d *t* = −1.61). No drift to inherit, costs only deepen the hole, nothing to scale. |
| **"Does the confirmation candle add edge?"** | ![Busted](https://img.shields.io/badge/Confirmation_adds_edge%3F-Busted-8b949e?style=flat-square) | Drop the confirming third candle (harami-only placebo) and the 20-day return **improves** by **+198 bps** (+91.4 vs −106.9; Welch *t* = **−2.49**, *p* = **0.015**). The celebrated confirmation candle is a *negative* contributor — it makes you buy after a one-day pop. |

> **In one sentence:** The three-inside-up looks like a tidy bullish reversal, but encode it mechanically (down candle → inside harami → confirming close past the first bar, no eyeballing) and fire it 64 times across 5 indices over 21 years, and it **loses to buying on random days** (and goes negative at 20 days) — while the famous *confirmation candle*, the part the lore prizes most, actually **subtracts** ~198 bps (*p* = 0.015): it makes you chase a one-day pop.

## What we tested

We encode the tightest mechanical version a proponent would accept. **Bar A** is a down candle ending a 5-day downtrend; **Bar B** is a strict body-harami (its whole range inside A's body); **Bar C** confirms by closing back above A's open and above B's close. The signal completes on C's close, the long is entered at the **next close** (one documented lag), and we measure the forward 5/10/20/60-day return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The Signal axis is **confirmed-vs-random** (a Welch *t* against a drift-matched random-entry baseline) — the only honest test on an upward-drifting tape. The thesis axis is the **confirmation-candle placebo**: re-run on the bare harami (confirmation removed) and measure the third candle's marginal contribution. Tradability charges costs on every signal. A deterministic synthetic control with a *planted* three-inside-up bounce proves the detector is live (edge 0 → *t* = −1.18, no false positive; planted bounce → *t* = +4.43), so the flat/negative real-tape result is a genuine "nothing there".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a three-inside-up is, why a "confirmed" reversal on a rising market can still lose, the confirmed-vs-random race, and the confirmation-candle test — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | mechanical harami+confirmation, one-sample HAC *t*, the random-entry Welch test, the harami-only placebo (the thesis), per-ticker deltas, costs, and a synthetic planted-reversal control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`three_inside/`](three_inside/). Pattern read on closed bars; entry is the next close (one lag). Basket is surviving liquid ETFs — but this is a single-instrument pattern study, so the random-entry baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
