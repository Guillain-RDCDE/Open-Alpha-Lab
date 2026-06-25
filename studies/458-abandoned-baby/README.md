# Study 458 — Abandoned-Baby 👶

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the island doji beat random? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The abandoned baby's pattern-minus-random delta is **positive at every horizon** (**+34 / +35 / +47 / +288 bps** at 5/10/20/60 days) and clears the desk's *t* ≥ 2 bar **at 60 days only** (Welch *t* = **+2.58**, *p* = 0.011). A genuine flicker — more than most chart tools — but isolated to one horizon and riding just **80 trades**; 5/10/20-day Welch *t* ≤ 1.0. |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | One significant horizon, ~16 trades per name across two decades, and the sign **flips by instrument** (QQQ −259, GLD −29 vs SPY +169, DIA +266 bps at 20d). Something's there; nothing scalable or deployable. |
| **"Does the island doji forecast a turn?"** | ![Busted](https://img.shields.io/badge/Island_doji_forecasts%3F-Busted-8b949e?style=flat-square) | Drop the island gaps (gap-scramble placebo: keep a doji-after-a-decline, draw entries without the gaps) and the result is intact — **31%** of geometry-free draws match or beat the real island (*p* = **0.307**), and the gaps "matter" in only **1 of 5** names at 60d. The *abandoned* part carries no information. |

> **In one sentence:** The abandoned baby — a doji marooned by gaps on both sides, candlestick lore's rarest reversal — actually *does* edge out a drift-matched random baseline (unlike most chart tools), but only convincingly at 60 days, on 80 trades, with the sign flipping between instruments; and the gap-scramble placebo (*p* = 0.31) proves the **island gaps add nothing** — so it lands a weak, fragile flicker whose *island* thesis is busted.

## What we tested

We encode the tightest mechanical version a proponent would accept. A **doji** is a body ≤ 10% of the bar's high-low range; the bullish version requires a **prior decline** (the bar before the doji is a down candle below its SMA-20); the **island** is a body-gap *down* to the doji and a body-gap *up* to a confirmation candle that closes higher. The pattern is confirmed on the close of the confirmation bar and entered at the **next close** (one documented lag); we measure the forward 5/10/20/60-day return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026 — **80** pooled islands). The Signal axis is **pattern vs a drift-matched random-entry baseline** (a Welch *t*) — the only honest test on an upward-drifting tape — plus a **gap-scramble geometry placebo** that keeps the doji-after-a-decline pool but destroys the island gaps. Tradability charges costs on every trade. A deterministic synthetic control with *planted* island bottoms proves the detector is live (edge 0 → *t* = −0.19; planted islands → *t* = +5.96), so the thin, horizon-isolated real-tape result is an honest "almost nothing there".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what an abandoned baby is, why a rare dip-buy on a rising market looks good, the pattern-vs-random race, and the gap scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | mechanical island detection, one-sample HAC *t* vs the beta trap, the random-entry Welch test (one horizon clears 2), the gap-scramble placebo, per-ticker sign-flips, costs, and a synthetic planted-island control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`abandoned_baby/`](abandoned_baby/). Doji body ≤ 10% of range; prior down-candle below SMA-20; body-gap island both sides; entry is the next close (one lag). Basket is surviving liquid ETFs — but this is a single-instrument pattern study, so the random-entry baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
