# Study 674 — VIDYA 📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does VIDYA timing beat holding? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | VIDYA(14, cmo=9)'s daily active spread vs buy-and-hold is **−3.10 bps/day at HAC *t* = −3.54** on SPY (gross *t* = −3.04), negative on **all five** basket tapes (3/5 individually *t* ≤ −2), both sample halves, and **every** CMO lookback from 5 to 30 bars. A position-shuffle permutation gives ***p* = 0.9885** — the timing is *worse* than random. |
| **Tradability** — could you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Net Sharpe **0.387 vs buy-and-hold 0.647** (gross 0.493 — still behind). No cost level tested rescues it, there's no break-even, and the long/short flip is worse (Sharpe **−0.198**). |
| **Speeds up in volatile/trending regimes?** | ![Mixed](https://img.shields.io/badge/Speeds_up_in_volatile%2Ftrending_regimes%3F-Mixed-8b949e?style=flat-square) | *Trending* holds: VI (the speed knob) correlates **+0.38** with trend strength, and a step-response confirms VIDYA freezes flat then matches an EMA's catch-up once a move saturates its CMO. *Volatile* is **busted**: VI correlates **−0.10** with realized volatility — the high-vol tercile's mean speed is *lower*, not higher, than the low-vol tercile's. |

> **In one sentence:** Chande's VIDYA genuinely does what its formula says — it freezes flat when `|CMO|` is near zero and speeds up to EMA-like tracking once a *trend* (not a *volatile* period — those correlate the wrong way, *t* = −0.10) saturates its own oscillator, and that honestly fires 37-40% fewer whipsaws than a plain SMA/EMA — but turned into a price-cross timing rule it still loses to buy-and-hold by 3.1 bps/day at *t* = −3.54, loses to a random reshuffle of its own calls (*p* = 0.99) on every one of five tickers, and its "better than a plain MA" edge never clears *t* = 2 on any of ten head-to-head checks, at any CMO-period setting.

## What we tested

Tushar Chande's 1992 "Adapting Moving Averages to Market Volatility": an EMA whose
smoothing constant is scaled by `VI = |CMO|/100` (his own Chande Momentum Oscillator,
Study 185), pitched to "speed up in volatile, trending markets and slow to a near-freeze
in quiet ones" — beating a fixed SMA/EMA of the same length. We split that into two
honest checks: **the mechanism itself** (does `VI` actually track realized volatility,
and separately trend strength — Chande's own pitch conflates the two — via correlation,
tercile splits, a deterministic step response and a real-tape tracking-distance check),
then **the trading claim**: a daily long/flat price-cross rule on VIDYA(14, cmo=9), net
of one-way costs × NAV with one execution lag, raced against the equivalent **SMA(14)**
and **EMA(14)** rules and against **buy-and-hold** on a five-ticker liquid basket — SPY,
QQQ, AAPL, MSFT, XLE, daily total-return bars to 2026-06-30. The Signal axis uses a
Newey-West HAC *t* on the daily active spread, a 2,000-draw position-shuffle permutation,
and a CMO-lookback robustness sweep (5→30 bars, the free knob distinct from the base
period). A deterministic synthetic tape with a *planted* trend is the positive control
proving the harness banks an edge when one exists. **Dedup:**
[185-chande-momentum](../185-chande-momentum/) (CMO itself as a trading signal, not as
VIDYA's internal speed knob), [433-kama-adaptive](../433-kama-adaptive/) (its closest
cousin — an efficiency-ratio-driven adaptive EMA that *increases* turnover, the opposite
of VIDYA), [672-mcginley-dynamic](../672-mcginley-dynamic/) (a different self-adjusting
brake that also cuts whipsaws) and [673-t3-tillson](../673-t3-tillson/) (a fixed-coefficient
stack with no state-dependent adaptation at all) — none decomposes the "volatile" claim
from the "trending" claim the way this study does.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "adapts to volatility" actually means, why the mechanism check splits the claim in two, the head-to-head race vs SMA/EMA and vs just holding, in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the VI-vs-volatility and VI-vs-trend correlation/tercile checks, the step-response and tracking-distance mechanism tests, the HAC-*t* active-spread race, the VIDYA-minus-SMA/EMA head-to-head, the permutation placebo, cost/per-instrument/split/CMO-period robustness, and the planted-trend synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`vidya/`](vidya/). Real tape is Yahoo daily, `auto_adjust=True` (total-return), as-of 2026-06-30. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
