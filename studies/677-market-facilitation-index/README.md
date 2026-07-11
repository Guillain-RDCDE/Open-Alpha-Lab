# Study 677 — Market Facilitation Index 🚦

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do the bar colors predict what happens next? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Green's continuation score is **−0.85** Welch *t* on SPY (**−0.33** pooled across 6 tickers, ~50k state-days) and Squat's is **+0.54** (**+1.35** pooled) — both under **t = 2**, both on the *wrong* side of the claim, sign flips ticker-by-ticker, label-shuffle placebo *p* = 0.40 / 0.58. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The one tradeable version — long only on Green days — **loses significantly to buy-and-hold** (HAC *t* = −3.55 net, −3.14 even gross), and a volume-free SMA(50/200) crossover beats it outright with none of the machinery. |
| **"Green flags continuation, Squat flags reversal?"** | ![Busted](https://img.shields.io/badge/Green%2FSquat%3F-Busted-8b949e?style=flat-square) | Both named predictions fail on the tape, fail the placebo, flip sign across six independent instruments, and the "continuation" trading rule actively underperforms simply holding the index. |

> **In one sentence:** Bill Williams' BW-MFI four-color classifier — Green (MFI↑,
> Volume↑) for continuation, Squat (MFI↓, Volume↑) for reversal — clears no bar on 33
> years of SPY (and five other ETFs): both continuation scores sit under *t* = 2, both
> point the wrong way, the sign isn't even stable across tickers, and trading the
> "continuation" half **loses to buy-and-hold significantly**, even before costs.

## What we tested

We build the **BW-MFI** ratio `(High-Low)/Volume` on RAW daily bars for SPY and a
five-ETF basket (QQQ, DIA, IWM, XLE, GLD), cross its bar-to-bar direction against
volume's own bar-to-bar direction to classify every day into **green / fade / fake /
squat**, and test a **continuation score** — `sign(today's return) × tomorrow's
return` — by color against a 2,000-draw label-shuffle placebo, pooled across the basket
and checked ticker-by-ticker for a consistent sign. Two state-conditioned timers (ride
Green, sidestep Squat) race **NET** of one-way costs against buy-and-hold, one execution
lag (the color known at close *t* earns the return of *t+1*), with a Newey-West HAC *t*
and a sign-permutation placebo. A deterministic synthetic control with a **tunable
planted** continuation/reversal effect proves the harness would find this exact pattern
if it existed (*t* > 17 at a modest knob) — it doesn't, on any of the six real tapes.
**Dedup:** siblings [424-ease-of-movement](../424-ease-of-movement/) (the closest
relative — a range/volume ratio, but never crossed against volume's own direction),
[418-money-flow-index](../418-money-flow-index/) (unrelated "money flow" construction),
[423-force-index](../423-force-index/) (a continuous ΔClose×Volume oscillator, not a
4-way categorical state) and [676-gator-oscillator](../676-gator-oscillator/) (same
author, a moving-average construct with no volume term at all) test different
mechanisms; none tests BW-MFI's specific volume-direction-crossed-with-MFI-direction
claim. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the four bar colors are supposed to mean, why "in gear" sounds like it should work, and what actually happens the day after — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the classifier, the continuation-score Welch/placebo splits, the pooled and per-ticker replication check, the NET timer race with HAC *t* and cost sweep, the sign-permutation placebo, and the 20-seed synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`market_facilitation_index/`](market_facilitation_index/). Tapes are RAW OHLCV
(for the BW-MFI ratio) plus adjusted close (total return, for every return computation),
1993-02-01 → 2026-06-30. No survivorship — every tape is a single continuously-traded
ETF. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
