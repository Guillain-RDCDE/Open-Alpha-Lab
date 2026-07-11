# Study 675 — Qstick

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the zero-cross beat holding? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Qstick(8) up-cross vs a **drift-matched random** baseline: Δ = **−12.8 / −11.9 / −22.5 / +24.5 bps** at 5/10/20/60 days, and the cross-vs-random Welch *t* **never clears 2** (max |t| = 1.68, and negative — i.e. *worse* than a coin — at 5/10/20 days). Every one of 5 ETFs individually trails its own random baseline at 20 days. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No residual edge once the free drift is removed; a sign-scramble placebo shows the body **ordering** carries no information (**p = 0.814**). You'd capture the same drift more cheaply by **holding the index**. |
| **Just a slow trend proxy?** | ![Confirmed](https://img.shields.io/badge/Just_a_slow_trend_proxy%3F-Confirmed-8b949e?style=flat-square) | Qstick correlates at **r ≈ 0.78** (pooled basket) with a plain trailing-momentum proxy that never looks at the open — and even that simpler proxy barely (Welch *t* = 1.99 < 2) beats random. The open/close split adds no forecasting power over the trend it echoes. |

> **In one sentence:** Chande's Qstick — an 8-day average of close-minus-open, sold as a
> buying/selling-pressure gauge — fires 2,210 mechanical zero-cross trades across 5 ETFs over
> 21 years and **loses to buying on random days** at three of four horizons (never clearing
> *t* = 2 either way), because it is structurally a laggier read of ordinary price momentum
> (r ≈ 0.78 correlation) rather than an independent forecast: **None × Mirage**.

## What we tested

Tushar Chande's **Qstick** (Chande & Kroll, *The New Technical Trader*, 1994) smooths the daily
candle body — close minus open, here normalised by the prior close for cross-instrument
comparability — over an 8-day window (the original default), and the folklore reads its
**zero-cross** as a trend-timing trigger: cross up, buying pressure has taken over, buy. We
encode that literally — up-cross entered at the **next close** (one documented lag), forward
5/10/20/60-day returns on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return,
2005 → 2026) — and test it against the only honest baseline on an upward-drifting tape: a
**drift-matched random entry** (Welch *t*), plus a **sign-scramble placebo** that destroys the
body's time order while keeping its marginal. The third axis asks Chande's own defining claim
directly: is the close-minus-open split doing real work, or is Qstick just a slower, noisier
copy of trailing price momentum? We measure the correlation and race the naive momentum-only
cross against the same random baseline. A deterministic synthetic control with a *planted*,
mean-reverting buying-pressure factor proves the detector is live (null → t ≈ 0 across 20
seeds; planted lead → t = +5.12), so the flat real-tape result is a genuine "nothing there."
**Dedup:** siblings [423-force-index](../423-force-index/) (price-change × volume),
[473-balance-of-power](../473-balance-of-power/) (body normalised by the bar's *range*, not the
prior close), [185-chande-momentum](../185-chande-momentum/) (the same author's unrelated
up-sum/down-sum oscillator) and [129-heikin-ashi](../129-heikin-ashi/) (a recursively smoothed
*candle*, not a moving average of one statistic) never test Chande's specific
close-minus-open smoothing — this study does. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what Qstick is, why "buyers won today" sounds like a signal, the cross-vs-random race, and why it's really just a slower way to read the trend, in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the beta trap (one-sample vs Welch-vs-random), per-ticker deltas, the ordering placebo, the trend-proxy correlation and momentum-cross race, costs, and a synthetic planted-lead control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`qstick/`](qstick/). Qstick = SMA₈((close−open)/prior close); entry is the next close
(one lag). Basket is surviving liquid ETFs — but this is a single-instrument trend study, so the
random-entry baseline neutralizes the drift/survivorship. **Not investment advice** — research &
education. See [LICENSE](../../LICENSE).*
