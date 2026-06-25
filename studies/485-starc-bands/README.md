# Study 485 — STARC Bands 📊

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the band touch revert? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The "buy the lower band" rule does **not** beat a drift-matched **random-entry** baseline where reversion should be strongest: pierce − random = **−7.1 / −28.6 / +70.4 / +258.8 bps** at 5/10/20/60d, with Welch *t* **negative at 5–10d** and not significant at 20d (*p* = 0.145). It clears *t* ≥ 2 only at **60 days** (Welch *t* = **+3.34**) — one horizon out of four, a long-hold drift effect, with an incoherent cross-section (Δ negative in QQQ/IWM). The big one-sample *t*'s (20d **+2.70**, 60d **+5.54**) are mostly beta. |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | The only horizon that beats random (60d) survives neither the geometry placebo (the bands are irrelevant) nor the cross-section, and is a drift effect you'd capture more cheaply by **holding the index**. A number clears the bar, but nothing robust to scale. |
| **"Does the band touch forecast reversion?"** | ![Busted](https://img.shields.io/badge/Forecast_reversion%3F-Busted-8b949e?style=flat-square) | Permute the **ATR** so the band half-widths are random (SMA + marginal kept) and the result is untouched: **96%** of scrambled-width envelopes match or beat the real one (*p* = **0.964**). The ATR scaling — the whole point of STARC — carries no information. |

> **In one sentence:** STARC bands look like a clean volatility channel — a 6-day SMA flanked by ±2·ATR — but encode "buy the close below the lower band" mechanically and fire it 241 times across 5 ETFs over 21 years, and it **loses to buying on random days** at 5–10 days (where reversion should live), wins only at a 60-day drift horizon, and the ATR scaling is **irrelevant** (scramble it, *p* = 0.96): the bands describe volatility, they don't forecast the bounce.

## What we tested

We encode the tightest mechanical STARC rule a proponent would accept. The center is a **6-bar SMA** of the close; the bands are **SMA ± 2·ATR(15)** with Wilder's ATR — all **causal** (trailing closes/ranges only, no look-ahead). A long fires on the first close **below the lower band**, entered at the **next close** (one documented lag), and we measure the forward 5/10/20/60-day return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The Signal axis is **pierce vs a drift-matched random-entry baseline** (a Welch *t*) — the only honest test on an upward-drifting tape — plus a **shuffled-ATR placebo** that randomises the band widths while keeping the SMA and price marginal. Tradability charges costs on every pierce. A deterministic synthetic control with a *planted* lower-band reversion proves the detector is live (edge 0 → *t* = +0.89, win 54%; planted bounce → *t* = +2.07, win 61%), so the flat short-horizon real-tape result is a genuine "nothing there".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a STARC band is, why a dip-buy on a rising market always looks good, the pierce-vs-random race, and the ATR scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | causal SMA/ATR bands, one-sample HAC *t* vs the beta trap, the random-entry Welch test, the shuffled-ATR placebo, per-ticker deltas, costs, and a synthetic planted-bounce control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`starc_bands/`](starc_bands/). Bands are causal (SMA(6) ± 2·ATR(15), Wilder); entry is the next close (one lag). Basket is surviving liquid ETFs — but this is a single-instrument channel study, so the random-entry baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
