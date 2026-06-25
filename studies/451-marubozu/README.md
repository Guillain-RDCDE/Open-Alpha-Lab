# Study 451 — Marubozu 🕯️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the wickless body forecast continuation? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The "buy the bullish marubozu" rule does **not** beat a drift-matched **random-entry** baseline — it *loses to it at every horizon*: marubozu − random = **−40.2 / −31.1 / −44.1 / −73.3 bps** at 5/10/20/60 days, and the marubozu-vs-random Welch *t* is **negative everywhere** (worst-to-best −1.37 … −0.57). Even the one-sample *t* never clears **+0.27** — the pattern is too rare (only **73** in 21 years) and too unremarkable to carry even the usual beta. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No residual edge — in fact *negative* versus a dart-throwing entry — and costs only deepen the hole. With ~73 strict marubozus across 5 tapes in 21 years there is nothing to scale; you'd capture more by **holding the index**. |
| **"Does the no-wick body forecast?"** | ![Busted](https://img.shields.io/badge/No--wick_body_forecasts%3F-Busted-8b949e?style=flat-square) | Scatter the marubozu *label* onto random bars (body-shuffle placebo) and the result is intact: **79%** of randomly-labelled sets match or beat the real one (*p* = **0.788**). The wickless shape carries no information. |

> **In one sentence:** the marubozu looks decisive because a tall green wickless candle screams conviction — encode it mechanically (body ≥ 95% of the range, wicks ≤ 2%) and fire the "buy the bullish marubozu" rule 73 times across 5 indices over 21 years, and it **loses to buying on random days** at *every* horizon (and the geometry placebo leaves the result untouched, *p* = 0.79): a vivid description of yesterday, not a forecast of tomorrow.

## What we tested

We encode the tightest mechanical version a proponent would accept. A **bullish marubozu** is an up-bar (close > open) whose real body fills **≥ 95%** of the high-low range and whose upper *and* lower wicks are each **≤ 2%** of the range — a genuinely wickless candle, computed only from the bar's own OHLC (so the *detection* has no look-ahead). A long fires on the marubozu, entered at the **next close** (one documented lag — the marubozu bar's own large return is never harvested), and we measure the forward 5/10/20/60-day return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The Signal axis is **marubozu vs a drift-matched random-entry baseline** (a Welch *t*) — the only honest test on an upward-drifting tape, since a marubozu is by construction a big up-day — plus a **body-shuffle placebo** that scatters the marubozu label onto random bars while keeping the price marginal. Tradability charges costs on every trade. A deterministic synthetic control with a *planted* marubozu-continuation proves the detector is live (edge 0 → *t* = +0.29; planted continuation → *t* = +17.86, win 95%), so the flat real-tape result is a genuine "nothing there".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a marubozu is, why a big-green-day rule on a rising market looks fine, the marubozu-vs-random race, and the label scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the body-fill rule, one-sample HAC *t* vs the (barely-firing) beta trap, the random-entry Welch test, the body-shuffle placebo, per-ticker deltas, costs, and a synthetic planted-continuation control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`marubozu/`](marubozu/). A bullish marubozu = up-bar with body ≥ 95% of range and wicks ≤ 2%; read on the bar's own close, entered the next close (one lag). Basket is surviving liquid ETFs — but this is a single-instrument event study, so the random-entry baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
