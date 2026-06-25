# Study 475 — DeMarker 📉➡️📈

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the oversold turn pay? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The "buy the DeMarker rising out of <0.3" rule beats a drift-matched **random-entry** baseline by a positive but small margin at *every* horizon (entry − random = **+9.0 / +25.3 / +55.8 / +63.1 bps** at 5/10/20/60 days) and the entry-vs-random Welch *t* **clears 2 at exactly one horizon** (20d **+2.61**, *p* = 0.009); 5/10/60d are +0.76/+1.60/+1.80, never significant. More than the usual mirage — but single-horizon. |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | The lone significant horizon is carried by **IWM**: drop it and the 20-day Welch *t* falls from **+2.61 to +1.40** (*p* = 0.162). One horizon, one ticker — exactly the horizon×ticker multiplicity that fails out of sample. Nothing robust to scale. |
| **"Forecasts exhaustion turns?"** | ![Mixed](https://img.shields.io/badge/Forecasts_exhaustion%3F-Mixed-dab617?style=flat-square) | Phase-scramble the oscillator's timing (rotate the DeMax/DeMin streams) and the result *does* dent: the real DeMarker beats **95.6%** of scrambles (*p* = **0.044**, just under 0.05), and every ticker's delta is positive. So the timing is *marginally* load-bearing — not the flat nothing of a busted indicator, but not a confirmed exhaustion signal either. |

> **In one sentence:** DeMark's DeMarker is one of the rare chart tools that *doesn't* simply lose to random — the oversold turn beats a drift-matched dart at every horizon and clears *t* ≥ 2 at 20 days — but the edge is **weak and fragile**: it's significant at one of four horizons, carried by one of five tickers (drop IWM → *p* = 0.16), and the timing only *just* survives a phase-scramble (*p* = 0.044). A faint, real-looking nudge wrapped in beta, not a deployable signal.

## What we tested

We encode the tightest mechanical version a proponent would accept. The **DeMarker** (DeMark, period 14) is built from the highs and lows — DeMax = max(ΔHigh, 0), DeMin = max(−ΔLow, 0), DeMarker = SMA(DeMax) / (SMA(DeMax) + SMA(DeMin)) ∈ [0, 1] — using bars through *t* only (no look-ahead). A long fires the first bar the DeMarker was below **0.3** yesterday and turns **up** today (rising out of oversold), entered at the **next close** (one documented lag), and we measure the forward 5/10/20/60-day return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026, **1167 entries**). The Signal axis is **entry vs a drift-matched random-entry baseline** (a Welch *t*) — the only honest test on an upward-drifting tape — plus a **phase-scramble placebo** that rotates the oscillator's inputs off their days (same readings, wrong timing) and a **per-ticker / drop-IWM** robustness check. Tradability charges costs on every entry. A deterministic synthetic control with a *planted* exhaustion bounce keyed to the trigger proves the detector is live and unbiased (edge 0 → *t* ≈ 0 averaged over seeds; planted bounce → *t* = +9.65, win 84%), so the fragile real-tape result is honestly measured.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the DeMarker is, why a dip-buy on a rising market always looks good, the entry-vs-random race, and the one-ticker fragility — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the mechanical oscillator, one-sample HAC *t* vs the beta trap, the random-entry Welch test, the per-ticker / drop-IWM robustness, the phase-scramble timing placebo, costs, and a synthetic planted-bounce control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`demarker/`](demarker/). DeMarker period 14, oversold 0.30; the oscillator uses bars through *t* only; entry is the next close (one lag). Basket is surviving liquid ETFs — but this is a single-instrument timing study, so the random-entry baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
