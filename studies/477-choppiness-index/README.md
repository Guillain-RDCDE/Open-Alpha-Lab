# Study 477 — Choppiness Index 🌊📏

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a low CI time a tradable trend? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The "buy when CI goes low" rule does **not** beat a drift-matched **random-entry** baseline: low-CI − random = **+17.7 / +0.3 / −7.7 / +43.8 bps** at 5/10/20/60 days, and the low-CI-vs-random Welch *t* **never clears 2** (max **+1.54** at 5d, *p* = 0.124). The big one-sample *t*'s (20d **+5.56**, 60d **+7.93**) are **pure beta** — the upward drift every long entry inherits, made worse by the fact that the CI is sign-blind. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No residual edge once the free drift is removed; costs only deepen the hole. You'd capture the same drift more cheaply by **holding the index**. Nothing to scale. |
| **"Does CI forecast trend vs chop usefully?"** | ![Busted](https://img.shields.io/badge/Forecasts_trend_vs_chop%3F-Busted-8b949e?style=flat-square) | Scramble the CI's trend-vs-chop structure (return-shuffled placebo) and the result barely moves: **41%** of nonsense CIs match or beat the real one (*p* = **0.409**). The specific regime reading carries no information. |

> **In one sentence:** The Choppiness Index looks insightful because indices drift up — encode it mechanically (trailing-only CI, no eyeballing) and fire the "low CI ⇒ a trend is starting, go long" rule 936 times across 5 indices over 21 years, and it **merely ties buying on random days** (Welch *t* never clears 2; the structure placebo leaves the result untouched, *p* = 0.41): all tide, no gauge.

## What we tested

We encode the tightest mechanical version a proponent would accept. The Choppiness Index is computed on a **trailing** N = 14 window — `CI = 100·log₁₀(Σ TR / (max high − min low)) / log₁₀(N)`, bounded 0–100, using only bars through *t* (no look-ahead); a long fires on the **onset** of a low-CI reading (the first bar CI drops below **38.2**, the canonical "trending" band), entered at the **next close** (one documented lag), and we measure the forward 5/10/20/60-day return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The Signal axis is **low-CI vs a drift-matched random-entry baseline** (a Welch *t*) — the only honest test on an upward-drifting tape, doubly so because the CI is **sign-blind** — plus a **return-shuffled placebo** that destroys the trend-vs-chop structure while keeping the price marginal. Tradability charges costs on every entry. A deterministic synthetic control with a *planted* "low-CI ⇒ momentum" structure proves the detector is live (edge 0 → *t* ≈ 1; planted momentum → *t* = +7.47), so the flat real-tape result is a genuine "nothing there".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the Choppiness Index is, why a long rule on a rising market always looks good, the low-CI-vs-random race, and the structure scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | trailing CI, one-sample HAC *t* vs the beta trap, the random-entry Welch test, the return-shuffled placebo, per-ticker deltas, costs, and a synthetic planted-momentum control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`choppiness_index/`](choppiness_index/). CI is trailing-only (N = 14, low-CI band 38.2); entry is the next close (one lag). Basket is surviving liquid ETFs — but this is a single-instrument regime study, so the random-entry baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
