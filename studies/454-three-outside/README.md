# Study 454 — Three-Outside-Up / -Down 🕯️🕯️🕯️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the pattern forecast? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The three-outside-up does **not** beat a drift-matched **random-entry** baseline at the desk's bar: pattern − random = **+0.6 / +7.0 / +64.3 / +15.5 bps** at 5/10/20/60 days, and the Welch *t* **never clears 2** (best **+1.82** at 20d, *p* = 0.069; 5/10/60d = +0.04 / +0.31 / +0.27). The big one-sample *t*'s (20d **+3.93**, 60d **+3.45**) are mostly **beta** — the upward drift every long-only entry inherits. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No residual edge once the free drift is removed; costs only deepen the hole. You'd capture the same drift more cheaply by **holding the index**. ~85 signals/name in 21 years — nothing to scale. |
| **"Does engulf-plus-confirm forecast?"** | ![Busted](https://img.shields.io/badge/Engulf%2Bconfirm_forecasts%3F-Busted-8b949e?style=flat-square) | Drop the confirming third bar and pick random engulfs and the result barely moves: **89%** of confirmation-blind engulf draws match or beat the real confirmed entries (*p* = **0.890**). The confirmation carries no information. |

> **In one sentence:** The three-outside-up looks convincing because indices drift up — encode it mechanically (strict engulf on bars t-2,t-1, strict confirm on bar t, no eyeballing) and fire the long rule **428 times** across 5 indices over 21 years, and it **fails to beat buying on random days** (best Welch *t* = +1.82 at 20d, *p* = 0.07) — and the confirmation-shuffle placebo leaves the result intact (*p* = 0.89): all tide, no tool.

## What we tested

We encode the tightest mechanical version a proponent would accept. A **bullish engulfing** is bars (t-2, t-1): a down candle, then an up candle whose real body fully covers it (strict open/close coordinates). The **three-outside-up** adds a **confirming** third bar t that closes above bar t-1's close; everything is known by the close of t (no look-ahead). A long fires at the **next close** (one documented lag), and we measure the forward 5/10/20/60-day return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The Signal axis is **pattern vs a drift-matched random-entry baseline** (a Welch *t*) — the only honest test on an upward-drifting tape — plus a **confirmation-shuffle placebo** that re-samples size-matched entries from the pool of *all* engulfs, ignoring the third bar (the direct "does the confirmation matter?" null). Tradability charges costs on every entry. A deterministic synthetic control with a *planted* multi-day continuation proves the detector is live (edge 0 → *t* ≈ 0; planted continuation → *t* = +7.90, win 75%), so the flat real-tape result is a genuine "nothing there".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a three-outside is, why a long-only rule on a rising market always looks good, the pattern-vs-random race, and the confirmation scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | mechanical engulf+confirm, one-sample HAC *t* vs the beta trap, the random-entry Welch test, the confirmation-shuffle placebo, per-ticker deltas, costs, and a synthetic planted-continuation control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`three_outside/`](three_outside/). The engulf is bars (t-2, t-1); the confirming bar is t (read on its close, no look-ahead); entry is the next close (one lag). Basket is surviving liquid ETFs — but this is a single-instrument pattern study, so the random-entry baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
