# Study 497 — Woodie's Pivot Points 🪝

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does S1 support price? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The "buy the S1 support" rule does **not** beat a drift-matched **random-entry** baseline: touch − random = **+2.7 / +2.2 / −8.1 / −4.4 bps** at 5/10/20/60 days, and the touch-vs-random Welch *t* **never clears 2** (max **+0.59** at 5d, *p* = 0.556). The huge one-sample *t*'s (20d **+7.78**, 60d **+10.23**) are **pure beta** — an S1-touch fires on most sessions, so the rule is essentially buy-and-hold the upward-drifting index. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No residual edge once the free drift is removed; costs only deepen the hole. You'd capture the same drift more cheaply (and more fully) by **holding the index**. Nothing to scale. |
| **"Do Woodie levels hold better than random?"** | ![Busted](https://img.shields.io/badge/Woodie_levels_hold%3F-Busted-8b949e?style=flat-square) | Swap S1 for a *randomly-placed* support of the same touch frequency and the result barely moves: **66%** of random levels match or beat the real Woodie line (*p* = **0.655**). The specific close-weighted level carries no information. |

> **In one sentence:** Woodie's pivot looks like real support because it sits just below price on a market that drifts up — encode it mechanically (P = (H+L+2C)/4, S1 = 2P−H from yesterday's bar, one lag) and fire the "buy the S1 touch" rule 7,002 times across 5 indices over 21 years, and it's a **dead heat with buying on random days** (and a random support level does just as well, *p* = 0.66): all tide, no tool.

## What we tested

We encode the tightest mechanical version a proponent would accept. Woodie's pivot double-weights the close — **P = (H + L + 2C)/4**, with support **S1 = 2P − H** — computed from **yesterday's** (H, L, C) (a documented one-day lag, knowable at today's open). A long fires when today's **low pierces the prior-day S1** (the "support holds" setup), entered at the **next close** (one more lag), and we measure the forward 5/10/20/60-day return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The Signal axis is **touch vs a drift-matched random-entry baseline** (a Welch *t*) — the only honest test on an upward-drifting tape, and doubly so here because the S1-touch fires on a majority of sessions — plus a **random-level placebo** that swaps S1 for a randomly-placed support while keeping the touch frequency and price marginal. Tradability charges costs on every touch. A deterministic synthetic control with a *planted* S1 bounce proves the detector is live (edge 0 → *t* = +0.56; planted bounce → *t* = +18.62), so the flat real-tape result is a genuine "nothing there".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a pivot point is, why a near-always-long buy on a rising market looks good, the touch-vs-random race, and the random-level swap — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the Woodie formula, one-sample HAC *t* vs the beta trap, the random-entry Welch test, the random-level placebo, per-ticker deltas, costs, and a synthetic planted-bounce control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`woodie_pivots/`](woodie_pivots/). Levels use the prior bar (one lag); entry is the next close (one more lag). Basket is surviving liquid ETFs — but this is a single-instrument level study, so the random-entry baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
