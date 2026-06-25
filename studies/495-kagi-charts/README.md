# Study 495 — Kagi Charts 〽️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the yang switch forecast? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The "buy the yang (thick) switch" rule does **not** beat a drift-matched **random-entry** baseline: switch − random = **+11.9 / −30.8 / +55.6 / +134.3 bps** at 5/10/20/60 days, and the switch-vs-random Welch *t* **never clears 2** (max **+1.27** at 60d, *p* = 0.205). The one-sample *t* that nudges up with horizon (60d **+2.84**) is **drift** — the upward climb every long-only entry inherits. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No residual edge once the free drift is removed; the rule sits out on yin and pays costs on every switch. You'd capture the same drift more cheaply by **holding the index**. Nothing to scale. |
| **Do Kagi reversals time the market?** | ![Busted](https://img.shields.io/badge/Time_the_market%3F-Busted-8b949e?style=flat-square) | Re-parameterise the Kagi's reversal into a different (equally valid) threshold and the result barely moves: **55%** of randomly-parameterised Kagis match or beat the real 4% one (*p* = **0.545**). The specific shoulders and waists carry no information. |

> **In one sentence:** the Kagi chart looks decisive because its thick "yang" line shows up in trends — and indices trend up — so encode it mechanically (4% reversal, shoulders/waists, no eyeballing), fire the "buy the yang switch" rule 144 times across 5 indices over 21 years, and it **fails to beat buying on random days** (Welch *t* tops out at +1.27), while re-drawing the Kagi with a different reversal leaves the result untouched (*p* = 0.55): all tide, no tool.

## What we tested

We encode the tightest mechanical version a proponent would accept. The **Kagi line** is built
bar-by-bar from closes (no future data): it extends in the current direction and **reverses on a
4% counter-move**, recording a **shoulder** at each up→down turn and a **waist** at each down→up
turn. The line switches to **yang** (thick) the bar it breaks above the prior shoulder and to
**yin** (thin) when it breaks below the prior waist. A long fires on the **yang switch** close,
entered at the **next close** (one documented lag), and we measure the forward 5/10/20/60-day
return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The Signal axis is
**switch vs a drift-matched random-entry baseline** (a Welch *t*) — the only honest test on an
upward-drifting tape — plus a **threshold-scramble placebo** that re-draws the Kagi with a random
reversal (1%–8%) while keeping the price marginal. Tradability charges costs on every switch. A
deterministic synthetic control with a *planted* post-yang-switch momentum burst proves the
detector is live (edge 0 → *t* = +1.25; planted burst → *t* = +7.47), so the flat real-tape
result is a genuine "nothing there".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a Kagi chart is, why a thick line in a rising market always looks smart, the yang-switch-vs-random race, and the reversal scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the mechanical Kagi line, one-sample HAC *t* vs the drift trap, the random-entry Welch test, the threshold-scramble placebo, per-ticker deltas, costs, and a synthetic planted-momentum control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`kagi_charts/`](kagi_charts/). The Kagi line is causal (closes up to *t* only), reverses on a 4% counter-move, switches thickness at prior shoulders/waists; entry is the next close (one lag). Basket is surviving liquid ETFs — but this is a single-instrument trend study, so the random-entry baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
