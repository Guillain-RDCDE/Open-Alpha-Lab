# Study 474 — Accelerator Oscillator ⚡

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does acceleration channel price? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The "two green bars above zero" rule does **not** beat a drift-matched **random-entry** baseline: AC − random = **+0.5 / +19.0 / +8.1 / −4.0 bps** at 5/10/20/60 days, and the AC-vs-random Welch *t* **never clears 2** (best **+1.73** at 10d, *p* = 0.083 — it *reverses* to −0.16 by 60d). The big one-sample *t*'s (20d **+6.50**, 60d **+7.13**) are **pure beta** — the upward drift every long-only entry inherits. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No residual edge once the free drift is removed; costs only deepen the hole. You'd capture the same drift more cheaply by **holding the index**. Nothing to scale. |
| **"Does acceleration forecast price?"** | ![Busted](https://img.shields.io/badge/Forecasts_price%3F-Busted-8b949e?style=flat-square) | Circularly **rotate** the AC series relative to price (keeping its values exactly) and the result barely moves: **89%** of meaningless time-shifts match or beat the real one (*p* = **0.886**). The AC-to-price alignment carries no information. |

> **In one sentence:** Bill Williams' Accelerator Oscillator looks predictive because it's a smoothed second derivative of a market that drifts up — encode it mechanically (AC = AO − SMA5(AO), trailing only) and fire the "two green bars above zero" rule 2,131 times across 5 indices over 21 years, and it **ties buying on random days** at every horizon (Welch *t* never clears 2; rotate the AC and the edge is untouched, *p* = 0.89): all tide, no tool.

## What we tested

We encode the canonical Williams rule. The Awesome Oscillator is AO = SMA5(median) − SMA34(median); the Accelerator is **AC = AO − SMA5(AO)**, the second derivative of price — all SMAs trailing, so AC at bar *t* uses only bars ≤ *t* (no look-ahead). A long fires on the first bar where **AC turns up** (two consecutive rising bars: AC[t] > AC[t−1] > AC[t−2]) **and AC > 0** (the classic "don't buy with a red bar"), entered at the **next close** (one documented lag), and we measure the forward 5/10/20/60-day return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The Signal axis is **AC-up vs a drift-matched random-entry baseline** (a Welch *t*) — the only honest test on an upward-drifting tape — plus a **rotated-AC placebo** that circularly shifts the AC series relative to price, destroying the timing while keeping its values exactly. Tradability charges costs on every entry. A deterministic synthetic control with a *planted* acceleration episode proves the detector is live (edge 0 → *t* = −0.67; planted acceleration → *t* = +5.33), so the flat real-tape result is a genuine "nothing there".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what acceleration even means, why a momentum rule on a rising market always looks good, the AC-vs-random race, and the AC-rotation scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the AO/AC arithmetic, one-sample HAC *t* vs the beta trap, the random-entry Welch test, the rotated-AC placebo, per-ticker deltas, costs, and a synthetic planted-acceleration control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`accelerator_oscillator/`](accelerator_oscillator/). AC = AO − SMA5(AO), AO = SMA5(median) − SMA34(median), all trailing; entry is the next close (one lag). Basket is surviving liquid ETFs — but this is a single-instrument momentum study, so the random-entry baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
