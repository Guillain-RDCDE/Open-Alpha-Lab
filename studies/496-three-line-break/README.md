# Study 496 — Three-Line-Break 🧱

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the 3-line flip forecast? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The "buy the up-reversal" rule does **not** beat a drift-matched **random-entry** baseline: reversal − random = **+0.2 / −28.4 / −22.4 / −32.5 bps** at 5/10/20/60 days, and the reversal-vs-random Welch *t* **never clears 2** (max **+0.02** at 5d, *p* = 0.986). The big one-sample *t*'s (20d **+5.29**, 60d **+6.26**) are **pure beta** — the upward drift every long-on-strength entry inherits. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No residual edge once the free drift is removed; costs only deepen the hole, and the rule sits *out* of the market on every down-flip. You'd capture the same drift more cheaply by **holding the index**. Nothing to scale. |
| **"Does the 3-line reversal forecast?"** | ![Busted](https://img.shields.io/badge/3--line_reversal_forecasts%3F-Busted-8b949e?style=flat-square) | Shuffle the *order* of the daily moves (same returns, scrambled sequence) and the result barely moves: **26%** of scrambled-sequence tapes match or beat the real one (*p* = **0.261**). The specific brick flips carry no information. |

> **In one sentence:** Three-Line-Break looks uncanny because indices drift up — encode it mechanically (break number 3, causal bricks, no eyeballing) and fire the "buy the bullish reversal" rule 943 times across 5 indices over 21 years, and it **ties or loses to buying on random days** at every horizon (and the sequence placebo leaves the result untouched, *p* = 0.26): all tide, no tool.

## What we tested

We encode the tightest mechanical version a proponent would accept. TLB bricks are built **causally** from past closes only (a new line on a close past the prior brick; the chart flips **down→up** only when the close breaks the top-extreme of the **3** latest down-lines — no look-ahead); a long fires on each up-reversal bar, entered at the **next close** (one documented lag), and we measure the forward 5/10/20/60-day return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The Signal axis is **reversal vs a drift-matched random-entry baseline** (a Welch *t*) — the only honest test on an upward-drifting tape — plus a **shuffled-returns sequence placebo** that destroys the brick order while keeping the price marginal. Tradability charges costs on every reversal. A deterministic synthetic control with a *planted* post-reversal continuation proves the detector is live (edge 0 → *t* ≈ 0; planted continuation → *t* = +6.13), so the flat real-tape result is a genuine "nothing there".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a Three-Line-Break chart is, why a long-on-strength rule on a rising market always looks good, the reversal-vs-random race, and the sequence scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | causal TLB bricks, one-sample HAC *t* vs the beta trap, the random-entry Welch test, the shuffled-returns placebo, per-ticker deltas, costs, and a synthetic planted-continuation control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`three_line_break/`](three_line_break/). Bricks are built causally from past closes (break number 3); entry is the next close (one lag). Basket is surviving liquid ETFs — but this is a single-instrument trend study, so the random-entry baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
