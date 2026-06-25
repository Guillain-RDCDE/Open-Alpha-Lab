# Study 490 — Arms Index (TRIN) ⚖️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does buying the panic beat random? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The high-TRIN panic entry **beats** a drift-matched **random-entry** baseline at every horizon (Δ = **+29.4 / +27.6 / +33.3 / +48.2 bps** at 5/10/20/60 days) and the timing placebo is significant — so it is **not** pure beta. But the panic-vs-random Welch *t* **never clears 2** (max **+1.53** at 5d, *p* = 0.126). A real lean, statistically soft. |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | A ~25–30 bps premium over random survives a 1 bp round-trip cost almost untouched, but it rests on ~460 **clustered** crash-period entries, never reaches *t* = 2, and is largely the generic volatility rebound. A lean for a contrarian timer, not a scalable stand-alone edge. |
| **"Does TRIN forecast short-term turns?"** | ![Mixed](https://img.shields.io/badge/Forecasts_turns%3F-Mixed-dab617?style=flat-square) | The shuffled-**timing** placebo says *yes* — the placement of high-TRIN days carries information ahead of rebounds (*p* = **0.001–0.041** across horizons). The harder vs-random test says *not strongly*. Real structure, weak forecast. |

> **In one sentence:** The Arms Index is the rare technical gauge that **isn't** just beta in a costume — buy the panic (a high-TRIN breadth spike) and you really do beat random days by ~30 bps, and scrambling *when* the panics fall destroys the edge (placebo *p* ≤ 0.04), so the timing carries real information — but the panic-minus-random gap never clears *t* = 2 and most of it is the ordinary "bounce after a crash day," so it lands a genuinely honest **Weak × Fragile × Mixed**.

## What we tested

We encode the tightest mechanical version a proponent would accept. True exchange breadth is unavailable offline, so we build a **breadth-proxy TRIN** from a 5-ETF basket (SPY QQQ IWM DIA GLD): each ETF is one "issue" (advances if up that day), its move magnitude proxies issue volume, and `TRIN = (advancers/decliners)/(up-move/down-move)` (regularised so quiet days don't blow up the ratio — the top-TRIN days land exactly on the real washouts, Mar 2020 & Dec 2008). A long fires when TRIN exceeds its **90th percentile** (the panic tail), entered at the **next close** (one documented lag), and we measure the forward 5/10/20/60-day return on SPY (yfinance daily total-return, 2005→2026, 461 panic entries). The Signal axis is **panic vs a drift-matched random-entry baseline** (a Welch *t*) — the only honest test on an upward-drifting tape — plus a **shuffled-TRIN timing placebo** that scrambles *when* the panic days fall while keeping the marginal. Tradability charges costs on every entry. A deterministic synthetic control with a *planted* post-panic bounce proves the detector is live (edge 0 → *t* ≈ 0 across seeds; planted bounce → *t* = +2.72), so the weak real-tape result is an honest "a little something," not a dead pipeline. **Breadth caveat:** a 5-issue proxy is far coarser than the 3000-issue NYSE TRIN — that caps the test, and the verdict is stated against that cap.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what TRIN is, why "buy the panic" sounds smart, the panic-vs-random race, and the timing scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | breadth-proxy TRIN, one-sample HAC *t* vs the beta trap, the random-entry Welch test, the threshold sweep, the shuffled-timing placebo, costs, and a synthetic planted-bounce control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`arms_index_trin/`](arms_index_trin/). TRIN is a 5-issue breadth proxy (|return| as volume, regularised); the panic entry is the next close (one lag). The Signal axis races the panic against a drift-matched random baseline; the timing placebo scrambles when the panic days fall. **Breadth proxy ≠ true NYSE TRIN — this caps the test. Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
