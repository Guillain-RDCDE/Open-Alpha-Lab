# Study 493 — New-Highs-New-Lows 📈

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the breadth thrust forecast? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The "buy the breadth thrust" rule does **not** beat a drift-matched **random-entry** baseline: thrust − random = **−34.0 / −58.2 / −88.8 / −86.7 bps** at 5/10/20/60 days, and the thrust-vs-random Welch *t* is **negative at every horizon** (max **−0.92** at 60d, *p* = 0.358). The one-sample *t*'s (60d **+2.94**) are **pure beta** — the upward drift every long entry inherits, plus breadth's mechanical lag behind price. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No residual edge once the free drift is removed — the thrust actually *underperforms* random entries, and costs only deepen the hole. You'd capture the same drift more cheaply by **holding the index**. Nothing to scale. |
| **"Does the NH-NL line lead the index?"** | ![Busted](https://img.shields.io/badge/Leads_the_index%3F-Busted-8b949e?style=flat-square) | Scramble the cross-sectional breadth structure (shuffled-membership placebo) and the result barely moves: **82%** of nonsense breadth lines match or beat the real one (*p* = **0.818**). The NH-NL aggregation carries no information. |

> **In one sentence:** The new-highs/new-lows line looks like it "leads" because indices drift up *and* breadth is mechanically high right after a rally — encode it mechanically (52-week extremes, a smoothed net-new-high line, a +0.20 thrust) and fire the "breadth leads, buy the index" rule 114 times over 21 years, and it **loses to buying on random days at every horizon** (and the breadth-structure placebo leaves the result untouched, *p* = 0.82): all tide, no tool.

## What we tested

We encode the tightest mechanical version a proponent would accept. Each day we count basket members at a fresh **52-week (252-day) high** minus those at a fresh low (trailing data only, no look-ahead), divide by the basket size, and smooth over 10 days — the **NH-NL breadth line**. A long fires on SPY when that line crosses **up** through **+0.20** (a breadth thrust), entered at the **next close** (one documented lag), and we measure the forward 5/10/20/60-day return (yfinance daily total-return, 2005→2026). The Signal axis is **thrust vs a drift-matched random-entry baseline** (a Welch *t*) — the only honest test on an upward-drifting tape — plus a **shuffled-membership geometry placebo** that destroys the cross-sectional breadth structure while keeping each member's marginal new-high rate. Tradability charges costs on every thrust. A deterministic synthetic control with a *planted* breadth-lead proves the detector is live (edge 0 → *t* = −0.75; planted lead → *t* = +3.02), so the flat real-tape result is a genuine "nothing there".

> ⚠️ **Breadth proxy.** A true NH-NL universe is *thousands* of issues; offline we proxy it with **5 liquid ETFs** (SPY QQQ IWM DIA GLD). That's a **coarse proxy that caps the test** — but the failure is lopsided enough (loses to random at every horizon, placebo *p* ≈ 0.82) that a richer basket would have to overturn a very clear result.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the NH-NL line is, why a long entry on a rising market always looks good, the thrust-vs-random race, and the breadth scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the mechanical breadth line, one-sample HAC *t* vs the beta trap, the random-entry Welch test, the shuffled-membership placebo, per-index deltas, costs, and a synthetic planted-lead control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`new_highs_new_lows/`](new_highs_new_lows/). Breadth = net 52-week new-high fraction across a 5-ETF proxy basket, 10-day smoothed; thrust = up-cross of +0.20; entry is the next close (one lag). The breadth basket is a coarse proxy for true exchange breadth (see docs) — but this is a single-instrument timing study, so the random-entry baseline neutralizes the drift. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
