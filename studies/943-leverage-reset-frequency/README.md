# Study 943 — Reset Frequency ⚖️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the edge real? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | *Real on the return, absent on the Sharpe; positive at 2x, negative at 3x.* At 2x the monthly reset earned **+1.05 bps/day (HAC *t* = +2.79)** on the real tape, and the gap is positive in choppy months and negative in trending ones in both sleeves (slope *t* = −6.8 / −7.2). But the *claim* fails: the risk-adjusted advantage is **+0.022, bootstrap CI [−0.010, +0.062]**, it is not forecastable (lagged-ER *t* = +0.72), and at 3x it is **−0.569 (*t* = −3.62)**. Survivorship sits on the daily-reset side: SSO/UPRO survived, the closed 2x/3x products are not on this tape. |
| **Tradability** — is it bankable? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | At 3x the monthly reset buys no extra leverage on average — **2.98x mean against the daily arm's 3.00x** — and pays for its extra return in risk: an **8.42x peak**, 58.5% vol against 51.3%, a **margin call on 2011-08-08** (and on 2008-09-29 on the longer SPY tape), and **negative equity** at 4x in October 2008. Half of its apparent edge over the funds is just their ~1 pp/yr fee-and-tracking drag. |

> **In one sentence:** The daily reset is not the villain — it is the *hero of trending months* and the villain only of choppy ones, and the monthly reset that forum wisdom prescribes buys **+0.02 of Sharpe** at 2x, nothing at all at 3x once you can be margin-called, and a position that could go to negative equity at 4x.

## What we tested

Build the alternative explicitly: SPY on margin, financed at **^IRX + 50 bps**, levered
back to **2x / 3x once a month** and left to drift in between — then race it,
**excess-of-cash** (minus BIL), against a daily-reset replication and against **SSO** and
**UPRO** themselves. One execution lag (the reset is decided at the month-end close, in
force next session), 2 bps one-way on exposure turnover, and a **25% maintenance margin** —
the spread, the cost and the margin ratio are all **labelled assumptions and all swept**;
a called account is credited the cash leg, not zero. HAC *t* on the daily difference,
paired block-bootstrap CIs, an era cut **with a test of the era difference**, a
trending-versus-choppy decomposition (and its *lagged*, tradable form), a 2004-2026 SPY
stress that reaches the 2008 crash UPRO never saw, and a synthetic control with a
choppiness knob. **Dedup:** distinct from **61-slow-burn** and **100-melting-ice** (which
study the daily-reset fund *as it is*, never building the monthly alternative),
**942-inverse-etf-structural-loss** (the same drag on the *inverse* wrapper, sign not
frequency), **944-optimal-leverage-realized** (*how much* leverage, reset held daily) and
**945-leverage-financing-cost** (what the borrow *costs*, not when it is trued up) —
943 holds the multiple and the financing fixed and varies only the reset clock — plus
**102-free-rebalance** (rebalancing *between assets*, not leverage on one),
**836-timing-luck** (rebalance *phase*, not frequency), and **593-hfea** /
**594-leverage-rotation-200sma** (allocation studies using leveraged funds as ingredients
rather than testing the wrapper's mechanics).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the daily reset is blamed, what a monthly reset really does to your exposure, the margin call that ends the story, the honest verdict |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the excess-of-cash race, HAC difference *t*, paired bootstrap CIs, the efficiency-ratio decomposition and why it is arithmetic, era cut, three sweeps, the live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`reset_freq/`](reset_freq/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
