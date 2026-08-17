# Study 938 — Open or Close 🔔

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the fill venue systematically change the P&L? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Pooled over **538 fills** the open-minus-close slippage is **+1.39 bps per trade** — *t* = +0.33 naive, **+0.23 clustered on the trade date** (the fills sit on only 361 dates, four correlated tapes flipping together) — and the open fill wins **49.6%** of the time (95% Wilson [45.4%, 53.8%]; the honest cluster-bootstrap interval is wider still, [44.1%, 55.1%]). The two rules print **opposite signs** on the same tapes — **+37.0 bps/yr** monthly (HAC *t* = +1.53) against **−26.1 bps/yr** weekly (*t* = −0.80) — both bootstrap CIs straddle zero, and the era cut disagrees with itself. The one |*t*| ≥ 2 cell (EEM monthly, 37 trades) is one hit in eight draws, and its mechanism reverses sign on the higher-powered weekly rule. *Survivorship: none — four continuously listed index ETFs, no cross-section.* |
| **Tradability** — is there a venue rule worth writing down? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Nothing to bank: the open fill's +0.031 monthly Sharpe edge is mirrored by a −0.021 weekly one, so no sign can be committed to in advance. What *is* robust is the noise — across the 8 rule × tape cells the realised gap ranges **[−35.8, +66.6] bps/yr** (sd 39.5), ~±0.4 pp/yr of free track-record luck. Charge the opening auction its real (wider) spread and both rules drift toward the **close**. |

> **In one sentence:** filling the same moving-average timing rule at tomorrow's open instead of tomorrow's close changes nothing you can rely on — over 538 real fills on four ETFs the venue is a **coin flip (49.6% win rate)** whose realised contribution still swings ±0.4 pp/yr of pure luck — so pick the **closing** auction, not for return but because it is the cheaper and deeper of the two.

## What we tested

Faber's moving-average filter — hold the ETF while its period-end close is above the mean of the
last `L` period-end closes, else hold **BIL** — in two flavours (**10-month**, month-end; **20-week**,
week-end — faster *and* shorter-horizon, ~100 sessions of lookback against ~210, so not a pure
frequency change) on **SPY / IWM / EEM / EFA**, 2007-05-30 → 2026-06-30, 2 bps one-way × NAV,
long-only so no borrow. Exactly **one** execution lag: signal through the close of `t`, fill on
`t+1` — at that day's **open** for one arm, at its **close** for the other. On every non-trade day
the arms are identical, so the whole gap is `Δweight × intraday`; any flaw in the *signal* cancels
between the arms. Adjusted OHLC (`auto_adjust=True`) makes the intraday leg **price-only** and puts
the dividend in the overnight leg; both are labelled, as are excess-of-cash and raw growth rates.
The opening-auction spread penalty and the overnight share of the cash accrual are **PROXIES** and
are swept. Inference is clustered on the trade date, because the fills are not independent.
**Dedup:** distinct from **01-overnight-anomaly** and **788-overnight-intraday-tug-of-war**
(which *harvest* the split as a premium), **110-faber-timing** (which asks whether the rule beats
buy-and-hold at all), **836-timing-luck** and **937-tranched-rebalancing** (which *day of the
month* you rebalance — synthetic null and real tape respectively; this is the *moment of the
day*), and **352-opening-range-breakout** / **80-cold-open** (which trade the opening session as
a signal rather than use it as a fill point).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the choice looks free, the coin-flip result, the ±0.4 pp/yr of luck it still buys you, the honest advice |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the sliver algebra, per-tape HAC *t*, the clustered-on-date correction, bootstrap CIs, era cut, the entry-minus-exit decomposition, the proxy sweeps, the live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`open_close_exec/`](open_close_exec/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
