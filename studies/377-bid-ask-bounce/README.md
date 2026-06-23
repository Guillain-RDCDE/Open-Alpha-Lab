# Study 377 — Bid-Ask-Bounce 🏓

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is short-horizon reversion real? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The one-day reversion is statistically real on the tape (gross Sharpe **1.20**, *t* = **5.5**, lag-1 autocorr **−0.065**) — but Roll's estimator reads a **109 bps** effective spread out of the *same* autocovariance, and the synthetic control shows a **pure bid-ask bounce** produces an identical gross signal with **zero** underlying reversion. Real-as-a-statistic, but a microstructure artefact, not a price-predictive edge. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The reversal book breaks even at a **~7 bps** round-trip; the spread it must actually cross is **~109 bps** — **16× too wide**. Net of even a 10 bps round-trip the edge is already gone (Sharpe **−0.53**). You cannot harvest a reversion that *is* the spread. |
| **Free lunch?** | ![Busted](https://img.shields.io/badge/Free_lunch%3F-Busted-8b949e?style=flat-square) | "Buy yesterday's losers" backtests like alpha **only** because the backtest prints at the close and never pays the bounce it trades against. Roll (1984) named the trap 40 years ago — short-horizon "mean reversion" is, in large part, **the bid-ask bounce in disguise**. |

> **In one sentence:** small-cap daily returns really do mean-revert and a one-day reversal book really does backtest at Sharpe 1.2 — but Roll's 1984 model shows that exact negative autocorrelation is what a *pure bid-ask bounce* around an efficient random walk manufactures (cov₁ = −(s/2)²), and net of the ~109 bps spread the data itself implies, the "edge" the book breaks even at ~7 bps is gone, so the reversion is real-as-a-statistic, a mechanical microstructure artefact, and undeployable as a strategy.

## What we tested

The offline core is **Roll's (1984) microstructure model**: a *true* price that is a pure random walk — **zero** predictable reversion — plus a **bid-ask bounce** where each trade prints at the bid or the ask. Roll proved the bounce alone injects a negative lag-1 autocovariance, **cov₁ = −(s/2)²**, into observed returns — *illusory* mean reversion that no one can harvest, because capturing it means crossing the very spread that creates it. We then run the same idea on the **real tape**: a one-day cross-sectional reversal book (long yesterday's losers, short the winners) on a fixed **35-name** small/mid-cap basket over **20.9 years** (2005–2026), where spreads — and the bounce — are widest. yfinance has no bid/ask, so the effective spread is **inferred** by Roll's estimator and labelled a proxy throughout. The reversal book is measured **gross and net** of a one-way half-spread, with a 1-day execution lag; a planted-edge synthetic control separates a *genuine* reversion (which survives the spread) from the *bounce* (which does not).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the bid-ask bounce is, why "buy the losers" looks like free money on a chart, and why the profit is exactly the spread you'd pay — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | Roll's cov₁ = −(s/2)² identity and spread estimator, the gross-vs-net reversal book on the small-cap tape, a block-bootstrap null, break-even spread vs the implied spread, and a planted-edge / faithful-engine control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`bid_ask_bounce/`](bid_ask_bounce/). The effective spread here is an explicit **Roll-estimator proxy** (yfinance has no quotes), not an observed bid/ask. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
