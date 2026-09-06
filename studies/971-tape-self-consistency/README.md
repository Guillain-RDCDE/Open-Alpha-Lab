# Study 971 — Does the Tape Agree With Itself? 🔍

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — are there real inconsistencies in the tape a backtest reads? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | The audit ran **62** checks across 8 tickers and returned **1 errors** and **3 warnings**. The weekly and monthly bars compound from the daily ones to within **4504 bps** at worst; rebuilding the total-return series from price plus dividends and splits reproduces the provider's own adjusted close to **+0.005%/yr** at worst (XLU); and the reference calendar shows **0** missing sessions in total. |
| **Tradability** — would any of them change a result you would publish? | ![Useful](https://img.shields.io/badge/Useful-2ea44f?style=flat-square) | The same buy-and-hold statistic computed from the daily tape and from the provider's own weekly bars differs by up to **0.052** of Sharpe and **+0.01%/yr** of CAGR (SPY) — most of which is the arithmetic of measuring volatility at a different frequency rather than a data fault. The fault that *would* change a published number is the reconstruction gap, and the audit prices it at +0.005%/yr. |

> **In one sentence:** A free daily feed is much better than its reputation — the weekly and monthly bars agree with the daily ones to a few basis points and the adjusted close is reconstructible to +0.005%/yr — but it is not perfect, and the checks that catch the exceptions cost six functions and a test suite.

## What we tested

Every study on this desk begins by trusting a free data feed, and none of them checks
it. This one does. We ask the same provider for the same eight tickers **four different ways**
in a single pass — daily total-return bars, daily unadjusted bars with the dividend and split
events attached, weekly bars, monthly bars — and then run six checks whose answers are not
matters of opinion: do the weekly and monthly bars compound from the daily ones; can the
adjusted close be **rebuilt** from price plus dividends plus splits (the computation every
total-return backtest silently depends on); are any sessions missing against a reference
calendar inferred from the whole sample; does each split move the as-traded price by its ratio
and leave the adjusted price alone; do the reported dividends account for the gap between the
price and total-return CAGRs; and does the tape contain duplicates, non-positive prices or
impossible moves.

The universe is chosen for corporate-action variety — **AAPL, NVDA and TSLA** for splits,
**VYM and XLU** for large dividends, **SPY and QQQ** for the well-behaved case, **GLD** as the
no-dividend control. The whole audit is also run against a synthetic tape with a dropped
session, an unapplied split and a missing dividend planted at known positions, because an audit
that has never failed has not been tested.
**Dedup:** distinct from **347-look-ahead-bias** and **345-survivorship-bias** (biases in how
data is *used*), **917-nav-staleness-timezone** (a real economic effect, not a data fault),
**972-adjustment-mode-matters** (which adjustment convention to *choose*, given a correct feed)
and **919-index-methodology-change** (the index changing, not the feed misreporting it).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a price feed actually promises, the one calculation every backtest inherits without checking it, and what the audit found |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | six consistency checks with severities, total-return reconstruction from corporate actions, calendar coverage, and the whole audit validated against a tape with planted faults |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`tape_audit/`](tape_audit/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
