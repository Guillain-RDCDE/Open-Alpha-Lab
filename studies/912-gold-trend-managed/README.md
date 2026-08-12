# Study 912 — Gold + Trend 🥇

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the edge real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Excess-of-cash Sharpe advantage of the overlay is **−0.19** (HAC *t* = −1.83) — the **wrong sign**: trend-managed gold has a *lower* risk-adjusted return than just holding gold. The bootstrap CI [−0.11, +0.75] includes zero; the advantage is negative in **both** eras. The drawdown benefit (+10 pp) is modest and **reverses** post-2016. |
| **Tradability** — is it bankable? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No positive edge to bank — negative gross, more negative with cost. The "drawdown insurance" costs ~4.3 pp/yr of CAGR (8.1% → 3.8%) for a worst-loss cut that is absent in the recent decade. |

> **In one sentence:** Applying Faber's 200-day trend filter to gold does *genuine* timing (it beats a random control) but still **under-performs buy-and-hold gold** on the real, costed, excess-of-cash tape — because gold's biggest rallies are sharp V-recoveries the slow filter re-enters too late — so it is a Sharpe-losing volatility reducer, not the better-Sharpe drawdown-managed diversifier the folklore promises.

## What we tested

Hold **GLD** only when its price is above its **200-day** moving average, else sit in
**BIL** (T-bills); binary, no shorting, one-day rebalance lag, 5 bps one-way cost. Every leg
is raced **excess-of-cash** (minus BIL's total return) against buy-and-hold gold and a
**random control** matched to the same 63% in-market frequency, over GLD∩BIL 2007-05-30 →
2026-06-30 (total-return closes, `auto_adjust=True`). Bootstrap Sharpe CIs, a two-era cut, a
cost sweep, a calendar-year table, and an IAU cross-check. **Dedup:** distinct from
**110-faber-timing** (same rule on equities, where it *is* a real shield), **640-gold-overnight**
(intraday overnight effect), **649-gold-seasonality** (calendar effect), and
**831-gold-real-yield-timing** (macro real-yield signal, not the self-referential 200-day MA).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the dead-decade idea in plain language, why the overlay lags, the V-recovery whipsaw, the honest verdict |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | excess-Sharpe race, HAC return-diff *t*, bootstrap CIs, era cut, cost sweep, the live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`gold_trend/`](gold_trend/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
