# Study 1021 — Stability Breeds Instability 🌋

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does prolonged low volatility predict bigger drawdowns years later, beyond what mean reversion alone delivers? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | It leans Minsky's way, but the tape can't certify it. One SD more prolonged calm goes with a **+1.4%** deeper 24-month max drawdown, but the raw HAC t is **0.88** (non-overlapping 0.75). A no-feedback GARCH-t fitted to the same 92 years predicts **−1.3%** instead, because calm is supposed to be followed by calm. The real slope beats that null at one-sided **p = 0.08**, and a long-memory FIGARCH-t at **p = 0.06**: suggestive, short of 5%. The result also depends on the trend window, sits wholly in the post-1979 half and rests on **7** independent ≥20% drawdowns. At one month the sign is the other way and certain (log-RV AR(1) φ = 0.69, t = 14.6). |
| **Tradability** — does selling the calm pay? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Halving equity after prolonged calm (one-month lag, 10 bp, bills on cash, 1952-2018) cut max drawdown from −50% to −46%. It also lowered net excess Sharpe, **0.487 vs 0.496** (block-bootstrap p(Δ≤0) = 0.60), and turned $1 into 717× instead of 986×. Calm months paid *more* equity premium than the rest, so stepping out of them is the cost. A plain vol-target did no better (0.455). |

> **In one sentence:** Long calm is followed by somewhat bigger drawdowns than a no-feedback volatility model predicts, but not significantly so (raw t = 0.88; GARCH p = 0.08, FIGARCH p = 0.06), and selling the calm moved net Sharpe by −0.01: "low vol = sell" is a story the tape hints at and cannot cash.

## What we tested

**The claim, steelmanned.** Minsky held that *stability is destabilising*: when measured risk
stays low, agents lever up and sell insurance until the calm breaks. Danielsson, Valenzuela &
Zer (*Review of Financial Studies*, 2018; see [references](docs/references.md)) made it
empirical. Long stretches of volatility below its slow trend preceded crises years later. On
trading desks it travels as "low VIX = complacency = sell".

**Setup.** We build a past-only *prolonged-calm* score (5-year average of 12-month realised vol's
shortfall below its trailing 10-year mean) on Fama-French monthly total returns, 1926-2018. We
regress forward 1-36-month max drawdown, ≥20% crash odds, realised vol and excess return on it,
using HAC (2H lags) and non-overlapping samples. **The confound is the study.** Volatility
mean-reverts, so the real slope is set against GARCH-t and FIGARCH-t worlds fitted to the same
tape with no risk-taking feedback. The daily S&P 500 (1990-2022, price) gives the
short-horizon clustering sign. The VIX tape (2014-2018) is one labelled anecdote. A calm
de-risk rule is raced against buy-and-hold and a vol-target.

**Dedup:** [03-fear-gauge](../03-fear-gauge/) buys *high*-VIX spikes for a short-horizon
rebound. [16-storm-shy](../16-storm-shy/) and [591-vol-managed-portfolio](../591-vol-managed-portfolio/)
scale exposure by *current* vol. [111-vix-term-structure](../111-vix-term-structure/) times
equities on the VIX curve's slope, [578-cross-asset-correlation-regime](../578-cross-asset-correlation-regime/)
on correlation regimes, and [992-vol-clustering-halflife](../992-vol-clustering-halflife/)
measures how fast vol reverts. This study is the only one that asks whether *years* of
*low* vol forecast crashes *years* later, net of that mean reversion.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | Minsky's paradox in plain language, ninety years of calm and storm, why "calm then storm" happens even with no Minsky in the world, and why selling the calm cost money |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the signal, a horizon × outcome HAC grid, GARCH-t and FIGARCH-t mean-reversion nulls, a nine-way window grid and half-samples, the two-horizon profile with a null band, the VIX anecdote, planted/null synthetic controls, and the backtest with bootstrap and cost sweep |

Reproducible headline run: [docs/results.md](docs/results.md). Sources & literature map: [docs/references.md](docs/references.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`calm/`](calm/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
