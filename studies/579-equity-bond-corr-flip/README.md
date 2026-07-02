# Study 579 — Equity-Bond-Corr-Flip 🪢

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

The stock-bond correlation flipped positive in 2022 — does its sign tell you when the 60/40 hedge stops working?

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the correlation sign predict the forward 60/40? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Right *sign* but sub-threshold: forward 60/40 averages **+0.885%/mo** after negative-corr months vs **+0.566%/mo** after positive-corr months, spread **+0.319%/mo** — but two-sample *t* = **+0.89** (placebo *p* **0.36**), below the *t* ≥ 2 bar, and the sign **flips across sub-windows** (−0.30%/mo in 2002-09, +0.61%/mo in 2010-19). The mechanism is real; this tape can't certify it. |
| **Tradability** — does timing the flip pay? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | A de-risk-when-positive timer **cuts risk but costs return**: max drawdown **−28.5% → −23.6%**, vol **9.9% → 7.1%**, Sharpe **0.88 → 0.97**, but net return edge **−1.84%/yr**. A defensible risk overlay, not alpha — and the whole "flip" is **one** macro regime change (only 4 negative-corr months post-2022). |

> **In one sentence:** the stock-bond correlation flip is *economically* real — it went positive in early 2021, stayed positive through all of 2022's −14% 60/40 year, and a timer following its sign would have cut that drawdown (−28.5% → −23.6%) — but as a *timing rule* the edge is `WEAK` (right sign, *t* 0.89, placebo *p* 0.36, sign unstable across windows) and its tradable form is `FRAGILE` (a de-risking overlay that gives up 1.84%/yr of return and rests on a single 24-year regime change).

## What we tested

The folklore that the **sign** of the trailing stock-bond correlation is a regime timer for the
60/40 portfolio: negative correlation → bonds hedge, diversification works; positive correlation
(as in 2022) → the hedge breaks, so de-risk. We take **SPY** (equity) and **TLT** (20+yr Treasury)
daily adjusted closes, 2002-08 → 2026-05 (286 complete months), compute the sign of the trailing
6-month return correlation, and test whether the **forward one-month** 60/40 return is higher after
negative-correlation months: a two-sample (Welch) *t*, a **label-shuffle placebo** null, a
**five-window** robustness sweep, a rules **timing overlay** (de-risk when positive, gross and net
of a 5 bps switching cost) benchmarked on return / vol / drawdown / Sharpe, and a deterministic,
seed-robust synthetic positive control that plants the effect and proves the engine catches it.
*Distinct from [578 Cross-Asset-Correlation-Regime](../578-cross-asset-correlation-regime/) (broad
average-pairwise-correlation state) and [502 Betting-Against-Correlation](../502-betting-against-correlation/)
(a cross-sectional stock signal): this is the single SPY/TLT pair as a **60/40 diversification
timer**.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the stock-bond correlation is, why the 2022 flip broke the 60/40, and why "de-risk on the flip" cuts losses but not the way you'd hope |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the regime split with a two-sample *t*, the placebo null, the five-window sign-flip, the timing overlay's return/vol/drawdown/Sharpe, and the seed-robust synthetic positive control |

The fingerprinted real-data run (SPY + TLT, 286 months, price fp `884aad7ef4eb`, monthly-panel fp
`55f8c175e893`, as-of 2026-06-30) is in [docs/results.md](docs/results.md); the offline machinery
proof runs on the deterministic synthetic world in
[`equity_bond_corr_flip/data.py`](equity_bond_corr_flip/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`equity_bond_corr_flip/`](equity_bond_corr_flip/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
