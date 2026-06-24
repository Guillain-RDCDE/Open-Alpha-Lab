# Study 436 — Moving-Average Envelopes

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the band time anything? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The SMA(20) ±5% envelope buy-the-dip rule earns net Sharpe **+0.278** on SPY back to 1993, HAC *t* = **+1.97** — *below* the *t* ≥ 2 bar. A block-permutation placebo that keeps the same exposure and scrambles the timing is not beaten (*p* = **0.232**): the band carries no information over "be long this fraction of the time." No panel tape clears *t* = 2. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | In the market only **6.6%** of the time, the rule is *diluted beta*: its +0.278 Sharpe is worse than buy-and-hold (**+0.644**) and a bare SMA(200) filter (**+0.727**). Turnover is tiny (~2.6/yr) so costs are irrelevant — the gross simply never beats just holding. |
| **Beats Bollinger Bands?** | ![Busted](https://img.shields.io/badge/Beats_Bollinger%3F-Busted-8b949e?style=flat-square) | On the identical rule, instrument and window the volatility-scaled **Bollinger** band earns **+0.444** (*t* = +3.11) vs the envelope's **+0.278**, and beats it on four of five tapes. The fixed-percent envelope is the *weaker* band — the folk claim is backwards. |

> **In one sentence:** a percent moving-average envelope, turned into an honest long/flat timing rule on SPY since 1993, earns a positive-but-sub-2 Sharpe (+0.278, *t* = +1.97) that a same-exposure mis-timing placebo can't be told apart from (*p* = 0.23) — it is diluted beta that loses to buy-and-hold *and* to the Bollinger Band it is supposed to beat.

## What we tested

Percent envelopes (Granville/Hurst, 1960s) plot a moving average flanked by lines a **fixed percent** above and below — and their advocates claim the fixed band beats Bollinger's *volatility-scaled* band because it doesn't "breathe." We take the steelman literally: SMA(20) ± 5% on five liquid index tapes (SPY, QQQ, DIA, IWM, EFA), turned into a daily **long/flat** rule (buy when the close pierces the lower band, exit at the mid), entered with **one execution lag**, charged one-way costs × NAV, and raced **excess-of-cash vs excess-of-cash** against buy-and-hold. The "it's better" claim is then *actually tested* against the two obvious simpler benchmarks — a **Bollinger Band** (same rule, volatility-scaled width) and a bare **SMA(200)** trend filter. A block-permutation placebo (scramble the timing, keep the exposure) and a planted-edge synthetic control decide whether the band times anything or is just being long.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what an envelope is, how it differs from Bollinger, why "in cash 93% of the time" turns an edge into diluted beta, and why it loses to just holding — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the long/flat book with one execution lag, net excess-vs-excess Sharpe vs buy-and-hold, HAC *t*, the block-permutation placebo, the Bollinger/SMA head-to-heads, the cost & *k* sweeps, and a planted-edge synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`ma_envelopes/`](ma_envelopes/). Prices are total-return adjusted (`auto_adjust`); all Sharpes are net, excess-of-cash. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
