# Study 752 — TGA-Drawdown 🏦

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a falling TGA lift SPY next? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | At the thesis's own horizon (next month) the drawdown mean sits **below** base (**+0.91%** vs **+0.96%** — wrong sign); across 1/2/3/6 months the Welch *t* peaks at **+0.85** and the HAC slope explains **≤ 0.4%** of variance with a *t* that never leaves ±1. Indistinguishable from noise. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The "hold when the TGA is drawing down" overlay **underperforms buy-and-hold** gross *and* net (**+6.7%** vs **+11.6%**/yr, Sharpe **0.66** vs **0.78**). Acting on the signal *destroys* return. |
| **Hidden liquidity lever?** | ![Hidden_liquidity_lever%3F: Not_supported](https://img.shields.io/badge/Hidden_liquidity_lever%3F-Not_supported-8b949e?style=flat-square) | The lead/lag scan is incoherent — corr ≈ **0** (**−0.03 / +0.02**) at exactly the positive leads a real lever needs, its extremes scattered at *L* = −5 and *L* = +6. The injection neither leads nor lags cleanly; it co-wanders with a shared macro backdrop. |

> **In one sentence:** the "hidden fiscal QE" story that a Treasury cash drawdown injects reserves and lifts stocks over the following weeks doesn't survive the tape — the next-month sign is actually *negative*, no horizon or HAC regression clears *t* ≈ 1, the injection shows no coherent lead, and a rule built on it loses to buy-and-hold — largely because the biggest "injections" are debt-ceiling scrambles, not stimulus.

## What we tested

The macro-liquidity thesis — most associated with Michael Howell's *Capital Wars* / CrossBorder Capital "global liquidity" work and the "net liquidity = Fed − TGA − RRP" framework all over financial social media — says the **Treasury General Account** is a hidden liquidity lever: when the Treasury draws down its cash balance at the Fed, that cash becomes bank reserves (a "liquidity injection") and lifts risk assets over the following weeks; when it rebuilds, reserves drain and stocks are pressured. We rebuild that signal on a monthly TGA tape (a hardcoded, clearly-labelled monthly **proxy** of the weekly FRED `WTREGEN` operating cash balance, since FRED is firewalled here) aligned to month-end SPY, and measure forward 1/2/3/6-month returns conditional on the TGA drawing down vs building — with a strict one-month execution lag, a Welch *t*, a **Newey-West (HAC)** predictive regression, a placebo null, a lead/lag scan, and a cost-charged timing overlay. A deterministic synthetic control with a *planted* drawdown→returns link confirms the engine recovers a real edge and can't manufacture one from noise.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "the TGA drawdown is stealth stimulus" reads as a story the tape doesn't tell, why the biggest injections are debt-ceiling scrambles, and why trading it loses — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | drawdown-vs-build split returns, a Welch *t* + HAC (Newey-West) predictive regression + placebo null, the lead/lag identification test, the timing overlay vs buy-and-hold, robustness (window / ex-COVID), and a synthetic planted-edge control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`tga_drawdown/`](tga_drawdown/). The TGA tape is a hardcoded **monthly proxy** of the weekly FRED `WTREGEN` operating cash balance (approximate levels; landmark moves faithful), named as such. SPY is total-return adjusted. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
