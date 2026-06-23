# Study 381 — TIPS-Breakeven 🧮

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does breakeven inflation predict forward returns? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | Across **18** signal × asset × horizon tests, only **3** cross `|t| ≥ 2` — and the placebo null demotes them: the two **TIPS** hits are a **mechanical self-prediction** (the proxy mean-reverting; placebo *p* ≈ **0.11–0.16**), and the lone gold hit (*t* = **−3.21**) is a **1-in-18 false positive** that dies at other horizons. On **equities** there is nothing. Proxy/self-prediction support, not an independent edge. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Every conditional win-rate is **below** its base rate. The one "significant" gold rule, traded with a 1-month lag and 10 bps costs, loses **−1.3%/yr** net vs **+11.4%/yr** for simply holding gold. There is **no deployable, independent, cost-survivable edge**. |
| **Free macro-timing lunch?** | ![Busted](https://img.shields.io/badge/Free_macro--timing_lunch%3F-Busted-8b949e?style=flat-square) | A **multiple-testing + self-prediction illusion**: scan two signals × three assets × three horizons and ~one cell lights up by luck, while the "best" *t* you find is the breakeven proxy predicting **itself**. A deterministic synthetic control confirms the null is a *true* null, not an underpowered one. |

> **In one sentence:** the bond market's breakeven inflation rate co-moves with assets but **times** none of them you can trade — on a transparent `log(TIP/IEF)` proxy the only HAC *t*-stats that reach 2 are a mechanical self-prediction of TIPS and a single 1-in-18 gold false positive that loses money as a strategy, so breakeven is real-as-a-forecast, weak-as-a-signal, and a mirage as a trade.

## What we tested

True breakeven inflation (FRED `T10YIE`) isn't reliably fetchable here, so we **construct a transparent proxy**: `be = log(TIP / IEF)`, the inflation-protected Treasury ETF over a duration-matched nominal one, which rises with the market's inflation forecast (labelled a proxy throughout). We take its **level** (expanding z-score) and **3-month change** and ask whether either predicts forward 3/6/12-month returns of TIPS, equities (SPY) and gold (GLD), using Newey-West HAC predictive-regression *t*-stats, a shuffle placebo null, win-rates vs the base rate, and a traded sign rule with a 1-month execution lag and one-way costs. A deterministic synthetic control with a *planted* edge knob confirms the engine is unbiased (edge 0 ⇒ *t* ≈ 0) and powered (a real edge ⇒ *t* ≫ 2), so the real-tape null is genuine. (Same regress-a-macro-variable hazard as the rate studies [119](../119-real-rate-regime/) and [118](../118-fed-model/).)

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what breakeven inflation is, why "the bond market's forecast times your portfolio" sounds smart, and why a scan of macro signals finds luck — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the breakeven proxy, HAC predictive regressions across signal × asset × horizon, a placebo randomization null, the self-prediction artifact, costed sign rules, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`tips_breakeven/`](tips_breakeven/). Breadth-of-inflation-expectations here is an explicit **proxy** (`log(TIP/IEF)`), not the literal nominal-minus-real yield. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
