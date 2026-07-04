# Study 601 — Factor-ETF-Live-Test 🏷️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — did the wrappers deliver their stated factor exposure? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | Decisively, on the live tape: USMV beta **0.689** (*t* vs 1 = **−7.52**), realized vol **−20%** (CI 12–26%); MTUM loads on a 12-1 sector momentum spread at NW ***t* = +7.45**, VLUE on the value spread at ***t* = +9.99**, QUAL on an independent quality proxy at ***t* = +8.76** — with spread-sign splits at Welch *t* 3.6–6.1 and placebo *p* ≤ 0.0003. Caveat: these four are the flagship **survivors** of the smart-beta launch wave. |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | The exposure is trivially buyable (0.15%/yr, penny spreads, huge AUM) — but the **alpha** the exposure was sold to harvest never showed: CAPM alphas **−1.0% to +2.7%/yr, all \|*t*\| ≤ 1.24** (robust to NW lags 3/6/12), and the style spreads themselves paid ~zero over the live window. You can buy a risk *profile*, not an edge. |
| **"Did any beat SPY outright?"** | ![Busted](https://img.shields.io/badge/Beat_SPY_outright%3F-Busted-8b949e?style=flat-square) | **None at significance.** Three of four lagged on CAGR (USMV **−3.5 pp/yr**, VLUE −0.6, QUAL −0.4); MTUM finished ahead (+2.4 pp/yr) at *t* = **0.93** — noise. The wrappers delivered the factors; the factors didn't deliver the market-beating. |

> **In one sentence:** a decade-plus after BlackRock put the academic factor zoo in 0.15%/yr
> wrappers, the live tape says every fund faithfully delivers the exposure on its label
> (*t* = 7.4–10) — and none of the exposure paid: CAPM alphas are statistically zero, three of
> four funds lagged SPY outright, and USMV's celebrated result is a risk profile (−20% vol,
> down-capture 66%), not a premium — **exposure delivered, alpha not; Real, but Fragile**.

## What we tested

The four flagship iShares factor ETFs — **USMV** (min-vol, 2011), **MTUM** (momentum, 2013),
**VLUE** (value, 2013), **QUAL** (quality, 2013) — vs **SPY** on monthly **total returns**
(net of each fund's own fee) since each inception, as-of 2026-06 (176/158/158/155 months).
Per fund: CAPM beta and alpha with **Newey-West** errors, realized vol ratio with a paired
block-bootstrap CI, up/down capture, and an **exposure-delivery** test — a two-factor NW
regression of fund excess on [market, the style's realized long-short spread] (sector 12-1
WML built with one clean month of lag; IWD−IWF for value; SPHQ−SPY for quality) plus a
spread-sign month split with a 10,000-draw placebo. Third axis: the outright race (CAGR gap,
excess-vs-excess Sharpe, active-mean NW *t*). A deterministic synthetic world with planted
beta/loading/alpha proves the machinery. Flagship-survivor selection is named on the Signal
axis. **Dedup guard:** [330-low-volatility-anomaly](../330-low-volatility-anomaly/) and
[242-quality-minus-junk](../242-quality-minus-junk/) grade the *academic cross-sections*;
this is the **live product audit**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "factor exposure in an ETF" actually promised, the two halves of the pitch (profile vs premium), why USMV can lag by 3.5 pp/yr and still do its job, and why "it tracked the factor" ≠ "it beat the market" — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | NW CAPM per fund, beta-vs-1 tests, vol-ratio bootstrap, two-factor style loadings + spread-sign splits with permutation placebos, alpha lags robustness, the outright races, and the planted-parameter synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`factor_etf_live_test/`](factor_etf_live_test/). The audited unit is the LIVE
product net of its own fee; no trading rule is executed — the only constructed signal (the
sector WML proxy) carries one documented month of lag. **Not investment advice** — research &
education. See [LICENSE](../../LICENSE).*
