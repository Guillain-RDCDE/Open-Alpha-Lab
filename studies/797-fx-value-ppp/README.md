# Study 797 — FX Value (PPP) 💱

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do undervalued real exchange rates predict appreciation? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | Long-cheap/short-rich G10 currencies earns the right sign — **+1.02%/yr**, Sharpe **0.30** — matching a deep literature, but the autocorrelation-robust **Newey-West *t* = +1.58** (one-sample +1.37), *below* the desk's `t ≥ 2` bar. A 10,000-draw random-sign placebo gives **p = 0.09** (≈ 1 in 11), the hit rate 52.4% (Wilson [46.2%, 58.6%]) brackets 50%, and every trailing window (36–84 mo) is positive but sub-2. Literature says real; this G10-only tape reads **weak**. |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Net Sharpe **0.10** at a retail 5 bps and net return **negative** (−0.20%/yr) by a realistic institutional 20 bps, once turnover (≈0.30/mo) and a 100 bps/yr short-borrow spread are charged. The ~1%/yr gross cushion is thinner than its own frictions. |
| **Survivorship** on the Signal axis | ![Named](https://img.shields.io/badge/Survivorship-Named-8b949e?style=flat-square) | A fixed current-membership developed-market basket; the documented FX-value premium is disproportionately an **emerging-market** phenomenon this study deliberately excludes, so the G10 weakness is partly a basket choice — stated openly, not buried. |

> **In one sentence:** the textbook PPP-value trade — buy currencies that are cheap in
> real terms, sell the rich ones — has the **right sign** on the G10 (**+1%/yr**, Sharpe
> 0.30) and a mountain of research behind it, but the robust *t* is only **1.58**, a
> coin-flip beats it 1 time in 11, and the thin gross edge goes **negative** once you pay
> to trade and short it — so the honest read is **weak signal, mirage paycheck**.

## What we tested

The oldest currency-value idea, stated the way its believers state it: *"a currency
that's cheap relative to purchasing-power parity is a buy — prices and FX eventually
line up."* We make "cheap" precise with the **real exchange rate** —
`log q_i = log S_i(USD/FX) + log CPI_i − log CPI_US` — and rank the nine G10 currencies
on how far each sits **below its own 5-year trailing average** (undervalued = a long),
building a dollar-neutral long-cheap/short-rich book rebalanced monthly on **G10 FX
(yfinance) + national CPI (IMF IFS / Eurostat HICP)**, 2005→2025. CPI is lagged one
month (publication lag) and weights earn the *next* month's spot return — one execution
lag, zero look-ahead into an unreleased print. We grade it with a Newey-West HAC *t*, a
10,000-draw random-sign placebo, a trailing-window sweep, and a cost+borrow timer, plus
a 20-seed synthetic control that recovers a *planted* PPP reversion (t = 12.4) to prove
the machinery. **Dedup:** [215-big-mac-ppp](../215-big-mac-ppp/) is a single burger-basket
folklore **snapshot** with no real-rate *time series*; [364-fx-carry-trade](../364-fx-carry-trade/)
is the **opposite tilt** (rank on the rate differential); [147-fx-momentum](../147-fx-momentum/)
ranks on **trend**; [114-dollar-smile](../114-dollar-smile/) is the broad-**USD cycle**
level. This is the FX **value** factor — the real-rate deviation-from-PPP, traded. As-of
**2025-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "cheap in real terms" means, who's undervalued now (JPY) vs rich (GBP), why the climb is real but shallow, and why costs erase it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the real-rate value sort, Newey-West inference, the 10,000-draw sign placebo, the trailing-window sweep, the (snooped) 2015 split, the cost+borrow timer, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`fx_value/`](fx_value/). G10 FX + national CPI, a fixed developed-market
basket (survivorship named on the Signal axis). **Not investment advice** — research &
education. See [LICENSE](../../LICENSE).*
