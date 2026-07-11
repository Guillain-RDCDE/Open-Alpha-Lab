# Study 657 — Larry-Portfolio 🌾

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does 30% small-cap value / 70% bonds deliver 60/40-like returns at much lower equity risk? | ![Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square) | *Real on the risk cut · None on the return-match.* Larry (30% IJS/70% IEF) runs at **7.1% vol / −20.8% maxDD / 0.57 corr-to-SPY**, genuinely far calmer than 60/40's 10.2% / −29.8% / 0.96 — but it does **not** match 60/40's return: CAGR **−2.66 pts/yr** behind, Newey-West *t* = **−2.29**, bootstrap CI **[−4.92%, −0.52%]** entirely negative. The **Sharpe** (risk-adjusted) gap is a statistical **tie** (CI **[−0.292, +0.187]** includes zero). |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Two cheap, liquid ETFs, annual rebalance, 2 bps costs — trivial to run. But it certifiably trailed a free 60/40 on this tape, and the one thing that would justify the trade-off — a persistent small-value premium — is statistically **indistinguishable from zero** here: IJS − SPY earns **+0.15%/yr at *t* = +0.06** over the full 2002–2026 window. |
| **"Has the small-value premium decayed since Swedroe popularized it?"** | ![Mixed](https://img.shields.io/badge/Decayed%3F-Mixed-8b949e?style=flat-square) | Can't certify decay — the pre-/post-2007 era difference is *t* = **−1.11** — but can't certify the premium was ever robustly *present* on this modern ETF tape either (+5.08%/yr pre-2007 at *t* = 1.20; −0.97%/yr since, at *t* = −0.35 — neither significant). |

> **In one sentence:** the Larry Portfolio genuinely delivers the calm ride it promises
> (roughly a third less volatility and drawdown, half the correlation to a stock-market
> crash) but it does **not** deliver the "60/40-like returns" half of the pitch — it trails
> a free 60/40 by a certified **−2.66 pts/yr** over 2002–2026 — because the small-cap-value
> premium the whole construction leans on is statistically **absent** on this tape (*t* =
> +0.06), consistent with this desk's own teardowns of the size and value premia individually.

## What we tested

Larry Swedroe's "Larry Portfolio": put the equity-risk budget in the highest-expected-return
factor — small-cap **value** (IJS) — and hold only ~30% of it, parking ~70% in safe bonds
(IEF), the idea being the higher-octane sleeve lets a much smaller equity weight match a
conventional 60/40 (SPY/IEF)'s return while running far less total equity risk. Real daily
total-return closes since 2002-07-30 (IEF/SHY's shared inception — the binding window, and
the same start sibling study [97-balancing-act](../97-balancing-act/) uses for its own 60/40),
annual rebalance, 2 bps costs, Sharpe excess-of-cash (SHY). We test the return-match claim
with a Newey-West *t* and a circular block-bootstrap CI, the risk-adjusted (Sharpe) claim with
a *separate* ratio-level bootstrap, the risk-reduction claim directly (it needs no inference —
it's arithmetic on the weights), and the small-value premium's own decay with an externally
justified 2007 era split plus a CAPM-neutral synthetic control that proves the machinery is
unbiased. **Dedup:** [513-size-effect](../513-size-effect/) and
[530-book-to-market-value](../530-book-to-market-value/) already tear down the plain size and
value premia on stock baskets — this study buys the *ETF* and asks whether a *portfolio* built
on top can still deliver on Swedroe's promise; [655-ivy-portfolio](../655-ivy-portfolio/) and
[68-all-weather](../68-all-weather/) diversify across uncorrelated sleeves rather than
concentrating in one factor; [97-balancing-act](../97-balancing-act/) is the plain 60/40 this
study races against, built with the identical convention for a fair fight. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a smaller, spicier equity sleeve is supposed to punch above its weight, what actually happened when you check, and why "less risk" and "same return" turned out to be two different promises |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the HAC/bootstrap splits on return and Sharpe, the equity-risk arithmetic, the era-split decay test, and the CAPM-neutral synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`larry_portfolio/`](larry_portfolio/). IJS/IEF/SPY/SHY are broad, still-listed
ETFs — no survivorship. **Not investment advice** — research & education. See
[LICENSE](../../LICENSE).*
