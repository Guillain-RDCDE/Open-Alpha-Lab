# Study 948 — Capture Ratio 📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the edge real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | **0 of 14** funds show a positive capture spread at even a nominal \|*t*\| ≥ 2, let alone the Bonferroni bar of **2.91** that fourteen simultaneous tests demand; the bootstrap CI covers zero for **14 of 14**; against each fund's own zero-convexity twin, on its own sample, **0 of 14** reach *p* < 0.05 (smallest **0.127**); and the spread has **no rank persistence** across eras (rho = **−0.117**, *p* = 0.76, sign agreement 44%). **Survivorship:** the funds that closed are absent, so this is the flattering version. |
| **Tradability** — is it bankable? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Every fund is a low-beta index position (beta **0.52–0.87**) with no alpha — largest \|*t*\| on alpha across the panel is **1.56**, and it is negative — and only **1 of 14** beats its own benchmark on excess-of-cash Sharpe, on lower vol rather than higher return. The beta-neutral spread that would monetise the scorecard tops out at **+53 bps/month gross on 42 months** (*t* = +1.60) and at **+0.8 bps/month** among funds with a real history, which turns negative at 50 bps of borrow. |

> **In one sentence:** income funds do not "give up a little upside to avoid a lot of downside" — their up-capture and down-capture are the *same number*, their beta, so what they sell is a smaller position plus a distribution schedule; the six funds whose payoff shape *is* significant are all option-writers and all significantly **concave**, which is an accounting identity of selling calls rather than a discovered effect.

## What we tested

Up-capture, down-capture and their **spread** for fourteen US income ETFs — the
option-writers (QYLD, JEPQ, NUSI, XYLD, JEPI, PBP, DIVO, SPYI, RYLD) and a no-options
dividend control group (SCHD, VYM, DVY, SPHD, NOBL) — on **monthly total returns** against
the benchmark each is marketed against (SPY / QQQ / IWM), cash leg BIL, 2007-06 → 2026-06,
as-of 2026-06-30, after an **unadjusted-corporate-action screen** removed the 1-for-2
reverse split the vendor's "adjusted" tape had left in NUSI. The spread gets a joint
block-bootstrap CI, a Henriksson-Merton convexity twin with a HAC *t* judged against a
**Bonferroni** bar, a **fund-matched zero-convexity null** rebuilt on each fund's own
sample, a CAPM alpha, an excess-of-cash Sharpe race, an era cut with a Spearman
**rank-persistence** test, a benchmark-map sweep, an arithmetic-vs-geometric convention
sweep, and a beta-neutral traded arm (36-month rolling beta through *t*, held at *t*+1 —
the study's one execution lag) with cost × borrow sweeps.
**Dedup:** [62-premium-seller](../62-premium-seller/) *quotes* QYLD's
50%/58% capture as a description of one fund's shortfall — 948 makes the spread the object
of inference across fourteen funds, and shows the ratio is a biased estimator that is also
a pure re-encoding of (alpha, beta); [337-covered-call-etf](../337-covered-call-etf/) asks
where the *distribution* comes from, not what shape the payoff has;
[658-put-write-premium](../658-put-write-premium/) covers put-writing (PUTW);
[899-cash-plus-call](../899-cash-plus-call/) is the long-convexity mirror;
[900-quality-income](../900-quality-income/) and
[206-dividend-aristocrats](../206-dividend-aristocrats/) race dividend sleeves on returns.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a capture ratio promises, why the two numbers turn out to be the same number, and the ruler that reads positive on nothing at all |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the scorecard with bootstrap CIs, the HM convexity *t* against a Bonferroni bar, the fund-matched null, CAPM alphas, the Sharpe race, rank persistence, the assumption sweeps, and the live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`capture_ratio/`](capture_ratio/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
