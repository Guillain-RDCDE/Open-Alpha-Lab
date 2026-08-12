# Study 897 — CPPI Floor 🧱

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the mechanical floor protect *and* improve the outcome? | ![Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square) | Two claims, opposite fates. The **floor protection is real and mechanical**: max drawdown **−20.8%** vs buy-and-hold's **−55.2%**, floor **never breached** (0), 2008 cut from −47% to −11%, and a 30-seed synthetic control confirms the engine bounds the drawdown and never fires on a null. But the implied *"insurance improves the risk-adjusted outcome"* claim is **false**: CPPI's excess-of-cash **Sharpe is 0.164 vs buy-and-hold's 0.542** (bootstrap gain **−0.379**, 95% CI **[−0.691, −0.050]**, P(win) 1.4%), **no** multiplier or floor setting beats buy-and-hold, and the dynamic re-timing **subtracts** value (spanning α **−1.56%/yr**, *t* = −2.38 in the volatile era — 2020's de-risk-into-the-crash whipsaw turned a +18% year into −5%). *Short history: BIL bounds the window to one GFC-anchored ~19-year cycle — named on the Signal axis.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | As a **return** edge there is nothing to bank — CPPI trails buy-and-hold on Sharpe and CAGR **gross, before a basis point of cost**. The drawdown cap itself is cheap (2.3×/yr turnover on SPY, survives every cost) and genuine **as insurance**, but insurance is a *cost* paid in full out of upside (**2.4% CAGR vs 10.7%**), not a paycheck — and it isn't even a hard guarantee: an overnight gap past **1/m = 20%** breaches the floor. |

> **In one sentence:** CPPI's mechanical floor genuinely caps the drawdown (−21% vs −55%, never
> breached across 2008/2020/2022) — but you pay for every point of it in upside (excess Sharpe
> **0.16 vs 0.54**, ~99% confident), no setting beats buy-and-hold, and an overnight gap still cuts
> through, so the floor is **real** and the free lunch is a **mirage**.

## What we tested

Black & Jones (1987) / Perold & Sharpe (1988) **Constant-Proportion Portfolio Insurance**: hold a
multiplier **m = 5** times the cushion (NAV minus a **80%** floor that accretes at the cash rate) in
SPY, the rest in BIL bills, rebalancing daily as the cushion moves — a self-financing mechanical floor.
We race it **excess-of-cash** against **buy-and-hold SPY** and a **matched-average-exposure static mix**
(yfinance daily total-return, 2007-05-31 → 2026-06-30), with a leverage-clean re-timing spanning alpha,
a block-bootstrap Sharpe-difference CI, a two-era cut, multiplier / floor / cost sweeps, a **gap-risk**
stress (a 1-day drop past 1/m breaches the floor), and a 30-seed synthetic control. The window is one
GFC-anchored cycle (BIL lists 2007) — named on the **Signal** axis. **Dedup:**
[617-crash-insurance-cost](../617-crash-insurance-cost/) buys puts *outright* (an options bleed);
[624-buffer-etf-cost](../624-buffer-etf-cost/) tests packaged *buffer/defined-outcome ETFs* (a capped
wrapper); [659-costless-collar](../659-costless-collar/) is a *static options collar* (sell a call to
fund a put); [30-house-edge](../30-house-edge/) is the desk's reference on why a mechanical sizing rule
is a distribution reshaping, not alpha. This study is the **cushion-multiplier dynamic floor traded in
the underlying** — no options, no wrapper. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the floor really holds, why it's paid for in full out of the upside, and the 2020 whipsaw trap |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the three-book race, the excess-of-cash identity, the re-timing spanning alpha, the bootstrap Sharpe-difference CI, the two-era cut, the multiplier/floor/cost sweeps, the gap-risk stress, and the 30-seed control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`cppi/`](cppi/). Real tape via yfinance (`auto_adjust=True` total-return), cached under
`_cache/`. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
