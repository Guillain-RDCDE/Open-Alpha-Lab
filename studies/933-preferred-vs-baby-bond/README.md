# Study 933 — Same Issuer, Two Ladders 🪜

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the seniority premium priced? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Not measurably — and neither is the risk it was meant to pay for. The pairwise same-issuer spread is **+3.10%/yr, HAC *t* = +0.55**, bootstrap CI **[−6.80%, +13.57%]**; the Sharpe advantage **+0.166** has CI **[−0.47, +0.80]**; the spread **flips sign** by era (−1.16% / +7.17%) and is **−0.43%/yr** on the seven-year two-pair extension. Drop the **two distressed issuers** and the whole thing inverts: spread **−0.60%/yr**, Sharpe advantage **−0.116**, vol ratio **0.94 daily / 0.79 monthly**, junior rung riskier in **0 of 6** pairs, drawdowns −20.1% vs −19.3%. The 6/8 "riskier" count is a **daily-tape artefact** (4/8 weekly, 2/8 monthly; CMS-PB is 15.5% stale prints, AC1 −0.28, and its 24.0% daily vol is 11.4% monthly). *Survivorship named (both rungs must still be listed; the one measurable redeemed pair spreads **−10.63%/yr**), 4.4-year window, and in 6 of 8 pairs the seniority step is one rung or less.* |
| **Tradability** — can you bank it? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The dollar-neutral long-preferred / short-bond leg earns **+4.29%/yr gross, Sharpe 0.28, *t* = 0.65**, decaying to **Sharpe 0.09 at a 300 bp borrow assumption**, with a −26% drawdown and a −91% single-name leg inside it. Cost is irrelevant (it cancels between the ladders); the missing signal is the problem. And the broad preferred index **lost 4.75%/yr to the baby-bond basket** over the very window our eight preferreds won. |

> **In one sentence:** hold the two listed rungs of the *same* balance sheet and you are paid
> **+3.1%/yr with a *t* of 0.55** for standing further back — a number two distressed issuers
> carry entirely, which turns **negative** once they leave; and the consolation prize ("at least
> the junior rung is reliably riskier") is a thin-tape artefact that dies with them.

## What we tested

Eight US issuers list **both** a **preferred** and an exchange-traded **baby bond** ($25-par
notes): CMS, Duke, Brookfield Renewable, Brookfield Infrastructure, B. Riley, Oxford Lane,
Eagle Point Credit, Babcock & Wilcox. Each is an equal-weight ladder, monthly rebalance with
**one execution lag**, **25 bps one-way** (PROXY, swept 5–100 bps), both **excess-of-cash**
(BIL), on daily **total-return** closes, **2022-02-02 → 2026-06-30**. The headline is the
**pairwise, same-issuer** difference — a statistic, not a portfolio — which cancels the
obligor's credit factor and leaves the price of seniority alone: HAC *t*, joint block bootstrap,
era cut, per-issuer table, cost and borrow sweeps, five stress windows, a seven-year two-pair
extension, an alternative-rung cross-check, a redeemed-pair survivorship probe, the PFF sector
fallback — plus the two probes that decide it: a **frequency/staleness audit** (is the risk step
real, or bid-ask bounce?) and a **concentration probe** (does anything survive dropping the two
distressed obligors?). A planted +6%/yr synthetic premium fires 20/20 seeds; the null fires 0/20.
**Dedup:** distinct from **[338 preferred-stocks](../338-preferred-stocks/)** (PFF vs *stocks and
Treasuries*), **[909 preferred-reset-premium](../909-preferred-reset-premium/)** (*variable vs
fixed coupon* within the sleeve) and **[907 senior-loans-vs-hy](../907-senior-loans-vs-hy/)**
(seniority at *index* level across *different* borrowers) — this one never leaves the issuer.
As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | two rungs of one ladder in plain language, why the safer rung didn't save you in 2022 or COVID, the two companies that carry the whole average |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the pairwise estimator, HAC *t* + joint block bootstrap, the frequency/staleness audit, the concentration probe, the Jensen trap, borrow sweep, survivorship probe, the live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run (2022–2026, Fingerprint `9ae51d88019e`): [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`two_ladders/`](two_ladders/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
