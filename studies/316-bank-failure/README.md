# Study 316 — Bank-Failure 🏦

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | SPY +20d mean CAR **−5.50%** but HAC/plain *t* = **−0.93** (n=11), 95% CI **[−17.3%, +3.9%]**, and the **median is +3.2%** — indistinguishable from noise. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A "buy the blood" overlay loses **−2.79%/trade gross** before a cent of cost; 12 trades in 40 years — nothing to scale. |
| **A reliable warning?** | ![Misattributed](https://img.shields.io/badge/Misattributed-8b949e?style=flat-square) | The bearish average is **the autumn-2008 cluster** (−15.5%); strip it and the other six failures averaged **+2.8%**, and the 2023 SVB/Credit-Suisse crisis *bounced* **+4.6%**. |

> **In one sentence:** a headline bank failure tells you almost nothing about where the market goes next — the only reason the average looks like a "warning" is that one cluster of failures happened to be Lehman's autumn, and on a sample of eleven you cannot certify anything.

## What we tested

The folklore cuts both ways. The contrarians quote Rothschild — *"buy when there's blood in the streets"* — and Buffett's 2008 *Buy American* op-ed: a bank blowing up is peak panic, so it's a buy. The cautious camp says a bank failure is the **first visible domino** of contagion (Lehman → the GFC). We take both literally with a classic **event study**: line up SPY and the financials sector (XLF) around a hardcoded table of public-knowledge bank-failure dates — Continental Illinois, Bear Stearns, Lehman, WaMu, Wachovia, MF Global, SVB, Signature, Credit Suisse, First Republic — measure the cumulative move out to +20 trading days, and race it against a random-date placebo (a **synthetic control**), with a deterministic planted-drift tape as the positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what each failure actually did next, why the average lies, and the 2008-vs-2023 split in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-event CARs, HAC *t*, block-bootstrap CI, the placebo-clustering confound, the overlay's gross/net, the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`bank_failure/`](bank_failure/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
