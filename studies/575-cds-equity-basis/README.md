# Study 575 — CDS-Equity-Basis 🔀

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the CDS-equity basis predict equity returns? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The convergence effect is **real in the literature** (Kapadia-Pu 2012) and the engine is a **faithful detector** — on the planted synthetic tape the basis→forward-return slope is *t* **−5.79** (clustered by month), the decile long-short *t* **+4.78**, placebo *p* **0.0005**, and the null is flat (25-seed mean slope-*t* **−0.23**). But there is **no free real CDS tape** (single-name CDS is licensed OTC data), so it **cannot clear the REAL bar** — capped at `WEAK`. |
| **Tradability** — can you trade the gap? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The *inputs* are unreachable retail (Markit/Bloomberg CDS), the true effect is faint (corr **−0.08**) and largely arbitraged, and the short leg is the credit-worried, expensive-to-borrow tail: even on the frictionless synthetic tape, monthly costs + borrow halve the spread (gross **+1.18%/mo** → net **+0.65%/mo**). |

> **In one sentence:** when a company's CDS and its stock disagree about default risk, the *basis* between them does carry a genuine convergence signal — the credit-worried names' equities tend to catch down — but it is faint (corr ≈ −0.08), largely arbitraged in liquid names, and, fatally for a retail desk, **built on single-name CDS data that has no free feed at all**, so the machinery here works on a synthetic tape it can never point at a real one: `WEAK` × `MIRAGE`.

## What we tested

The **CDS-equity basis** convergence claim (Kapadia & Pu 2012; the credit-equity arbitrage
folklore): a name's **CDS spread** prices the credit market's view of default risk, while its
equity — via a **Merton (1974)** structural model — implies a credit spread of its own. Their
difference is the *basis*, and a *wide* basis (credit more worried than the stock) is supposed to
forecast a *falling* equity (it catches down to credit). Because clean single-name CDS is a
licensed OTC market with **no free retail feed**, this study is **synthetic-only**: we build a
deterministic name-by-month panel where a single knob (`convergence_beta`) plants the effect, then
run the full engine — a per-month z-scored basis, a pooled panel regression with **month-clustered**
standard errors, a decile long-short with an IID *t*, a **month-label-shuffle placebo**, a
sub-sample robustness sweep, costs + a punitive short borrow, and a **seed-robust (25-seed)
positive control** proving the engine catches a planted effect and stays flat at the null. The
data-availability ceiling is stated openly on the SIGNAL axis: a synthetic-only study can never be
`REAL`. *Distinct from the rates funding basis of
[382 Treasury-Basis-Trade](../382-treasury-basis-trade/) and from the equity-only distress scores
of [540 Distress-Risk-Anomaly](../540-distress-risk-anomaly/) — this is the **cross-asset
credit-vs-equity** gap.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the CDS-equity basis is, why credit and a stock can disagree about the same company, and why the gap *is* a signal but one you can't actually reach |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the per-month z-scored basis, the month-clustered pooled slope, the decile long-short with a placebo null, the sub-sample robustness, costs + borrow, and the seed-robust synthetic positive control |

The fingerprinted synthetic-control run (60 names × 95 months, planted `convergence_beta = −0.90`,
panel fp `884a4f7a09c4`, as-of 2026-06-30) is in [docs/results.md](docs/results.md); the offline
machinery lives in [`cds_equity_basis/data.py`](cds_equity_basis/data.py) — **there is no real
tape**, by design.

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`cds_equity_basis/`](cds_equity_basis/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
