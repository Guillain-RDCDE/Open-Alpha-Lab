# Study 577 — MBS-OAS-Signal 🏠

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does MBS-OAS widening lead risk-off? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The folklore is *coherent* and the engine is faithful: on the planted synthetic world the forward-return-on-OAS-change slope is **−0.88 %pt/sd** (*t* **−11.4**, placebo *p* **0.0005**), flat at the null (*t* **+0.26**, *p* **0.80**). But **there is no free agency-MBS OAS tape** — the canonical series are licensed (ICE BofA, Bloomberg); FRED has mortgage *yields*, not OAS. No real *t* ≥ 2 is reachable, so this is capped at **Weak**. Only the *change* leads — the *level* is flat (*t* −0.29). |
| **Tradability** — could you harvest it? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Even where the lead is real, the raw material is unreachable: a retail investor can't subscribe to the OAS feed, can't trade the MBS index cheaply, and gets OAS with a publication lag. A risk-off overlay lifts Sharpe **0.59 → 1.07 net** *on the planted world* — but that world isn't for sale. |

> **In one sentence:** the "mortgage-spread widening warns of trouble" story is plausible and its machinery is provably detectable — on a synthetic tape the standardised weekly OAS change front-runs risk-asset returns at *t* −11.4 and a de-risk-on-widening overlay nearly doubles the Sharpe — but the option-adjusted spread is a **licensed vendor series with no free equivalent**, so there is no real tape to certify it (**Weak**) and no reachable input to trade it (**Mirage**).

## What we tested

The cross-asset folklore that **agency-MBS option-adjusted spread (OAS) widening leads risk-off**:
when mortgage investors demand more yield over Treasuries (net of the prepayment option), that
risk-off move is supposed to front-run weakness in **equities and credit**, so a rising OAS should
predict *low* forward returns. We build a deterministic weekly generator whose single knob
`lead_beta` plants (or withholds) an OAS-change → next-week-return lead, then run the honest battery:
a predictive OLS whose *sign* is the claim, a **label-shuffle placebo** null, a tradable **risk-off
timing overlay** with costs, a **robustness sweep** across signal definitions (change vs level vs
4-week change), and a **seed-robust (25-seed) synthetic positive control**. Because **no free OAS
series exists** (ICE BofA / Bloomberg are licensed; FRED has mortgage yields, not OAS), the study is
**synthetic-only** and the data wall is stated openly on the SIGNAL axis — capping it below `REAL`.
*Distinct from [115 Credit-Spreads](../115-credit-spreads/) (the corporate HY/IG spread) and
[05 Twin-Spread](../05-twin-spread/) (a relative-value pair): this is specifically the agency-MBS
OAS as a cross-asset lead.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what an option-adjusted spread is, why widening is meant to warn, why we can only test it on a made-up tape, and what the machinery says when the effect *is* planted |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the predictive slope + *t*, the placebo null, the risk-off timing overlay net of costs, the change-vs-level robustness sweep, and the seed-robust synthetic positive control |

The fingerprinted synthetic headline run (779 weekly rows, 2011 → 2025, planted `lead_beta = -0.9`,
panel fp `5a74734302c4`, as-of 2026-06-30) is in [docs/results.md](docs/results.md); the offline
machinery lives in [`mbs_oas_signal/`](mbs_oas_signal/).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`mbs_oas_signal/`](mbs_oas_signal/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
