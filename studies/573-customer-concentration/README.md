# Study 573 — Customer-Concentration 🔗

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**Does depending on a handful of big customers make a stock riskier — and does the market pay you for that risk?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is customer concentration a risk / return factor? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The **risk** leg is emphatic *in the synthetic world*: concentrated firms have forward vol **24.5%** vs **21.2%** (spread +3.3pp, two-sample *t* **+29.4**; firm-level slope-*t* **+53.5**, corr **+0.94**), robust across every cut. But the **return** leg is soft even where a premium is *planted*: spread **+4.9%** yet two-sample *t* only **+1.61**, placebo *p* **0.10**, firm-level slope-*t* **+1.38** (corr +0.07). And there is **no free real tape** (concentration lives in paywalled 10-K / Compustat segment data) — so, capped at `WEAK`. |
| **Tradability** — does the return premium pay? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No investable customer-concentration index exists, and the return spread you'd trade doesn't survive its own noise: gross **+4.9%** → net **+3.7%** (5 bps/leg + 100 bps borrow), on a placebo *p* of 0.10. The premium is buried by the very fragility volatility it rides on. |

> **In one sentence:** customer concentration plausibly *does* make a firm riskier — the forward-volatility leg is strong and robust in the synthetic world (*t* +29 at the sort, +53 firm-level) — but the *return premium* for bearing that fragility is nearly undetectable (*t* ~1.6, placebo *p* 0.10) even when we plant it, because the same lumpy-demand fragility that raises risk also swamps the premium in noise; with no free real tape to certify against, that earns `WEAK` × `MIRAGE`.

## What we tested

The **customer-concentration** fundamental-risk claim (Patatoukas 2012; Dhaliwal, Judd, Serfling &
Shaikh 2016; Hertzel et al. 2008): a firm that sells a large share of revenue to a *few* big
customers is fragile — a lost customer can gut its cash flows — so it should carry higher forward
**risk** and, if that risk is priced, a return **premium** (or a behavioural *discount*). Because
the concentration measure lives in paywalled 10-K "major customer" / Compustat segment disclosures,
**this study is synthetic-only** and capped at `WEAK` (a `REAL` stamp needs a robust *t* ≥ 2 on a
real tape). We build a Herfindahl-style concentration score on a deterministic 400-firm panel, sort
into terciles, and test **both** legs: a two-sample *t* on concentrated-minus-diversified forward
**volatility** (the risk story) and forward **return** (the premium/discount), a **label-shuffle
placebo** null, firm-level slopes whose *sign* is the claim, a tail-fraction robustness sweep, costs
+ a short borrow, and a seed-robust synthetic positive control (25 seeds) that proves the engine
catches a planted risk effect *and* a planted premium and stays flat at the null. *Distinct from
[540 Distress-Risk-Anomaly](../540-distress-risk-anomaly/) (a balance-sheet distress sort) and
[177 Megacap-Concentration](../177-megacap-concentration/) (index-level concentration) — this is
firm-level demand-side fragility.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what customer concentration is, why one big customer is a risk, why the riskier firms *were* noisier but the market didn't clearly pay for it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the tercile sort with two-sample *t*s on both vol and return, the placebo null, the firm-level slopes, the tail-fraction robustness sweep, costs + borrow, and the seed-robust two-leg synthetic control |

The fingerprinted synthetic run (400 firms, panel fp `bc1d5db4bfa5`, as-of 2026-06-30) is in
[docs/results.md](docs/results.md); the offline machinery lives in
[`customer_concentration/data.py`](customer_concentration/data.py). **Synthetic-only**: no free real
customer-concentration tape exists — stated on the SIGNAL axis.

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`customer_concentration/`](customer_concentration/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
