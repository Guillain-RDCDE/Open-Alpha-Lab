# Study 516 — Dividend-Month-Premium 💵

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there a premium in predicted-dividend months? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | Predicted-dividend months earn **+40.31 bps/month** more than non-dividend months (Welch *t* = **2.56**); the within-firm **per-name premium** is **+52.28 bps/month** at one-sample ***t* = 3.63**, random-calendar placebo **p ≈ 0.001**, robust to the prediction threshold (per-name *t* = **3.5–3.8**). Clears the **t ≥ 2** bar on the real tape. Carries an explicit **survivorship** caveat (the basket is steady-payer survivors, a mild upward tilt). |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | A long-only "hold only in predicted-dividend months" overlay stays **net-positive** and beats buy & hold at every cost — but the surplus over simply staying invested is **thin** (~**1.6 pp/yr** at 5 bps, **~0.2 pp/yr** at 10 bps) on a **30-name survivor basket**. A real premium, a fragile vehicle — not INVESTABLE. |
| **"Really predictable in advance"?** | ![Confirmed](https://img.shields.io/badge/Predictable_in_advance%3F-Confirmed-8b949e?style=flat-square) | The **past-only predicted** flag (no look-ahead) is **as strong as** the contemporaneous **actual ex-div-month** flag (*t* = **3.63** vs **3.14**). You don't need to know *when* the dividend lands — only that the cadence says this is a payment month. Hartzmark-Solomon's predictability claim holds. |

> **In one sentence:** the Hartzmark-Solomon dividend-month premium is one of the rare academic factors that survives an honest replication — on a fixed 30-name large-cap survivor basket, predicted-dividend months out-earn the rest by **+40 bps/month** (per-name one-sample *t* = **3.63**, placebo *p* ≈ **0.001**), the effect is genuinely **predictable in advance** from the payment cadence (the past-only flag is as strong as the look-ahead one), yet the *tradable* surplus over simply staying invested is thin (~1.6 pp/yr) and rests on steady-payer survivors — so **Real, but Fragile**.

## What we tested

We rebuild Hartzmark & Solomon (2013) as a clean calendar study on a fixed **30-name large-cap survivor basket**: from each name's full dividend history we learn its **payment-month cadence**, flag every month the name is **predicted** to pay (using only payments *strictly before* that month — no look-ahead), and compare its predicted-dividend-month return against its own non-dividend months. The Signal axis tests the pooled premium with a Welch *t* and the within-firm **per-name** premium with a one-sample *t*, plus a 20,000-draw same-density random-calendar placebo. The third axis compares the **predicted** (past-only) flag against the **actual ex-dividend-month** flag to prove the premium is predictable *in advance*. Tradability charges one-way costs × the per-predicted-month round trip on a long-only overlay vs buy & hold. A deterministic synthetic control with a *planted* in-month premium confirms the engine is faithful and that zero edge cannot fake significance. Prices are **total-return** adjusted, so the premium is a genuine price effect, not the mechanical cash payment. Survivorship (the basket is names still trading *and still paying* in 2026) is named on the Signal axis. As-of **2026-05-31** (last complete month).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "the month a stock is *predicted* to pay" means, why prices drift up in those months (yield-seeking demand), why it's genuinely knowable in advance, and why the edge over just staying invested is thin — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-month premium + within-firm per-name one-sample *t*, a same-density random-calendar placebo, the predicted-vs-actual predictability test, costs × turnover on the overlay, threshold robustness, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`dividend_month_premium/`](dividend_month_premium/). The signal is the **past-only predicted-dividend-month** flag learned from each name's payment cadence; the myth-check is the contemporaneous **actual ex-div month** flag. Basket is **survivors** (steady payers) — named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
