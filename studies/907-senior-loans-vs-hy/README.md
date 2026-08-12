# Study 907 — Senior Loans vs High-Yield 🏦

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the seniority premium a real risk-adjusted edge? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | Loans' lower vol is real (**5.4% vs 8.1%**) and they cushion rate/spread selloffs — but the excess-Sharpe advantage is a rounding error (**+0.029**) whose **sign flips with construction** (flagship BKLN-vs-HYG is **−0.017**), its bootstrap CI **[−0.26, +0.47] straddles zero**, and the return spread is **negative** (−1 %/yr, NW *t* = −0.83). Risk-adjusted, a **wash**. |
| **Tradability** — can you bank it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The natural trade (long loans / short HY) is **negative gross** (−1 %/yr, loans earn less) and **−2 to −3 %/yr** after borrow on the short HY leg + costs. The only real edge — lower vol — cannot be captured dollar-neutral. |
| **"Seniority = a free premium?"** | ![Busted](https://img.shields.io/badge/Free_premium%3F-Busted-8b949e?style=flat-square) | Seniority + the floating coupon genuinely halve your loss in the 2015-16 energy wave (**−6.8% vs −12.1%**) and the 2022 rate shock (**−4.5% vs −14.6%**) — but you pay in lower total return, and loans gapped **worse** in the one pure liquidity crisis (COVID: **−23.8% vs −21.9%**). |

> **In one sentence:** senior loans really are the calmer, lower-vol, rate-proof cousin of high-yield and they shine when the pain is spreads or rates — but they earn **less** total return, **tie** on risk-adjusted return (a bootstrap that can't tell them apart from HY), and gap **harder** in a liquidity run, so the "seniority premium" is a genuine *volatility discount*, not a harvestable *edge*.

## What we tested

The pitch, steelmanned: *"Senior secured loans (BKLN, SRLN) sit **above** high-yield bonds (HYG, JNK) in the capital stack at a similar yield — first lien, better recovery, floating coupon. A loan sleeve gives you the same fat carry with less risk: a **seniority premium** for free."* We race the **loan** sleeve against the **HY** sleeve, both **excess of cash** (BIL), on the common window bounded by BKLN's **2011-03-03** inception (so HY doesn't get to bank the 2008 GFC), and ask: is there a real excess-Sharpe advantage (bootstrap CI clear of zero, era-robust)? A return premium (HAC *t* ≥ 2)? Does seniority protect in the credit-stress episodes (2015-16 energy wave, 2020 liquidity crash, 2022 rate shock)? And does a long-loans/short-HY trade survive borrow + costs? A deterministic synthetic world with a **planted, tunable risk-adjusted edge** (loans engineered to lower vol; null = lower vol exactly offset by lower carry) proves the estimator recovers a real edge and never manufactures one. **Dedup:** distinct from **[340 bank-loans](../340-bank-loans/)** (loans vs *rates* — the duration story), **[115 credit-spreads](../115-credit-spreads/)** (the spread as a *signal*), **[796 corporate-bond-low-risk](../796-corporate-bond-low-risk/)** (a *cross-section* quality tilt), and **[832 high-yield-credit-momentum](../832-high-yield-credit-momentum/)** (a *timing* signal) — this is a static **loans-vs-HY seniority race**. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "senior secured" buys you, why lower vol isn't a free premium, the stress table where seniority helps (and where it bites) |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | excess-vs-excess Sharpe race, the construction sign-flip, the bootstrap CI on the advantage, era cuts, the costed long-short, and the planted-edge synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run (2011–2026, Fingerprint `e09ddb919d86`): [docs/results.md](docs/results.md).

---

*Engine: [`loans_vs_hy/`](loans_vs_hy/). Total-return closes (`auto_adjust=True`), Sharpe excess of BIL. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
