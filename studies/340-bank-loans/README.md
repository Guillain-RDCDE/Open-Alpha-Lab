# Study 340 — Bank-Loans 🏦

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the rate protection real? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Yes: BKLN's beta to long Treasuries is **−0.055** (HAC *t* = **−2.95**) — rate risk ≈ 0. Through the 2020–2023 repricing that cost TLT **−48.4%**, BKLN *gained* **+13.9%**. |
| **Tradability** — is it the safe bond substitute it's sold as? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | A real but modest income sleeve (CAGR **3.73%**, vol **5.8%**) — yet it fell in **7 of 7** equity crashes and gapped **−23.8%** in the 2020 liquidity shock. Credit + liquidity risk, thin in stress. |
| **A free lunch vs bonds?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The risk didn't vanish — it moved from *duration* to *credit*. Beta to stocks **+0.202** (HAC *t* **+4.48**); equity-beta − rate-beta **+0.257**, 95% CI **[+0.175, +0.349]**. |

> **In one sentence:** floating-rate bank loans **really do** shrug off rising rates — that part is true and certified — but the risk simply changed costume from *interest-rate* to *credit + liquidity*, so as a "safe alternative to bonds" it's a **mirage that springs in a recession, not a hike**.

## What we tested

The pitch, steelmanned: *"Worried about rising rates? Rotate out of bond funds into **floating-rate senior bank loans** (BKLN). The coupon resets with short-term rates, so the price barely falls when rates rise — a fat yield with almost none of the interest-rate risk that hammers a normal bond fund."* We take **BKLN** (Invesco Senior Loan ETF, total return) apart against a **long-duration** Treasury (TLT), an **intermediate** Treasury (IEF) and **equities** (SPY) over **2011–2026** (BKLN's inception bounds the window). Two clean tests: the **rate test** (every >5% rate-driven Treasury selloff — does BKLN shrug it off?) and the **credit test** (every >10% equity crash — does it cushion or pile on?), with a HAC *t* on each beta and a block-bootstrap on the rate-vs-credit loading. The offline control is a four-asset world with a `dur` knob (rate sensitivity) and a `credit_beta` knob (equity loading). **Distinct from [Study 338 (Preferred-Stocks)](../../338-preferred-stocks/)** (a perpetual junior equity-hybrid: *long* duration + equity tail) — bank loans are the mirror case (*zero* duration + credit tail) — and from **[Study 97 (60/40)](../../97-balancing-act/)** (an *allocation* race).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a "rate-proof" fund still lost 24% in 2020, the rate test vs the credit test, the two crash tables |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | rate-beta vs credit-beta, HAC *t* + block-bootstrap on the duration-for-credit swap, downside beta, capacity & liquidity |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run (2011–2026, joint fp `a4d0ad1fb115`): [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`bank_loans/`](bank_loans/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
