# Study 342 — BDC-Yield 🏦

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the levered-equity crash risk real? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Yes: BIZD's beta to **stocks** is **+0.72** (HAC *t* = **+5.48**), and **above 1** on the worst 10% of equity days; it fell in **6 of 6** equity crashes and lost **−55%** in COVID — *far more than the S&P 500*. Beta to bonds is *negative*. |
| **Tradability** — is the ~10% yield the safe income it's sold as? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No. Only **6.2% CAGR** of the **~10% headline** survived as total return (**38% phantom**), for vol (**20%**) *above* equities and a **−55% drawdown** — a *lower* return than just owning SPY, with *worse* risk. |
| **Does the ~10% yield survive credit cycles?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | A high distribution rate on a leveraged private-credit book is not bond-like income. In the only regime that matters — the crash — BIZD behaved like *levered* equity. |

> **In one sentence:** the BDC ~10% "income" is **levered private-credit equity wearing a yield label** — the crash risk is real (BIZD lost more than stocks in 2020), only ~6% of the headline ever became wealth, and the "yield that survives credit cycles" is a **mirage**.

## What we tested

The pitch, steelmanned: *"Business development companies (BIZD) pay a ~10% yield — high income with bond-like steadiness, because they hold senior secured loans to private companies."* We take **BIZD** (VanEck BDC Income ETF, total return) apart against an **equity** proxy (SPY) and a **bond** proxy (IEF, 7-10y Treasuries) over **2013–2026** (BIZD's inception bounds the window, so its first real test was COVID). We compare the quoted distribution yield to the realised total return, measure volatility and drawdown, then run the decisive test — **who BIZD moves with into a credit/equity crash** (downside beta to stocks vs to bonds, every >10% equity selloff), with a HAC *t* on the beta and a block-bootstrap. The offline control is a three-asset world with a `bdc_beta` knob (which may exceed 1 — BDCs run leverage) that makes the BDC leg levered-equity or genuinely bond-like. **Distinct from [Study 338 (Preferred-Stocks)](../../338-preferred-stocks/)** (an unlevered junior security) and **[Study 340 (Bank-Loans)](../../340-bank-loans/)** (the loans held directly): BIZD is the *levered fund equity* sitting on top of those loans, sold on a headline distribution rate.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a "10% bond" lost more than stocks in COVID, where 38% of the yield went, the crash table |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | full vs downside beta, HAC *t* + block-bootstrap, yield-vs-total-return gap, drawdown co-movement |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run (2013–2026, joint fp `1773bea09c5d`): [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`bdc_yield/`](bdc_yield/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
