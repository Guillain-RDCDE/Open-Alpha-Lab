# Study 338 — Preferred-Stocks 🎩

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the equity-like crash risk real? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Yes: PFF's beta to **stocks** is **+0.55** (HAC *t* = **+5.58**), rising to **~1.0** on the worst 10% of equity days; it fell in **7 of 7** equity crashes and lost **−63.7%** in the GFC — *more than the S&P 500*. Beta to bonds is *negative*. |
| **Tradability** — is it the safe income sleeve it's sold as? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No. You earn a **bond's return** (CAGR **3.88%** vs IEF **3.26%**) for **equity-grade risk** (vol **18.7%**, max drawdown **−65.5%**, *deeper than SPY*). The "bond-like safety" evaporates in every crash. |
| **"Bond-like safety"?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Senior-to-common + a fat coupon ≠ a bond. In the only regime where it matters — the crash — PFF behaves like stock, not like a bond. |

> **In one sentence:** preferred shares are **equity-grade crash risk wearing a bond's coupon** — the risk is real (PFF lost more than stocks in 2008), and the "bond-like safety" the label sells you is a **mirage**.

## What we tested

The pitch, steelmanned: *"Preferred shares (PFF) are a third asset class — bond-like safety with an equity-like yield. A fixed ~6% coupon, senior to common stock, so you get high income with much less risk than equities."* We take **PFF** (iShares Preferred & Income Securities ETF, total return) apart against an **equity** proxy (SPY) and a **bond** proxy (IEF, 7-10y Treasuries) over **2007–2026** (PFF's inception bounds the window, so its first test was the GFC). We measure yield/return, volatility and drawdown, then run the decisive test — **who PFF moves with into the crash** (downside beta to stocks vs to bonds, every >10% equity selloff), with a HAC *t* on the beta and a block-bootstrap. The offline control is a three-asset world with a `pref_beta` knob that makes the preferred leg equity-in-disguise or genuinely bond-like. **Distinct from [Study 97 (60/40)](../../97-balancing-act/)** (an *allocation* race) and **[Study 69 (Safe-Haven)](../../69-safe-haven/)** (a *hedge*): this is a single-instrument *identity* test.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a "bond" lost more than stocks in 2008, the coupon-for-risk trade, the crash table |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | full vs downside beta, HAC *t* + block-bootstrap, drawdown co-movement, the financials-concentration read |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run (2007–2026, joint fp `225bc4dab397`): [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`preferred_stocks/`](preferred_stocks/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
