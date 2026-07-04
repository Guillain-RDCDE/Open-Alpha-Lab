# Study 610 — Fallen-Angels-Premium 😇

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do ejected bonds really outperform broad high-yield? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | ANGL beats HYG by **+18.30 bps/mo** over 14.2 live years (HAC *t* = **2.44**); the CAPM-style alpha is **+15.03 bps/mo at *t* = 2.36** and holds at ***t* = 2.36–2.92** across every control (JNK benchmark, rate factor, IG credit). Caveats named: FALN's shorter 2016→ window alone is *t* = 1.43 (same sign); the 2019→2026 half alone is *t* = 1.13 (decay **not** certified, Welch *t* = 0.83). |
| **Tradability** — can you actually collect it? | ![Investable](https://img.shields.io/badge/Tradability-Investable-2ea44f?style=flat-square) | The premium comes pre-packaged in a $3bn, penny-spread, 0.35%-fee ETF: **one switch** (sell HYG, buy ANGL), cost drag ≤ **1.4 bps/yr** vs a **+218 bps/yr net** spread, fees already inside the tape. The admission price is **risk**, not friction: β = 1.12 and a **−29.3% vs −22.0%** COVID drawdown. |
| **"Just longer duration + lower-quality beta?"** | ![Busted](https://img.shields.io/badge/Just_duration_%2B_beta%3F-Busted-8b949e?style=flat-square) | Duration: after the HY control the rate loading is **β_IEF = +0.011 (*t* = 0.2)** — the alpha doesn't move. Beta: β_HYG = 1.12 explains only ~3 of the 18 bps/mo; the remaining **+15 bps/mo alpha** survives every tilt control. The forced-seller story, not the tilt story, fits the tape. |

> **In one sentence:** the fallen-angel premium is one of the rare packaged-carry claims that clears the bar live — the actual ETF (ANGL) has beaten the actual benchmark (HYG) by **+2.18 pp/yr for 14 years** with a HAC *t* of **2.4**, the alpha is **not** a duration or quality tilt in disguise (rate beta ≈ 0 after the HY control), and it is collectible in one negligible-cost trade — you are simply paid for wearing a deeper crash (−29% vs −22% in March 2020) and an episodic, downgrade-wave-driven payoff.

## What we tested

The claim: bonds ejected from investment-grade indices get dumped by **forced sellers** (IG mandates and insurance capital rules *must* sell on downgrade), enter high-yield oversold, and outperform the broad HY market — the *fallen-angel premium* (Ben Dor & Xu 2011; index studies since 1997, which are **context only** here). The live test: monthly total returns of **ANGL vs HYG/JNK** (2012-05 → 2026-06, 170 months) with **FALN** (2016→) as an independent-issuer confirmation. We run the raw monthly excess spread (HAC *t*), a CAPM-style excess-vs-excess alpha on a tradable rf (BIL), then the **duration/quality honesty check** (add IEF, then LQD — does the alpha die?), a midpoint decay test (Welch on the half-split difference), the drawdown comparison, and the one-switch cost math. Exactly **one execution lag** (entry at the 2012-04-30 month-end close, returns accrue from May 2012). A deterministic synthetic world with a planted-alpha knob proves the HAC machinery is faithful (never cited as evidence). **Distinct from [Study 115 — Credit-Spreads](../115-credit-spreads/)** (do HY spreads *predict equities*? — cross-asset timing) **and [Study 340 — Bank-Loans](../340-bank-loans/)** (are floating-rate loans a *safe bond substitute*? — a duration-for-credit risk swap): this is a *within-credit selection* claim — who you hold inside high-yield.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | who is forced to sell a downgraded bond and why, what "buying the dumped" earns, the crash you carry for it, and why one boring ETF switch collects the whole thing — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC excess spread, CAPM-style alpha, the IEF/LQD tilt controls, FALN confirmation, half-split decay test, excess Sharpe race, drawdowns, cost drag, synthetic faithful-engine control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`fallen_angels_premium/`](fallen_angels_premium/). Total-return, net-of-fee ETF tape (yfinance), as-of 2026-06-30, fingerprint `2009967185e2`. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
