# Study 887 — High-Yield Muni Premium 🏛️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does HY-muni really out-earn IG-muni on a credit premium? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | HYD beats MUB by **+20.21 bps/mo (+2.43 pp/yr)** over 17.3 live years and the excess-Sharpe edge is sign-consistent across halves (+0.07 full; +0.13 / +0.15), **but it fails the robust bar**: HAC *t* = **1.80** (< 2), the bootstrap 95% mean CI **[−1.44, +42.23]** touches zero, neither sub-era clears *t* = 2, and the spread **inverts to −78 bps/mo (*t* = −4.35) in 2022**. Directionally real, statistically thin, crisis-fragile. |
| **Tradability** — is it bankable? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Costs do **not** kill it (one-switch hold drags ≤ **3.5 bps/yr** vs **+239 bps/yr net**, fees inside the tape — so not a Mirage). It is Fragile: the Signal is thin, the payoff is crisis-fragile (**−35.6% vs −13.7%** COVID drawdown), and the one mechanical pickup — the **tax wrapper** (TEY **9.17%** vs **6.04%**; after-tax HYD **5.38%** > HYG **4.67%**) — only helps a top-bracket investor in a taxable account and leaves the after-tax Sharpe race a near-tie. |
| **"A paid credit premium in a tax-favored wrapper?"** | ![Half true](https://img.shields.io/badge/Credit_%2B_tax_wrapper%3F-Half_true-8b949e?style=flat-square) | The **wrapper** is real & mechanical (muni coupons are federally tax-exempt; TEY 9.2% vs 6.0% is arithmetic). The **credit premium over IG munis** is the part that won't certify: +2.4 pp/yr gross but *t* = 1.80, CI to zero, −78 bps/mo 2022 inversion. |

> **In one sentence:** high-yield munis really do carry a fat, tax-advantaged yield — the tax-equivalent yield is **9.2%** vs **6.0%** for taxable junk and the *after-tax* return beats it (5.4% vs 4.7%) — but the *credit* premium over investment-grade munis is thin (HAC *t* = **1.80**, bootstrap CI touching zero) and turns sharply negative when illiquidity bites (**−78 bps/mo in 2022**, a **−35.6%** COVID crash), so it earns **Weak / Fragile**, not a green stamp.

## What we tested

The claim: HY munis (HYD) pay a fat, tax-advantaged credit spread over investment-grade munis (MUB) — *a paid credit premium in a tax-favored wrapper*. The live test uses monthly total returns of **HYD vs MUB/TFI** (2009-03 → 2026-06, 208 months) with **HYG** as the taxable-HY yardstick and **BIL** as the tradable risk-free. We run the raw monthly HYD−MUB spread (HAC *t*, block-bootstrap mean CI), an excess-vs-excess Sharpe race, an era cut plus the crisis windows (2020, 2022) where muni illiquidity bites, the **tax-equivalent yield** and after-tax return/Sharpe race (income backed out as total-return minus price-return), the drawdown bill and the one-switch cost math. Exactly one documented execution lag. A deterministic synthetic world with a planted-premium knob proves the HAC/bootstrap machinery is faithful (never cited as evidence). **Dedup:** distinct from **[576 — Muni-Treasury-Ratio](../576-muni-treasury-ratio/)** (a muni/Treasury *valuation-timing* ratio), **[616 — Muni-CEF-Tax-Loss](../616-muni-cef-tax-loss/)** (a *seasonal tax-loss* effect in muni CEFs) and **[115 — Credit-Spreads](../115-credit-spreads/)** (HY spreads as an *equity-timing* signal); this is the tax-exempt cousin of **[610 — Fallen-Angels-Premium](../610-fallen-angels-premium/)** — a *within-muni credit* carry with the extra after-tax question.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a high-yield muni is, why the yield looks huge once you count the tax break, the credit spread you actually earn vs plain muni bonds, and the crash you carry for it — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC spread + bootstrap CI, excess Sharpe race, era/crisis cut, tax-equivalent yield & after-tax race, drawdowns & one-switch costs, synthetic faithful-engine control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`hy_muni/`](hy_muni/). Total-return, net-of-fee ETF tape (yfinance), as-of 2026-06-30, fingerprint `db0172501766`. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
