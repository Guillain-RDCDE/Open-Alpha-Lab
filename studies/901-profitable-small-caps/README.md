# Study 901 — Profitable Small-Caps 🌱

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do profitable small caps beat plain small caps on risk-adjusted return? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | On the live ETFs (2017-06→2026-06, excess-of-cash) **CALF and XSHQ post Sharpe 0.39 — *below* plain IWM/IJR (0.43)** and far below SPY (0.71). Every quality-minus-plain Sharpe difference is **negative** with a bootstrap CI straddling zero and **HAC *t* ≈ 0**; the size/market decomposition leaves **no alpha** (+0.9 %/yr at *t* = 0.27 — CALF is β ≈ 1.06 small-cap beta); the tiny gap is **not era-robust** (sign flips −0.135 → +0.045). The synthetic control recovers a planted edge at *t* = 2.6, so this is a true null, not a dead detector. |
| **Tradability** — is any of it bankable? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Nothing to bank, and what looks like a candidate is pure small-cap beta. Costs deepen the hole: costed net gap −0.05 Sharpe; long-quality/short-plain isolation trade nets **−0.6 %/yr (CALF)** / **−2.0 %/yr (XSHQ)** after 0.70 %/yr borrow+costs. CALF's 0.59 % ER is a standing headwind. |
| **"The size effect, cleaned" — harvestable via ETF?** | ![Busted](https://img.shields.io/badge/Cleaned_size_premium%2C_tradable%3F-Busted-8b949e?style=flat-square) | Over 2017–2026 **large-cap SPY beat every small-cap flavour, cleaned or not**. The quality screen didn't rescue the size premium into a tradable edge; a long-only profitable-small ETF simply didn't pay it. |

> **In one sentence:** Asness et al. are surely right that the size premium in the *stock
> cross-section* lives in profitable names — but packaged as a long-only ETF, CALF and XSHQ
> did **not** out-Sharpe plain small caps (0.39 vs 0.43), carry no cleaned alpha (β ≈ 1.06
> small-cap beta, *t*(α) = 0.27), aren't era-robust, and lose to large caps outright — so the
> "cleaned size premium" is **not** harvestable this way after costs.

## What we tested

The claim (Asness-Frazzini-Israel-Moskowitz-Pedersen 2018, *Size Matters, If You Control Your
Junk*): the size premium is weak until you control for quality — junky small caps drag it to
zero — and huge once you hold quality fixed. We put it in **tradable** form: do the flagship
**profitable / high-quality small-cap ETFs — CALF** (FCF cash cows) and **XSHQ** (S&P
SmallCap quality composite) — beat **plain small caps** (IWM Russell 2000, IJR S&P 600) and
**SPY** on **excess-of-cash Sharpe** (every leg minus BIL, the T-bill cash leg), controlling
for the size/market tilts and net of realistic costs? Inference is annualised excess-Sharpe
with a **paired circular-block bootstrap** on the Sharpe difference, **Newey-West HAC** *t* on
the daily return difference and on a size/market beta decomposition, an era cut (pre-/post-2021),
a calendar-year table and a costed isolation trade. A deterministic synthetic world with a
**planted, tunable quality edge** proves the Sharpe-race machinery is unbiased (null stays
dark, planted edge recovered at *t* = 2.6). **Short-history caveat, on the Signal axis:** CALF
and XSHQ are 2017-vintage — ~9 years of live tape (COVID + 2022 bear, no pre-GFC cycle).
As-of **2026-06-30**. **Dedup:** distinct from [513-size-effect](../513-size-effect/) (the
*raw* small-minus-big premium), [657-larry-portfolio](../657-larry-portfolio/) (small-**value**,
not quality), [242-quality-minus-junk](../242-quality-minus-junk/) (the QMJ factor across all
caps — this is its small-cap-ETF slice) and [362-piotroski-f-score](../362-piotroski-f-score/)
(single-name accounting score, not an ETF race).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "profitable small caps" is supposed to beat plain small caps, what the two cash-cow/quality ETFs actually did, and why large caps quietly won the decade — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | excess-of-cash Sharpe races on the common window, paired Sharpe-difference bootstraps, HAC *t* on the daily difference, the size/market beta decomposition (no cleaned alpha), the era cut, the costed isolation trade, and the planted-edge synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`profitable_small/`](profitable_small/). Every Sharpe is excess of BIL (the cash leg),
raced on the common window all contestants share; costs charge the ER gap plus one-way spreads
(and borrow on the short leg of the isolation trade). **Not investment advice** — research &
education. See [LICENSE](../../LICENSE).*
