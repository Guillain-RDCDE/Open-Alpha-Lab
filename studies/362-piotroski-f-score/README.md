# Study 362 — Piotroski-F-Score 📊

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the 9-point score separate winners from losers? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | High-minus-low spread is **positive (+2.6%/yr)** but at HAC **t = 0.62 / placebo p = 0.26** it is indistinguishable from random portfolios — and **non-monotone**: the 8-9 bucket trails 6-7, and the spread **flips sign** across thresholds. Literature-real (in small-cap deep value), but this large-cap survivor tape can't certify it. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Net of costs + borrow the spread is **+1.7%/yr at t = 0.41**, and the "winner" leg (**+14.8%/yr**) actually **trails the equal-weight basket** (+16.4%). On the universe where the edge is documented (micro-cap, hard-to-borrow value), frictions are *worse*. No NAV-scale edge. |
| **Winners vs losers?** | ![Busted](https://img.shields.io/badge/Winners_vs_losers%3F-Busted-8b949e?style=flat-square) | On this tape the high-score firms don't beat the market; the score mostly **sidesteps a thin tail of losers** and is non-monotone at the top. A sensible *quality screen* — not the winner-picking machine the headline sells. |

> **In one sentence:** Piotroski's F-Score is a real factor *in its native habitat* — small, illiquid, deep-value firms — but lifted onto a survivor-biased large-cap basket it stops separating winners from losers: the high-minus-low spread (+2.6%/yr, HAC t = 0.62) is indistinguishable from random stock-picking, the "winners" leg trails the equal-weight market, the score is non-monotone at the top, and costs + borrow finish whatever is left.

## What we tested

We rebuild **all nine binary F-score points** — four profitability, three leverage/liquidity, two operating-efficiency — from **EDGAR companyfacts** (`data.sec.gov`, public, no key) for a fixed basket of large/mid-cap US firms, sort each fiscal year by score, and hold a **high-minus-low** (≥7 vs ≤4) cross-sectional portfolio the *following* calendar year (a conservative reporting lag), with forward returns from yfinance. We judge it with a **Newey-West t**, a **placebo null** of random same-size portfolios, a **monotonicity** check across the 0–9 ladder, and costs with **borrow on the short**. The basket is a **survivor** panel (named on the Signal axis — every firm lived to 2026, a mild upward tilt that shrinks the loser tail), and only the ~21 firms with a clean COGS/gross-margin line clear the all-nine intersection, so this is a deliberate test of the F-score **outside** the small-cap deep-value universe where it was discovered. A deterministic synthetic control with an *injected* edge confirms the engine is faithful **and** that this panel can't reach significance unless the planted edge is real.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the nine checks are, why "high score = winner" is mostly the market in disguise, and why the top bucket trailing the middle gives the game away — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | F-score construction from EDGAR, high-minus-low vs the equal-weight benchmark, a Newey-West t + placebo randomization null, the monotonicity ladder, costs + borrow, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`piotroski_f_score/`](piotroski_f_score/). The basket is an explicit **survivor** panel (named on the Signal axis), and the universe is large-cap — **not** the small-cap deep-value habitat where Piotroski's edge is documented. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
