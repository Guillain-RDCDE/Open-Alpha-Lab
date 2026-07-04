# Study 638 — Value-Momentum-Everywhere 🌍

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do value & momentum pay everywhere, and does the combo clear the bar? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The literature (Asness-Moskowitz-Pedersen 2013, 1972–2011) says real; **our free tape can't certify any of it**: all eight sleeve-legs (VAL & MOM × country ETFs / G10 FX / Treasury futures / commodity futures) sit at \|HAC *t*\| < 2, the famous hedge correlation is **≈ 0** (paper: ~−0.5), and the global 50/50 combo — the claim under test — lands at Sharpe **−0.29**, HAC ***t* = −1.49** (296 months, wrong sign), with nothing in either the ≤2011 or ≥2012 half. Survivor ETF panel, spot-only FX and roll-noisy continuous futures named. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The combo is **negative before costs** (−1.88%/yr gross) at ~0.65× NAV monthly one-way turnover; at 10 bps one-way + 50 bps/yr borrow on the ETF short leg the net is **−2.81%/yr** (net *t* = −2.21). Nothing to harvest. |
| **"Just diversification arithmetic?"** | ![Confirmed](https://img.shields.io/badge/Just_diversification_arithmetic%3F-Confirmed-8b949e?style=flat-square) | The realized combo Sharpe equals the textbook two-asset formula **to the 4th decimal** (−0.267 vs −0.267) — the "free lunch" is √-arithmetic that *multiplies its ingredients*: real ones in AMP's sample, ~zero on this tape, pure noise in [401 — signal-stacking](../401-signal-stacking/) (the null twin). Zero × 1.41 = zero. |

> **In one sentence:** the most famous combo claim in factor investing — value and momentum pay in every asset class and hedge each other into a free lunch — rebuilds on free data (13 country ETFs, 9 G10 spot pairs, 11 futures, 1997/2001→2026) into eight statistical-zero legs, a hedge correlation of ≈ 0, and a global 50/50 combo of Sharpe **−0.29** (HAC *t* = −1.49) that matches the two-asset diversification formula exactly — the blender works, but this tape supplies no fruit.

## What we tested

We rebuild Asness–Moskowitz–Pedersen (2013) with one uniform recipe in four sleeves: **VALUE** = the 5-year reversal (months t−59..t−12 — AMP's own non-stock value proxy), **MOMENTUM** = the classic 12-1 (months t−11..t−1), long the top third / short the bottom third, equal-weight, rebalanced monthly with **one documented execution lag** (signal at month-end *t*, position earns month *t*+1). Sleeves: 13 country equity ETFs (total-return, survivors — named), 9 G10 spot pairs vs USD (price-only, no carry — labeled), ZF/ZN/ZB Treasury futures and 8 commodity futures (excess returns from the shared continuous-futures cache, roll noise named). The Signal axis puts HAC/Newey–West *t*'s on every leg and on the global everywhere-portfolios; the third axis tests whether the combo does anything beyond the two-asset diversification formula (it doesn't — that *is* the finding, mirrored against [401 — signal-stacking](../401-signal-stacking/) where the same arithmetic stacked noise). A 50-seed random-rank placebo proves the construction is unbiased, costs are one-way × traded notional with the ETF short leg paying borrow, and a deterministic synthetic world with plantable value/momentum edges proves the machinery detects the effect when it exists. Sub-period split at 2011/2012 (AMP's sample edge). The individual premia live separately in [147 — fx-momentum](../147-fx-momentum/), [196 — long-term-reversal](../196-long-term-reversal/), [364 — fx-carry-trade](../364-fx-carry-trade/) and [31 — trade-winds](../31-trade-winds/); the **combo** claim is what's new here. As-of **2026-05-31**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "value and momentum everywhere" promises, why the 50/50 blend was supposed to be a free lunch, and why a blender with no fruit makes no smoothie — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the 8-leg ingredient matrix with HAC *t*'s, the global combo and its sub-periods, the diversification-arithmetic decomposition, a 50-seed placebo, costs × turnover, construction variants, and a planted-effect synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`value_momentum_everywhere/`](value_momentum_everywhere/). The signal pair is AMP's universal proxies (5y reversal + 12-1); the myth-check is the two-asset diversification formula. EQ panel is **survivors**, FX is **spot price-only**, futures carry **roll noise** — all named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
