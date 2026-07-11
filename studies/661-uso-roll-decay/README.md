# Study 661 — USO-Roll-Decay 🛢️📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does USO structurally lose to the headline oil price? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | USO lost **80.5%** while CL=F (the front-month print every ticker calls "oil") was **+0.8%** over the same **20.22 years**, and the full-sample daily gap looks decisive at first glance (naive *t* = **-2.11**, NW(5) *t* = **-2.33**). But it is not a persistent per-year decay: **76.7%** of the entire divergence comes from just **4.5%** of days, inside the hardcoded 2009/2020 super-contango windows; drop 2020 alone and the naive *t* falls to **-1.15** (no longer separately certifiable); the calendar-year sign splits **13/21** negative, not "almost every year"; and the most recent **5.5 years running (2021 → 2026)** reverse hard **positive** — USO actually beat CL=F by **+14.8%/yr** (naive *t* = **+2.52**, NW(21) *t* = **+4.47**, bootstrap 95% CI **[+8.7%, +21.8%]/yr**, entirely excluding zero). The roll mechanism is real, but it cuts both ways with the curve's shape (contango vs. backwardation) — this is a regime-dependent wedge that hurt badly twice, not the one-way "mechanical loser" the folklore claims. |
| **Tradability** — can you deploy "long spot / short USO"? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | A constant-notional, monthly-rebalanced long-CL=F/short-USO book *looks* tradable at a glance (net Sharpe **0.37** at 5 bps + 0.75%/yr borrow) — until you strip the two named crisis windows: net return collapses to **+0.27%/yr at Sharpe 0.02**, statistically nothing, and the book would have run net *negative* through the entire 2021-2026 backwardation stretch. The entire realized P&L is two unforecastable historical storage crises, against a **-63%** max drawdown even in the full backtest and thin short-borrow exactly when the trade matters. |
| **"Does USO let you own oil the way retail investors think?"** | ![Busted](https://img.shields.io/badge/Owns_oil%3F-Busted-8b949e?style=flat-square) | Down 80% vs a flat headline price over the first 15 years, then **+14.8%/yr ahead of it** over the last 5.5 — and on the day oil made history (WTI front-month settling at **-$37.63**, 2020-04-20) USO barely flinched, **-10.9%**, not -306%. Whichever direction, it was never holding "the oil price" in the first place. Not a spot-crude proxy in any regime. |

> **In one sentence:** two real, un-forecastable storage crises (2009, 2020) did **76.7%** of a headline 80% cumulative loss, but the "mechanical, structural" framing does not survive scrutiny — drop 2020 and the daily *t* is no longer significant, only **13/21** calendar years are negative, and the most recent **5.5 years straight** show USO *beating* the oil price by **+14.8%/yr** (its own real *t*-stat) as the curve flipped from contango to backwardation: the decay is **Weak** (real but not persistent/mechanical), the carry trade is a **Mirage**, and "USO = oil" is **Busted** in every direction it's tried.

## What we tested

USO holds WTI futures, not crude — when the curve is in contango, its monthly roll mechanically
sells cheap and buys dear, a drag the fund pays that a "spot oil" holder never would. We race
USO's split-adjusted close against **CL=F** (the continuously-rolled NYMEX front-month print —
the exact number retail investors see quoted as "oil") from USO's 2006-04-10 inception through
the last complete month, testing the cumulative divergence, the daily roll-drag (naive + HAC *t*,
block-bootstrap), whether the decay concentrates in two hardcoded 2009/2020 super-contango
regimes, **whether it persists** (calendar-year sign count, a re-estimate excluding all of 2020,
and a strict post-2020 subsample — the adversarial check that overturned the original "mechanical
decay" framing to Weak), the single most extreme day in oil-market history (2020-04-20's -$37.63
settlement) as its own case study, and a costed, borrow-named "long spot / short USO"
carry-capture book. A deterministic synthetic control with a tunable planted drag proves the
machinery is unbiased.
**Dedup:** siblings [100-melting-ice](../100-melting-ice/) (general commodity contango decay),
[226-crude-seasonality](../226-crude-seasonality/) (calendar effects, not the futures-spot
wedge), [375-vxx-roll-decay](../375-vxx-roll-decay/) (same mechanism, VIX futures) and
[619-bito-roll-drag](../619-bito-roll-drag/) (same family, bitcoin, monthly CME roll) — none of
them benchmark USO against the exact price its own holders think it tracks. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why an oil ETF doesn't hold oil, why "buy low sell high" flips upside down on the monthly roll, what actually happened the day oil went negative — and why the obvious trade still doesn't pay, in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the naive/HAC *t*'s and block-bootstrap on the daily gap, the contango-stress regime decomposition, the April-2020 case study, the costed carry-capture book with its ex-crisis split, and the planted-drag synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`uso_roll_decay/`](uso_roll_decay/). CL=F is the continuously-rolled NYMEX front-month
futures print (no free physical-spot series exists) — exactly the number the "USO tracks oil"
claim is actually about. **Not investment advice** — research & education. See
[LICENSE](../../LICENSE).*
