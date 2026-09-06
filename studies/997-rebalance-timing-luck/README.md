# Study 997 — The Rebalance Lottery 🎰

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — how much of a strategy's result is the arbitrary choice of rebalance date? | ![Confirmed](https://img.shields.io/badge/Confirmed-2ea44f?style=flat-square) | The identical momentum rule, run at all **21 possible rebalance offsets** over 19 years, produced CAGRs from **+6.50% to +9.92%** — a spread of **3.42%** and a terminal-wealth ratio of **1.85×** between the luckiest and unluckiest start day. Nothing about the rule differed; only the day of the month it happened to trade. For scale, the rule's average edge over buy-and-hold was +0.29%/yr, so the luck is **11.6×** the edge, and 62% of the offsets beat the benchmark. A fixed-weight 60/40 shows far less — a spread of only 0.22% — because both variants hold the *same assets* and differ only in drift, while a ranking rule's variants hold different assets entirely. Any backtest of a selection rule that reports one rebalance date is reporting one draw from a distribution this wide. |
| **Tradability** — does overlapping portfolios remove it, and what does that cost? | ![Useful](https://img.shields.io/badge/Useful-2ea44f?style=flat-square) | The fix is real and it is cheap. Running all 21 offsets at once — rebalancing 1/21 of the book each day — removes the dispersion by construction, and it does not cost return: the blended portfolio delivered **+8.66%/yr at a Sharpe of 0.65**, against the average single-offset variant's +8.54% and 0.62. It also cut volatility by 0.78% and improved the worst drawdown by 2.3% against the average variant, because the sleeves are imperfectly correlated with each other. The catch a practitioner should know: it is operationally more demanding — a daily trade instead of a monthly one — and section 6 confirms it preserves genuine signal rather than diluting it away. |

> **In one sentence:** The same momentum rule spans 3.4% of CAGR depending only on which day of the month it rebalances — 11.6× its own edge — and running every offset at once removes that at no cost to return.

## What we tested

A backtest says "rebalance monthly". It does not say **which day**, and for most
published results nobody checked. This study runs an identical momentum rule at all 21 possible
rebalance offsets — changing nothing else — and measures the spread.

The spread is large, and it is larger than the strategy's own edge over buy-and-hold. That
matters because a reader shown one equity curve is being shown **one draw** from a distribution
they never see. The mechanism is made explicit by a control: a fixed-weight 60/40 through the
identical machinery shows an order of magnitude less dispersion, because its variants hold the
*same assets* and differ only in drift, while a ranking rule's variants hold *different assets
entirely*. Timing luck is a property of **selection**, not of rebalancing as such — which is why
it goes unnoticed in the asset-allocation literature and bites hardest in exactly the strategies
people are most excited about.

Then the fix, from Blitz, van der Grient & van Vliet (2010): run all offsets at once, 1/21 of the
book each day. It removes the dispersion **by construction** — there is only one blended
portfolio — and the study checks the two things that would make it a bad trade: whether it costs
return (it does not; volatility falls and the drawdown improves, because the sleeves are
imperfectly correlated), and whether it dilutes genuine signal, tested against synthetic worlds
with a *planted* momentum effect.
**Dedup:** distinct from **117-rebalancing-bands** and **969-rebalancing-bonus** (rebalancing
*policy* and its return effect), **994-small-account-lot-drag** (share indivisibility),
**996-palindrome-dates** (searching across hypotheses rather than across implementation dates)
and **860-backtest-overfitting** (parameter optimisation, whereas the offset here is not a
parameter anyone chose).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | two investors running the identical strategy a fortnight apart, and how far apart they end up |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | all 21 offsets of a momentum rule, the fixed-weight control that isolates the mechanism, a period sweep, overlapping portfolios, and a planted-signal check that the fix does not remove the edge |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`lottery/`](lottery/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
