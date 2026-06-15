# Study 168 -- Advance-Decline

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Post-divergence 21-day SPY return is **+95.8 bps** -- *above* the unconditional **+87.3 bps** baseline. The claimed bearish effect is absent and the sign is backwards. Permutation p-val = 0.61. Survivorship bias named. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Shorting SPY on every bearish divergence signal loses ~96 bps per 21-day trade (before costs), ~14.6 signals/yr. The intended trade is directionally wrong. |
| **Coincident?** | ![Not_a_leading_indicator](https://img.shields.io/badge/Not_a_leading_indicator-8b949e?style=flat-square) | The A/D line co-moves with the cap-weighted index; divergences largely reflect the equal-count vs cap-weight mismatch, not structural breadth weakness. |

> **In one sentence:** the "generals advancing, soldiers retreating" bearish divergence signal does not work -- on 21 years of real data, post-divergence SPY returns are slightly *above* the unconditional baseline, the lookback sweep shows no consistent direction, and the permutation test confirms it is statistically typical. A Mirage in generals' clothing.

## What we tested

A century-old piece of market folklore: when the S&P 500 makes a new high but the cumulative Advance-Decline line (net daily advancers minus decliners, cumulated) does NOT confirm the new high, the rally is narrow and fragile -- only a few large caps are doing the work. We build the A/D line from 497 current S&P 500 constituents (Yahoo daily, 2005-2026; survivorship-biased panel -- named), detect every bearish price-vs-breadth divergence (index at 63-day high while A/D is not), and test 21-day forward SPY returns against the unconditional baseline. A lookback sweep (21d--252d), a permutation baseline for multiple-comparisons correction, and a drawdown comparison complete the teardown. A synthetic tape with planted divergences confirms the machinery works when a real effect exists.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the folklore, the chart of A/D vs SPY, the distribution comparison, the lookback sign-flip, why the trade is directionally wrong |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stats, permutation baseline, Bonferroni on the lookback sweep, the synthetic positive control, the cost/tradability maths |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`advance_decline/`](advance_decline/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
