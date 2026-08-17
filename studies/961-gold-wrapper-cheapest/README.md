# Study 961 — Which Gold 🪙

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the edge real? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | The tape hands the fee back. Cheapest − priciest (GLDM − GLD) is **+26.17 bp/yr, naive *t* = +3.18**, positive in **65/96** months and in **9/9 calendar years**, with an **iid** bootstrap CI **[+9.5, +41.9]** clear of zero; positive in both eras (+20.98 / +30.21, significant in the late one) and **+26.66, *t* = +2.69 on the fee-stable sub-window**, the one cut in which no fee assumption does any work. Across all five wrappers the fee ranking inverts the outcome ranking **perfectly** — Spearman **−1.0000**, no inversions, at the **0.0167** hard floor of an exactly enumerated five-fund permutation — with pass-through **−0.89 to −0.94**, R² **0.99**, and ten of ten pairwise spreads carrying the sign the fee sheet predicts. The mechanism is contractual, not discovered. Named limits: **the HAC *t* (+6.32) is the *anti*-conservative one here** — this residual mean-reverts (acf1 −0.42 monthly) and the HAC *t* climbs with the bandwidth, so every number above is the naive one; only the *coarse* ranking resolves (one pair clears \|*t*\| = 2, the two 22-bp pairs sit at 1.9, the four sub-10-bp gaps at \|*t*\| ≤ 0.6 against a **22.5 bp/yr** detection floor); the early era is not significant alone; the fee sheet is a labelled **ASSUMPTION** carrying hindsight and the headline pair is a selection off it; the cohort is five survivors. |
| **Tradability** — is it bankable? | ![Investable](https://img.shields.io/badge/Investable-2ea44f?style=flat-square) | Not because the tape pins the number down — the honest interval is **10 to 42 bp** — but because the act is a **purchase decision with no forecast, no timing, no turnover and no capacity limit** on the wrapper that wins, and because even at the bottom of that interval the cheap wrapper is not the worse buy: a dominance argument, not a forecast. GLDM turns over **$433m a day** (median, last 12 months), repays even a punitive 30 bp round-trip penalty in **418 days**, and the gap compounds to **2.6% of terminal wealth over ten years**. Three things it is not: not a reason to sell an appreciated holding (**20–75 years** to repay the collectibles-rate tax bill), not a long/short (gross +18.2 bp after costs, **dead above ~18 bp/yr of borrow**), and not an argument for the fee-cheapest name at size (BAR trades $21m/day — **4.8 days of ADV** for a $100m ticket). Chasing last year's tracking winner *loses* to owning the cheapest (*t* = −0.14). |

> **In one sentence:** Five trusts hold the same bullion in the same vaults and struck at the same close, and over eight years they finished in the exact reverse order of their fee sheets with no inversions — the 30 bp between the cheapest and the priciest showing up as 26 realised basis points a year, worth 2.6% of a ten-year gold sleeve — but the ladder is only legible where it is wide: the seven-basis-point rungs in the middle are indistinguishable from nothing, and the wrapper you should actually buy is the cheapest one you can still fill in size.

## What we tested

Five US physically-backed gold trusts — **GLD (40 bp), IAU (25), GLDM (10), SGOL (17),
BAR (17.5)** — over their common window **2018-06-26 → 2026-06-30** (2,013 sessions, 96
complete months). We measure each wrapper's realised tracking difference against the
equal-weight cohort mean three ways (endpoint, daily, non-overlapping monthly), establish
the **detection floor** the pairwise noise imposes, test whether the fee ranking predicts
the outcome ranking (Spearman against an **exactly enumerated** 120-permutation null, plus
a pass-through regression), put **both a naive and a HAC *t* — and both an iid and a block
bootstrap** — on the cheapest-minus-priciest spread (this residual mean-reverts, so the HAC
*t* and the block CI are the *optimistic* pair and we quote the other one), cut it by era
and by a **fee-stable sub-window**, compound
it over a ten-year hold, and race the ownership rules **excess-of-cash** with one execution
lag and one-way costs × NAV. Then the counterweights, all swept: real-tape **dollar
volume** (a $100m ticket is 0.02 days of GLD and 4.8 days of BAR), a round-trip execution
differential, the capital-gains bill on switching, and borrow on the short leg. Fees are a
labelled **ASSUMPTION** (with GLDM's 18 → 10 bp cut a second one); no headline number is
*computed* from them, though the headline pair is *selected* off them — and on day one of
the window GLDM charged 18 bp and was not yet the cheapest of the five. **Dedup:** distinct from **920-total-cost-of-ownership** (same measurement on *equity*
index wrappers, 6–7 bp gaps, a trust's cash drag inseparable from its fee, five shared
years — here it is five wrappers not pairs, a 30 bp spread, eight years, and a grantor-trust
structure with no cash drag and no securities lending), from
**913-tracking-difference-persistence** (does last year's winner *persist* — asked here only
as a counterweight, and it loses), from **959-crypto-etf-fee-war** (same stack on ten
bitcoin ETFs, where a 24/7 asset against a 16:00 strike adds a 135 bp/day clock stub gold
does not have), and from the gold *timing* studies **912 / 640 / 649 / 831** (this study
takes no view on the metal at all — the gold price cancels out of every number in it).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | five wrappers and one price tag, the perfect ranking, what a decade of the wrong one costs, the three catches, the live synthetic control |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the detection floor, three estimators and why the HAC *t* is the anti-conservative one at both frequencies, the enumerated permutation null, pass-through, era cut and fee-stable window, the excess-of-cash race, four swept counterweights |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`which_gold/`](which_gold/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
