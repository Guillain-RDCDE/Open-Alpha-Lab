# Study 1002 — The Ten Best Days 📆

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the best-days statistic true, and is its symmetric twin ever quoted? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | The arithmetic is correct. On SPY over 33 years, missing the ten best sessions cuts the annualised return from 10.83% to 8.10% — a loss of **2.74% a year** from ten days out of 8,410. The brochure stops there. Missing the ten **worst** sessions raises it to 13.78%, a gain of **2.94% a year** — **1.07× the size of the loss**. The reason is not the one you would guess. In percentage terms the ten best days are the *bigger* ones (+8.74% against -8.36%), so "crashes are larger than rallies" is simply false here. The asymmetry survives because compounding is multiplicative: removing a day multiplies the result by 1/(1+x), so the quantity that matters is log(1+x) — and in log terms the worst days are the larger (-0.0875 against +0.0835). And the two sets are not scattered through history: the median distance from a best day to the nearest worst day is **6 sessions**, against 290 when the same returns are shuffled (p = 0.000). They arrive in the same storms — the best days occurred at a median drawdown of -34.7% and in volatility 5.8× normal. Being absent for all ten best days while present for all ten worst is not an outcome any rule can produce. |
| **Tradability** — what timing accuracy would actually be needed to beat buy-and-hold? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Which makes the interesting question the one nobody asks: how accurate would a timer need to be? Take a rule that sits out 20% of all sessions — 1,682 days — and score it on the share of those days that turn out to be down days. **Choosing at random already achieves 45.2%**, the unconditional down-day frequency, so that and not 50% is the benchmark. Break-even against buy-and-hold arrives at **47.7%**, an edge of only **2.5 percentage points** over random. That sounds modest, and the modesty is the trap: the frontier is savagely steep on both sides. Random selection — sitting out 20% of days with no skill whatsoever — returns 8.59% against buy-and-hold's 10.83%, and a timer 5 points *below* random gets 4.40%. The edge must be small, positive, and sustained across every one of 1,682 decisions; being slightly wrong is far more expensive than being slightly right is profitable. For scale, missing 84 days chosen at random costs 0.12% a year while missing the 84 *best* costs 11.93% — the brochure quotes the worst case of a selection no process generates. |

> **In one sentence:** Missing the ten best days costs 2.74% a year and missing the ten worst gains 2.94% — they are 6 sessions apart in the same storms, and a timer needs 2.5 points of accuracy above random just to break even.

## What we tested

"Miss the ten best days and you lose half your return." The arithmetic is
correct; this study confirms it and then measures the three things the brochure leaves out.

**The symmetric statistic.** Missing the ten *worst* days helps by more than missing the ten
best hurts, because the worst days are larger. Same calculation, sign flipped, never quoted.

**The days are neighbours.** Best and worst days cluster in the same few weeks. Against a
benchmark that shuffles the same returns — preserving every value and the whole fat tail, and
destroying only the order — the median distance from a best day to the nearest worst day
collapses. The brochure's implied counterfactual, absent for every best day and present for
every worst, is not merely unlikely: it is structurally unavailable. A synthetic control
isolates volatility clustering as the mechanism.

**Nobody misses days at random.** Missing days chosen at random costs almost nothing; the
brochure quotes the worst case of a selection no process generates. So the study asks the honest
question instead — what hit rate would a switching rule need to break even against buy-and-hold?
That threshold is demanding, and it is a far better argument for staying invested than the ten
days are.
**Dedup:** distinct from **1007-time-diversification** (horizon and risk), **211-market-timing**
(rule performance) and **304-fat-tails** (the distribution itself); the subject here is a
specific rhetorical statistic and what its omissions are worth.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the most-quoted statistic in fund marketing is true and misleading at the same time |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the symmetric statistic, a shuffle test for clustering, drawdown and volatility context, the random-days null, and the break-even hit rate for a timer |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`bestdays/`](bestdays/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
