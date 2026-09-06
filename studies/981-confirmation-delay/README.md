# Study 981 — The Price of Waiting ⏳

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does confirmation reduce whipsaw as advertised? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Yes, mechanically, and by a lot. Requiring five consecutive days of agreement cuts the share of round trips closed inside a month from **78%** to **30%** and the number of trades from 7.1 to 1.9 a year, across 12 tape × signal cells; the reduction holds on **100%** of them. The fast signal is where it bites: RSI's whipsaw share falls from 84% to 34%, while the 200-day average — which barely whipsaws to begin with — has little to gain. |
| **Tradability** — is the trade-off ever worth taking? | ![Investable](https://img.shields.io/badge/Investable-2ea44f?style=flat-square) | The waiting is not free. Across the grid the confirmed arms spent **3,636 sessions** holding cash while the raw signal was already positive, worth **-216,670 bps** of forgone return, against **+253,703 bps** avoided by exiting late. Some confirmation length beat the unconfirmed rule on Sharpe in **92%** of cells, by **+0.110** on average — and the winning length is different in almost every cell (4 different values of *k* across 12 cells), which is what choosing it in hindsight looks like. |

> **In one sentence:** Confirmation does exactly what it says — five days of agreement cuts whipsaw trades from 78% to 30% of all round trips — but it pays for that with 3,636 sessions of sitting out a signal that was already right, and the confirmation length that would have won is only knowable afterwards.

## What we tested

"Wait for confirmation" is the most universally repeated piece of trading discipline
there is: do not act on the first crossing, wait for the signal to hold. It is a genuine
trade-off — fewer false starts against later entries — and it is almost never measured, partly
because the obvious comparison is confounded (an unconfirmed rule acts sooner, so any
difference could be about timing rather than about confirmation). We remove that: **every arm
carries exactly one day of execution lag**, and the only thing that varies is how many
consecutive days the raw signal must agree before the position changes (k = 1, 2, 3, 5, 10,
21). Three signals with deliberately different tempos — the 200-day moving average, 12-1
momentum, and a 14-day RSI that crosses its midpoint dozens of times a year — on **SPY, IWM,
TLT and GLD**, long the asset while confirmed and in T-bills otherwise, 2 bps a switch.

The two sides of the trade-off are measured **separately** rather than netted: the share of
round trips closed inside a month (what confirmation is supposed to prevent) and the sessions
spent in cash while the raw signal was already right, priced in basis points (what it costs).
Then the question nobody asks: choosing *k* on the first half of each tape, does it help on the
second?
**Dedup:** distinct from **110-faber-timing** and **594-leverage-rotation-200sma** (the moving
average as a strategy, unconfirmed), **499-trendline-break** and **437-donchian-breakout**
(different signals, no confirmation study), **836-timing-luck** (*when* in the month you
rebalance), **940-turnover-budget** (how often you re-rank, not how long you wait) and
**401-signal-stacking** (combining signals rather than delaying one).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a whipsaw actually is, the two halves of the trade-off drawn separately, and the awkward question of which waiting period you would have chosen |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the confirmation operator and its lag-neutral construction, whipsaw and delay accounting, the full tape × signal × k grid, a first-half/second-half choice test, and trending versus choppy synthetic controls |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`confirm_delay/`](confirm_delay/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
