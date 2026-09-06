# Study 964 — All-Time High 🏔

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — are forward returns from a record high different from anywhere else? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Money invested on a record-high close earned **+12.4%** over the next twelve months against **+10.2%** from every other day — a gap of **+2.3%** (HAC *t* = +0.17 at the horizon lag; the non-overlapping cross-check gives -0.6%). Record highs are also not rare: **11%** of SPY's sessions closed at one. The direction is the finding — it is not negative — and every tape here is a survivor, which flatters it. |
| **Tradability** — does waiting for a dip beat owning the thing? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Waiting for a **5%** dip and sitting in bills otherwise was invested only 41% of the time and gave up **-5.08%/yr** of compounding on SPY (excess-Sharpe gap -0.22, *t* = -2.54); it won on excess Sharpe on **0 of 6** tapes. The drawdown it buys back is real — the return it gives up is larger. |

> **In one sentence:** Buying at a record high is not buying the top: the next twelve months paid **+12.4%** on average against **+10.2%** from every other day, and the rule the folklore implies — wait for a dip, hold bills meanwhile — cost **-5.08%/yr** on SPY while being out of the market 59% of the time.

## What we tested

"Never buy at an all-time high" is the most durable piece of investing folk wisdom
there is — it feels like risk management, it sounds like discipline, and it is repeated every
time an index prints a record. Stated at full strength: *a record high is where the buyers
are exhausted and the next move is down, so money put in there does worse than money put in
after a pullback.* We mark every session that closed at the running maximum of the
**total-return** index — a wealth peak, not a price peak — on **SPY, QQQ, EFA, EEM, TLT and
GLD**, and compare forward returns at 1, 3 and 12 months against every other session, and
against buying at 2%, 5%, 10% and 20% below the peak. Then we price the advice as the
portfolio it implies: hold the asset only while it is below its peak, T-bills otherwise, one
day of execution lag, costs on every switch.

Overlapping forward returns are the trap in this design — a 252-day window shares 251 days
with the next — so every *t* uses a HAC lag equal to the horizon, and a **non-overlapping**
cross-check (one observation per year, the rest discarded) is run alongside. The unfixable
caveat is stated rather than hidden: these six tapes are survivors that spent the sample in
secular uptrends, which flatters any record-high result.
**Dedup:** distinct from **236-fifty-two-week-high** and **869-breakout-52wk-high**
(cross-sectional stock sorts on 52-week highs), **202-fifty-two-week-low** and
**331-fifty-two-week-range** (the other end of the same range), **241-buy-the-dip** (buying
*after a fall*, no record-high condition), **110-faber-timing** / **594-leverage-rotation-200sma**
(moving-average timing) and **934-lump-sum-vs-dca** (*when* to deploy a lump sum, unconditional
on the market's state).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the advice feels right, what actually happened to money invested at records, the drawdown you buy and the compounding you pay for it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | conditional forward-return tables with horizon-lag HAC and a non-overlapping cross-check, drawdown-bucket comparisons, the dip rule as a portfolio with lag and costs, the patience sweep, survivorship caveats and the synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`ath_buy/`](ath_buy/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
