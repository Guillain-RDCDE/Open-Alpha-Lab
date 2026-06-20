# Study 329 -- One-Month-Reversal

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | Full-sample loser-winner spread HAC *t* = **+2.40** (+6.6%/yr) and dollar-neutral (spread beta **0.31**, not disguised beta) -- **but** a one-month gap kills it (skip=1 *t* = **-0.21**) and it is dead since 2002 (2015-2026 *t* = **-0.10**). |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | ~**78%** one-way monthly turnover; **break-even ~17 bps**; net *t* < 1 by 10 bps and negative by 20 bps. A signal that evaporates one month out cannot fund a 100%-refreshed book. |
| **Is it bid-ask bounce?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | Inserting one month between the formation close and the holding month removes the **entire** effect -- the Roll (1984) / Lo-MacKinlay (1990) microstructure illusion. |

> **In one sentence:** last month's losers really did beat last month's winners next month (*t* = +2.40 over 1990-2026) and it is genuinely market-neutral -- but the whole thing is bid-ask bounce dressed as alpha: leave a one-month gap and it vanishes, it has been dead since 2002, and at 78% turnover it is a mirage to trade.

## What we tested

Jegadeesh (1990, *Journal of Finance*) documented that monthly stock returns reverse: rank
the cross-section by **last month's** return, buy the losers and short the winners, hold one
month, and you collect ~2%/month of negative serial correlation -- the most-cited
short-horizon anomaly in equities. We run the exact monthly-rebalanced quintile spread on
the current S&P 500 (~398 names, 1990-2026), then pin it against the four things that decide
whether it is information or microstructure: a one-month-gap variant (defusing the bid-ask
bounce), a beta decomposition, a sub-period decay split, and a realistic turnover/cost
sweep. The offline core proves the engine on a synthetic panel with a tunable reversal knob;
the verdict is measured on the market.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "buy last month's losers" looks like a 90s goldmine, and how the bid-ask bounce and 78% turnover quietly empty it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC *t*, the skip=1 microstructure test, beta-neutrality, sub-period decay, the cost wall & break-even, the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`one_month_reversal/`](one_month_reversal/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
