# Study 254 -- WSB-Mentions

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Loud-minus-quiet spread HAC *t* = **-0.14**, hit-rate **47%**, pooled rank-IC = **-0.035** (p = 0.45) over 468 (stock, month) pairs; bootstrap Sharpe CI [-1.11, +1.19] straddles zero; no sub-period clears \|t\|=2. Mentions are *contemporaneous* with the move (endogenous), and delisted blow-ups (WISH/BBBY) drop from the short leg -- so even this zero is an upper bound. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Net spread @30 bps one-way = **-4.0%/yr** (*t* = -0.15); the loud leg behaves like a random half of the basket; illiquid meme names and heavy short borrow make a live book hopeless. No edge to trade. |
| **Endogeneity** | ![Lagging-proxy](https://img.shields.io/badge/Lagging--proxy-8b949e?style=flat-square) | WSB buzz spikes *because* the price already ran -- a contemporaneous attention shock used as a forward predictor is mechanically confounded (Long et al. 2021; Da-Engelberg-Gao 2011). |

> **In one sentence:** counting r/WallStreetBets mentions and buying the loudest meme names forecasts next month's return no better than a coin flip (*t* = -0.14, rank-IC = -0.04) -- the whole basket lost ~23%/yr regardless of buzz, and the proxy is a *lagging* echo of the move it supposedly predicts.

## The claim

> *Does the r/WallStreetBets mention count call the meme move?*

## What we tested

We hardcode a curated month x ticker WSB mention-count table for a 14-name meme
basket (GME, AMC, BB, BBBY, KOSS, PLTR, ... -- including names that went to zero),
January 2021 through December 2023, and join it to real Yahoo monthly returns.
Each month-end we rank the basket by that month's buzz, long the loud half and
short the quiet half, hold one month (one-month execution lag) and rebalance. We
pin the loud-minus-quiet spread against (a) the equal-weight basket, (b) a
random-portfolio control of identical loud-leg size, (c) a pooled Spearman
rank-IC, and (d) calendar-year sub-periods, then sweep turnover costs (30 bps
one-way to reflect illiquidity and short borrow) and test the "buy the rumor,
sell the news" reversal sign. A deterministic synthetic positive control
confirms the engine recovers a planted mention->return link (either direction)
when one exists.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the buzz timeline, the loud-vs-quiet race in plain language, why the whole basket bled, and why mentions lag the move |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | synthetic lead/reversal/null controls, HAC *t*-stats, bootstrap Sharpe CI, pooled rank-IC, sub-periods, turnover cost sweep, endogeneity & delisting caveats |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`wsb_mentions/`](wsb_mentions/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
