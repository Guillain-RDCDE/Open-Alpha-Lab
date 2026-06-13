# Study 104 -- Bollinger-Reversion

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Lower-band entry earns **+193 bps/trade** (t = +5.34) gross but the random-day control earns +141 bps (t = +6.20) on the same tape; the incremental delta vs random is only ~+52 bps with t ~ +0.63 -- not significant. The dominant signal is bull-market drift, not band-specific reversion. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Costs are not the issue (low turnover, ~6 trades/year/stock). The claim survives gross but barely beats a random buy, so any bear market or sideways regime would destroy the premium the bands are merely proxying. |
| **"Price always returns"?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The upper-band breakout ALSO earns +118 bps (t = +5.49) in the same period. When both the reversion rule and its exact opposite produce positive returns, the bands are not the signal -- the market trend is. |

> **In one sentence:** Bollinger Band lower-band entries beat a random buy by only ~52 bps per trade (t = +0.63) over 21 years -- the impressive gross edge (+193 bps) is mostly the bull-market drift that a random-day buyer collects just as easily, and the upper-band breakout works equally well, proving the bands themselves carry no magic.

## What we tested

A staple of retail trading books and forums: *"When the daily close pierces the lower Bollinger Band (20-SMA minus 2 sigma), buy -- price always returns to the middle band."*  We take this literally: lower-band entries on six liquid US equity tapes (SPY, QQQ, AAPL, MSFT, JPM, XLE) back to 2005, entered at the next day's open, exited after 20 days (horizon-fair) or at the 20-day SMA (the recipe's own exit), pinned against a **random-day control** on identical instruments and period.  We also test the *opposite* folk rule (upper-band pierce = breakout momentum buy) to expose the contradiction: if both work, neither is the signal.  A deterministic synthetic tape with tunable mean-reversion serves as the positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the band-touch recipe in plain language, why "price always returns" is partly true but mostly drift, the breakout contradiction, why win-rate is not edge |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-instrument HAC t-stats, delta vs random with its own t-stat, the breakout contradiction numerically, cost sweep, the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`bollinger_reversion/`](bollinger_reversion/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
