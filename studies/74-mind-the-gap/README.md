# Study 74 -- Mind-the-Gap

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Medium gaps fill **64.7%** of sessions (CI [63.4%, 65.9%]), robustly above 50% -- the pattern is real. But the *tradable* symmetric fade earns only +0.46 bps/trade with HAC *t* = **+0.30**: the fill frequency does not translate to a statistically certifiable return. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Near-zero gross edge means no positive break-even cost exists. At 1 trade/day any realistic round-trip (>= 0.5 bps) turns it into a loser; at 2 bps it is a statistically significant loser (*t* = -2.31, approx -8.8%/yr). |
| **Gaps always fill?** | ![Busted](https://img.shields.io/badge/Gaps_always_fill%3F-Busted-8b949e?style=flat-square) | Large gaps (> 1%) fill only **37.6%** of the time -- less than a coin. The "always" is demolished; the medium-gap tendency is real but not actionable. |

> **In one sentence:** opening gaps for medium-sized moves do fill 65% of the time -- a real pattern -- but the symmetric fade earns only +0.46 bps/trade (HAC *t* = +0.30, noise), large gaps reverse the claim entirely (38% fill), and any trading cost turns the near-zero gross edge into a confirmed loser.

## What we tested

A perennial claim in retail trading forums: *"An opening gap always fills -- when a stock gaps at the open, fade it and target the prior close."* We take it literally and measure two things separately: first, the **mechanical fill rate** (does the day's high/low bracket the prior close?) by gap-size bucket, to expose how much is noise vs real structure; second, the **tradable edge** -- a symmetric 1:1 ATR barrier backtest entering at the open in the fade direction, pinned against a random-direction control. Five tickers (SPY, AAPL, MSFT, TSLA, NVDA), 10 years of daily bars (2,514 sessions per ticker), auto-adjusted. A deterministic synthetic tape with a tunable gap-fill probability serves as the positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the fill-rate decomposition in plain language, the honest fair bet vs a coin, why costs bury it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-bucket Wilson CIs, per-instrument HAC *t*, bootstrap Sharpe CI, cost sweep, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`mind_the_gap/`](mind_the_gap/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
