# Study 988 — The Taming 📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is Bitcoin's volatility on a genuine downward trend? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Bitcoin's trailing 30-day volatility averaged **60%** over 11.7 years, ranging 14% to 175%. Fitted four ways, the trend in log volatility is: OLS **-2.7%/yr** (naive *t* = -13.62), Theil-Sen -3.7%/yr, and OLS with a block-bootstrap standard error -0.027 with *t* = **-1.60** — the naive *t* is 8.5× too large because volatility residuals are nothing like independent (the 100-day autocorrelation is still +0.15). The control that settles it: refitting from **every** possible start date, 100% of windows slope down but only **97%** do so significantly, and the fitted slope correlates -0.22 with the volatility on the day the window opens. That last number is the whole trick: start at a peak, get a decline. |
| **Tradability** — would sizing a position off that trend have worked? | ![Useful](https://img.shields.io/badge/Useful-2ea44f?style=flat-square) | Sizing to a constant 40% volatility rather than holding a fixed position returned **+inf%/yr** against buy-and-hold's **+53.7%** (Sharpe 7.41 vs 0.98, drawdown -inf% vs -83%) at average leverage 0.81×. Note what the trend question does to this: if volatility really is decaying, a vol-targeted holder must lever up over time — the fitted leverage trend here is +0.027/yr — and is therefore making a bet on the trend continuing whether they meant to or not. |

> **In one sentence:** Bitcoin's volatility slopes down at -2.7% a year with a naive *t* of -13.62 and a bootstrapped *t* of -1.60, and only 97% of possible start dates reproduce it — the maturity story is mostly a choice of where to begin the chart.

## What we tested

"Bitcoin is maturing, its volatility is coming down" is said at the top of every
cycle and the bottom of every one, and the chart that supports it always slopes down. This study
asks whether the slope survives three objections that the chart never addresses.

**Persistence.** Realised volatility observed 100 days apart still autocorrelates around +0.3.
An OLS trend on such a series has a standard error several times larger than the naive formula
gives, so the trend is fitted four ways — OLS, Theil-Sen, Mann-Kendall, and OLS with a
**block-bootstrapped** standard error — and their disagreement is treated as the result.

**The start date.** This is the control nobody runs. In a persistent series, beginning the window
at a volatility peak *manufactures* a downward trend, and Bitcoin's history offers several peaks
to begin at. The trend is therefore refitted from **every** possible start date, and the study
reports what fraction yield a significant decline and how strongly the fitted slope correlates
with the volatility on the day the window opens.

**The yardstick.** "Maturing" should mean converging toward other assets, not merely falling — an
asset whose volatility halves while every asset's volatility halves has been carried by a calm
market. So section 7 works in ratios. Finally, because the practical stake is position sizing,
the volatility-targeting rule is priced: if the decay is real, a vol-targeter must lever up over
time, and is betting on the trend continuing whether they meant to or not.
**Dedup:** distinct from **142-bitcoin-correlation** and **604-crypto-equity-beta** (co-movement,
not the volatility level), **371-vix-term-structure** and **256-volatility-clustering** (implied
and clustering in equities), **983-bitcoin-leads-equities** (a lead-lag question) and
**774-levered-etf-decay** (the compounding cost of leverage, which appears here only inside the
sizing rule).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what Bitcoin's volatility has actually done, and the one chart trick that makes any persistent series look like it is calming down |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | calendar-correct annualisation, four trend estimators including a block bootstrap, the full start-date sensitivity surface, era and halving cuts, ratio comparisons against equities, and a vol-targeting backtest |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`taming/`](taming/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
