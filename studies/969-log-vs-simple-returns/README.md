# Study 969 — Log or Simple 📐

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the convention change the numbers by enough to matter? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | It scales with the square of volatility, so it is invisible at one end of the tape and enormous at the other. On **BIL** (0% annualised vol) the mean simple and mean log returns differ by **0.00%/yr**; on **TQQQ** (61% vol) they differ by **18.9%/yr**, and the sigma-squared-over-two prediction accounts for 99% of it. The Sharpe ratio moves too: up to **0.31** between the two conventions. |
| **Tradability** — is there a rule that settles it? | ![Useful](https://img.shields.io/badge/Useful-2ea44f?style=flat-square) | Yes, and it is two lines. **Across time, use logs** (they add, and their mean exponentiates to the CAGR the path delivered). **Across assets, use simple returns** (a portfolio's return is a weighted average of them, exactly). The failure mode is weighting log returns: on an equal-weight book of 7 tapes it understated the CAGR by **6.11%/yr** — 1.80x of terminal wealth over 12 years — and it understated on **essentially every day**, because Jensen's inequality only points one way. |

> **In one sentence:** The two conventions differ by about half the variance, which is nothing on a bond fund (0.00%/yr) and **19%/yr** on bitcoin — so the rule is not 'pick one', it is 'logs across time, simple across assets', and the mistake worth hunting in a codebase is a portfolio built by weighting log returns.

## What we tested

Two definitions of "the return" are in daily use, both correct: the **simple**
return `P_t/P_{t-1} - 1`, which is what lands in the account, and the **log** return
`ln(P_t/P_{t-1})`, which adds up across time. Every codebase mixes them somewhere. This study
measures what the mixing costs, on eight tapes spanning **BIL** (0.3% annualised volatility) to
**BTC-USD** (over 60%), plus a 3× leveraged fund for the pathological case. Four things are
measured: the gap between the two means and how much of it the textbook σ²/2 approximation
actually explains (less than you would hope, once skew and kurtosis arrive); the terminal-wealth
cost of the commonest real bug — building a portfolio by weighting *log* returns; which
statistics move with the convention (the Sharpe ratio) and which do not (beta); and the four
mutually inconsistent ways practitioners annualise a mean return.

There is no *t*-statistic anywhere in this study and that is deliberate: none of this is an
empirical claim, it is arithmetic, and the only open question is **magnitude** — which is
exactly what decides whether a convention mismatch is pedantry or a bug worth hunting.
**Dedup:** distinct from **970-sqrt-time-scaling** (annualising a *volatility* under
autocorrelation), **942-inverse-etf-structural-loss** and **944-optimal-leverage-realized**
(volatility drag as a property of *leveraged products*, not of the return convention),
**102-free-rebalance** (whether rebalancing pays — here the bonus appears only as the yardstick
for the size of the error) and **156-martingale** / **157-kelly-sizing** (growth-optimal
betting, which uses log returns as a utility rather than as a bookkeeping convention).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the two definitions in one picture, why bitcoin's 'average return' can be positive while its investors lost money, and the one bug worth grepping for |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the σ²/2 expansion checked against the tapes including its skew and kurtosis residual, Jensen's inequality as a one-way error, terminal-wealth cost of the log-weighted portfolio, and Sharpe versus beta sensitivity |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`log_vs_simple/`](log_vs_simple/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
