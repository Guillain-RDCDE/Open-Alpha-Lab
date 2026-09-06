# Study 995 — Whose Sharpe Is It? 🌐

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — how much does an investor's home currency change a fund's measured Sharpe ratio? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Over 19 years, SPY delivered a Sharpe of **0.46** to a dollar-based investor and anywhere from **0.35** (CHF) to **0.50** (CAD) to investors based elsewhere — a spread of **0.15**, on the identical shares. Three separate channels do the work and they behave differently. The **variance** channel is mechanical: adding a currency leg raised volatility from 19.7% to a median 19.4% because var(a−c) = var(a) + var(c) − 2cov, and the median correlation between SPY and these currencies was only +0.27. The **drift** channel is luck: the dollar happened to move. The **rate** channel is the one nobody adjusts for — each investor's cash leg is their own, and using the US bill rate for everyone (the standard shortcut) biases every high-rate country's Sharpe downward. Across 5 assets the currency moved at least one pair's ranking by **1 places**. |
| **Tradability** — does hedging the currency improve it, and for whom? | ![Partial](https://img.shields.io/badge/Partial-dab617?style=flat-square) | Hedging is priced here rather than assumed, because a rolling forward hedge earns the **interest-rate differential** and not zero — selling dollars forward while US rates exceed yours costs you the gap, which is the charge hedged share classes pass on quietly. After that charge, a full hedge raised the Sharpe for **50%** of the currencies here, by a median -0.00. The variance-minimising ratio is not 1.0 either: its median across currencies is **1.57**, because SPY itself co-moves with risk-off currencies, so a full hedge is close to right. The practical reading is that the hedge buys volatility reduction reliably and return unreliably — which is the right way round for a long-horizon holder and the wrong way round for anyone hoping the hedge pays for itself. |

> **In one sentence:** The same SPY shares delivered Sharpe ratios from 0.35 to 0.50 depending on the holder's home currency — a spread of 0.15, wide enough to reorder how funds rank against each other.

## What we tested

"The S&P returned 10% a year with a Sharpe of 0.55." Whose 10%? A Japanese
investor who bought the identical shares earned a very different number from a Swiss one. This
study measures how different, from six home currencies, and separates the three channels that
are almost always quoted as one.

**Variance** is mechanical and permanent: `var(a − c) = var(a) + var(c) − 2cov(a, c)`, so unless
the currency moves *with* the asset, holding foreign assets adds risk — full stop. **Drift** is
luck: the dollar happened to go somewhere over this particular window, and that is close to
unforecastable. **The risk-free rate** is the channel nobody adjusts for: a euro investor's cash
leg is the euro deposit rate, and during 2015-2022 those differed from dollar rates by more than
two points, so a Sharpe computed with the US bill rate — the standard shortcut — is biased for
every non-US investor. The study derives each investor's own rate from covered interest parity
rather than ignoring the problem.

Two things make it practical rather than arithmetic. The **ranking test** asks whether a currency
merely shifts every Sharpe by the same amount (in which case allocators can ignore it) or
actually reorders funds against each other. And the **hedge is priced properly**: a rolling
forward hedge earns the interest-rate differential rather than nothing, which is exactly the
charge hedged share classes pass on quietly — and the variance-minimising hedge ratio turns out
not to be 100%, because US equities co-move with risk-off currencies (Campbell,
Serfaty-de Medeiros & Viceira 2010).
**Dedup:** distinct from **370-currency-hedging-costs** (the cost of the hedge in isolation),
**481-international-diversification** (whether to hold foreign assets at all),
**744-dollar-strength-and-returns** (the dollar as a return predictor) and
**970-annualisation-factors** (converting statistics across horizons rather than across
currencies).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the same fund actually delivered to investors in six different countries, and which part of the difference was skill, luck, or arithmetic |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | exact currency conversion, the variance identity checked against realised data, a three-way Sharpe-gap decomposition, implied foreign risk-free rates, ranking stability, and hedge ratios priced with the carry charged |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`whosesharpe/`](whosesharpe/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
