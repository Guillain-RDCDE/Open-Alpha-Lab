# Study 300 -- Sports-Sentiment

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Edmans, Garcia & Norli (2007) is a real, famous behavioural result -- but on the *tradable* single-country ETF tape the mean next-day return after a loss is **+33 bps** with Newey-West HAC t = **+1.6** (wrong sign, not significant). Literature support without |HAC t| >= 2 on the real tape is Weak, not Real. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Shorting the ETF the session after each loss nets **-43 bps/event**, is idle ~99% of the time, and pays wide single-country-ETF spreads and borrow. |

> **In one sentence:** the Edmans "loss effect" is genuine in local national indices, but it does not survive the translation to the version you could actually trade -- USD single-country ETFs over the next US session -- where the sign is wrong and the t-stat is insignificant.

## What we tested

The Edmans, Garcia & Norli (2007, *Journal of Finance*) "Sports Sentiment" result: after a
national soccer team is **eliminated** from a major tournament, its home stock market earns
a significantly **negative** next-day return (~ -49 bps; effect is asymmetric -- wins do not
reverse it). We hardcode 60 marquee World Cup / Euro / Copa elimination **losses** (1998-2024)
in `data.py`, pair each with the country's investable single-country ETF (EWU, EWG, EWQ, EWZ...;
^GSPC for the US), and measure the close-to-close return of the **next** trading session (a
one-day execution lag). We report the mean, a **Newey-West HAC** t-stat, an i.i.d. bootstrap,
sub-group cuts (penalty shootouts; World Cup vs Euro/Copa), and the short-on-loss PnL gross and
net of costs. A synthetic positive control plants a -49 bps effect and recovers it -- so the null
on the real tape is about the **tradable proxy and sample size**, not a broken test.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the wrong-signed mean, the distribution, the failed trade in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stat, bootstrap, sub-group cuts, gross/net costs, the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`sports_sentiment/`](sports_sentiment/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
