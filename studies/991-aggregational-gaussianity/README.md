# Study 991 — The Slow Bell 🔔

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do returns converge to normality as the horizon lengthens? | ![Confirmed](https://img.shields.io/badge/Confirmed-2ea44f?style=flat-square) | For SPY over 7,764 sessions, excess kurtosis falls from **10.6 at one day to 6.73 at 252 days** — so the stylised fact is real. But it is not the central limit theorem's rate. For independent draws the decay is exactly ``k₁/n``, which would put the 252-day kurtosis at **0.042**; the tape delivers 160× that. Fitting `kurtosis ~ horizon^(−b)` gives **b = -0.04** (se 0.12) against the i.i.d. value of 1.00, *t* = **-8.77**. The Hill tail index is **2.99** — above 2, so the variance exists and the theorem does apply, but below 4, meaning the kurtosis itself may not exist and every kurtosis number above is an unstable statistic rather than an estimate. |
| **Tradability** — at what horizon can you finally use normal-based risk arithmetic? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The practical question is when normal arithmetic becomes safe. Excess kurtosis first drops below 0.5 at **None days** against the 63 days independence would have predicted — a slowdown of **nan×**. At the longest horizon measured, 3-sigma moves still arrive **nan× more often** than a normal says. And a warning about the tests: at 252 days there are only 30 non-overlapping observations, where Jarque-Bera has about **33% power** against a *t*(4). A passing normality test at long horizons is mostly evidence that the sample is small. |

> **In one sentence:** Returns do become more normal with horizon, but at exponent -0.04 rather than the 1.00 independence would give — so the bell arrives roughly nan× later than the textbook implies, and the tests that would tell you have run out of data by then.

## What we tested

Daily returns are famously fat-tailed. The central limit theorem promises that
sums of many draws tend to a bell curve, so monthly returns — sums of 21 days — should be
tamer, and annual returns nearly normal. Every long-horizon planning model that uses normal
arithmetic rests on that promise. This study measures how fast the convergence actually happens.

The measurement has an unusual advantage: for **independent** draws with one-day excess kurtosis
*k₁*, the sum of *n* draws has excess kurtosis **exactly *k₁*/n**. Not asymptotically — as an
identity. So the tape can be compared against a benchmark with no estimation error in it, and
the gap is precisely the cost of the independence assumption being false. Fitting
`kurtosis ~ horizon^(−b)` turns that gap into one number, against the i.i.d. value of exactly 1.

Two further things are checked that the stylised-fact literature usually asserts. The **Hill
tail index** decides whether the theorem applies at all: below 2 the variance is infinite and
sums converge to a stable law that never becomes normal; below 4 the *kurtosis* is infinite and
every kurtosis figure in the study is a descriptive statistic rather than an estimate. And the
**power** of the normality tests is simulated at the sample sizes each horizon actually provides
— at a 252-day horizon there are about 32 non-overlapping observations, where Jarque-Bera can
barely distinguish a *t*(4) from a normal, so "annual returns pass a normality test" is close to
uninformative. The synthetic control turns fat tails and volatility clustering on independently,
showing that clustering, not tail fatness, is what slows the bell down.
**Dedup:** distinct from **311-fat-tails** and **427-return-distributions** (the unconditional
one-day distribution), **256-volatility-clustering** (the clustering itself rather than its
effect on aggregation), **990-var-breach-count** (tail risk in a forecasting frame) and
**970-annualisation-factors** (converting statistics between horizons rather than asking how the
distribution's shape changes).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | how fat-tailed returns really are, how quickly they calm down as you zoom out, and why 'annual returns are normal' is harder to check than it sounds |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the exact k1/n benchmark, a fitted decay exponent against the i.i.d. value of 1, Hill tail indices, simulated test power at realistic sample sizes, the overlapping-window inflation, and a two-knob synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`slowbell/`](slowbell/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
