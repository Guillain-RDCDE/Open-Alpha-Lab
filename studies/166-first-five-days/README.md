# Study 166 — First-Five-Days

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Hit-rate **69.7%** beats a coin (p = 0.0004) but **not the base rate of 73.7%** (p = 0.82); contrast HAC *t* = 4.87 full-sample is inflated by mechanical overlap (F5D is inside the full year) and regime correlation; post-1985 directional hit-rate 60.0% is not significant vs a coin (p = 0.13); n = 76 is tiny. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Strategy (long if up-F5D, flat otherwise) earns 8.01%/yr vs buy-and-hold's 7.91%/yr — a margin within noise at n = 76; the post-1985 directional call is too fragile to justify sitting out a full year of equity exposure. |
| **Beats the base rate?** | ![No](https://img.shields.io/badge/No-c0392b?style=flat-square) | The market ends positive in 73.7% of all years regardless of how the first five days go; the almanac's "early warning" does not improve on simply assuming equity markets drift upward. |

> **In one sentence:** the First-Five-Days Early Warning System beats a coin (69.7% hit-rate) but not the correct null — the market is already up 74% of years without any almanac — its large t-stat is partly mechanical (those five days are inside the year being predicted), its directional signal has collapsed post-1985, and the trading strategy barely edges buy-and-hold within noise.

## What we tested

The Stock Trader's Almanac "First-Five-Days Early Warning System" (Yale Hirsch, 1972 onward): if the S&P 500 closes higher at the end of the first five trading days of January than it started the year, it is said to predict a positive full year; a down opening week warns of a rough year. Using ^GSPC daily prices since 1950 we compute annual first-five-days log-returns and full-year log-returns for each of 76 years, then ask: does the **sign of the F5D return** predict the full-year sign and magnitude, more than could be explained by simply knowing that equity markets trend upward 74% of years? We run a binomial test vs the coin and vs the base rate, a HAC t-stat on the contrast, a permutation control, a sub-period stability check, and a strategy simulation vs buy-and-hold. A key confound — the F5D return is a component of the full-year return — is noted and explained.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the almanac in plain language, the base-rate trap, the return gap and why part is mechanical, the sub-period decay |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stat, binomial tests vs coin and base rate, mechanical-overlap discussion, permutation control, sub-period breakdown, strategy vs B&H |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`first_five_days/`](first_five_days/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
