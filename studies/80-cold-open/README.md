# Study 80 — Cold-Open

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | Contrast (up-Jan vs dn-Jan Feb-Dec) is +10.2 %, HAC *t* = **+3.94** — but collapses post-1985 (*t* = 1.99) and fails the base-rate test (p = 0.957); tiny n = 75. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Barometer strategy underperforms buy-and-hold (6.47 % vs 6.80 %/yr), misses 18 positive years out of 30 down-Januaries, and the post-1985 signal is too weak to justify tactical allocation. |
| **Beats a coin?** | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | Beats a fair coin (p = 0.001) but **not the base rate** (p = 0.957) — the market is up February-December 76 % of the time regardless of January; the barometer adds no information over that. |

> **In one sentence:** the January Barometer beats a coin (68 % hit-rate) but not the base rate of a drifting market (76 % of Feb-Dec periods are positive regardless), its 10-point return contrast is concentrated pre-1985 and fades post-publication, and going flat after a down January underperforms buy-and-hold — a calendar oracle that is mostly a mirror of the equity premium, not a crystal ball.

## What we tested

One of the most-quoted market aphorisms: *"As goes January, so goes the year"* (Hirsch 1972, Stock Trader's Almanac). Using ^GSPC daily prices since 1950 we compute annual January log-returns and February-through-December (rest-of-year) log-returns for each of 75 years, then ask: does the **sign of January** predict the rest-of-year sign and magnitude, more than could be explained by simply knowing that equity markets trend upward most years? We run a binomial test vs the coin and vs the base rate, a HAC-t contrast between up-Jan and down-Jan cohorts, a permutation control that shuffles January labels, a sub-period stability check, and a barometer-strategy simulation vs buy-and-hold.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the oracle in plain language, the base-rate trap, the hit-rate trick, why the strategy underperforms |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stat on the contrast, binomial tests vs coin and base rate, permutation control, sub-period decay, barometer vs B&H |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`cold_open/`](cold_open/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
