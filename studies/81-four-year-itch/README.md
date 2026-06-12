# Study 81 — Four-Year-Itch

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Year-3 leads at +14.0%/yr mean, but vs the rest Welch *t* = **+1.92** — just below the bar, with only **24** year-3 observations since 1929. SPY (8 cycles) *t* = +1.48; post-2001 (5 cycles) *t* = +0.98. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A 25%-invested year-3-only rule yields the same mean with more cash drag than buy-and-hold; worst year-3 is **−47%** (1931); no risk-adjusted edge once opportunity cost is charged. |
| **Beats a coin?** | ![NOT SUPPORTED](https://img.shields.io/badge/NOT_SUPPORTED-8b949e?style=flat-square) | The pattern is confined to pre-war data and does not clear standard significance in any modern subsample; Bonferroni-adjusted bar for 4 tests (|*t*| ≈ 2.4) is not met either. |

> **In one sentence:** year-3 of the Presidential cycle does carry the highest historical average return on the S&P 500 (+14.0% vs +5.8% for the rest, Welch *t* = +1.92), but the structural ceiling of only ~24 complete cycles prevents confirmation, and in modern data the signal evaporates entirely.

## What we tested

The *Stock Trader's Almanac* claim, in circulation since 1972: the pre-election year (year 3 of a U.S. Presidential term) is the best for equities because incumbents stimulate the economy before the vote. We test it on **^GSPC** daily returns 1928–2026 (~24 complete cycles, the maximum available), compounding daily returns to annual, assigning Presidential-cycle year labels (a pure calendar function, no look-ahead), and running a Welch t-test of year-3 vs the pooled years 1+2+4. We include subsample checks (post-1975, post-2001), an SPY cross-check, a permutation test, and a Bonferroni correction for the 4 year-of-term tests.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim in plain language, the bar-chart trap, the fair comparison, the modern-data check, why 24 data points is the ceiling |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-term-year HAC *t*-stats, jitter plot of all 24 annual returns, subsample decay, Bonferroni correction, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`four_year_itch/`](four_year_itch/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
