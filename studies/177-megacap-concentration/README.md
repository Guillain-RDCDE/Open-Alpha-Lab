# Study 177 -- Megacap-Concentration

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | Full 2008-2025: top-10 lags equal-weight by **-1.6 ppt/year** (HAC t = -0.31). Pre-2015: **-14 ppt/year** (t = -10.3). Post-2020: **+11.3 ppt/year** (t = +1.30). A regime phenomenon, not a general rule. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Survivorship-biased universe (all returns inflated); regime-unstable; no robust full-sample edge; a 2008-launch would have earned ~2.4%/year through 2014. |
| **Recency bias?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | The Mag-7 era (2020-2025) is real but only 6 years -- against 14 years (2008-2019) of flat-to-negative excess. The "always works" claim is falsified. |

> **In one sentence:** concentrating in the top-10 S&P 500 names by market cap looks like
> genius because of the Magnificent-Seven era -- but over the full 2008-2025 history the
> top-10 *lagged* the equal-weight index by 1.6 ppt/year, and pre-2015 it lost by 14 ppt;
> recency bias confirmed, mirage tradability.

## What we tested

A popular modern claim: each year, hold the 7 (or 10) largest S&P 500 companies by market
cap equally weighted. Proponents cite widening moats, scale advantages, and the Magnificent
Seven (Apple, Microsoft, Nvidia, Google, Amazon, Meta, Tesla) as proof. We test it across
the full EDGAR+yfinance history (2008-2025, 412 survivorship-biased names) using **prior
year-end** market cap ranks (no look-ahead), vs an equal-weight baseline and SPY total
return. We split by decade to expose the recency bias: the claim only works post-2020.
Universe is survivorship-biased (current S&P 500 members only); all returns are upper bounds.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the decade split, cumulative wealth, why 2020-2025 is not the whole story |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stats, decade splits, survivorship bias anatomy, synthetic positive control, multiple-comparisons check |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`megacap_concentration/`](megacap_concentration/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
