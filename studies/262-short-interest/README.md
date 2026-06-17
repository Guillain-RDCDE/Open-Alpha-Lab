# Study 262 -- Short-Interest

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | High-SI minus low-SI 3m spread = **-5.95%**, cross-sectional *t* = **-0.74**; OLS slope *t* = -0.61, R^2 = 0.006; label-shuffle p = 0.47; bootstrap 95% CI [-21.4%, +9.6%] straddles zero; sign flips across horizons. A single 60-name snapshot cannot deliver a robust HAC *t* >= 2 -- and the literature splits into two opposing camps (informed shorts *bleed* vs crowded shorts *squeeze*), which is the opposite of a clean signal. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The only directional read (short the crowded names) lands on the hard-to-borrow, high-fee, squeeze-prone leg: ~8%/yr+ borrow plus wide spreads turn a faint, insignificant gross spread into a negative net one (-3.85% over 3m). GameStop-style squeeze risk is concentrated exactly here. |
| **Survivorship bias** | ![Named](https://img.shields.io/badge/Survivorship--biased-8b949e?style=flat-square) | Universe = a fixed surviving basket; the short-interest reading is a single static snapshot (revised bi-monthly, stale by settlement lag). All results are upper bounds. |

> **In one sentence:** sorting a 60-name basket by short interest, the heavily shorted stocks leaned *slightly* toward bleeding (a faint nod to the "informed shorts" camp), but the spread is statistically a coin flip (*t* = -0.74), the sign flips across horizons, and the only tradable interpretation forces you to short the most expensive, squeeze-prone names -- a textbook **None / Mirage**.

## The claim

> *Do heavily shorted stocks bounce or bleed?*

## What we tested

A single cross-sectional sort: we hardcode a curated short-interest snapshot
(short % of float for ~60 well-known US names, as-of a recent settlement date),
join it to forward returns, and form a high-SI quantile (the most heavily
shorted) versus a low-SI quantile (the least shorted). We report the
high-minus-low spread with a cross-sectional t-stat, an OLS slope of forward
return on standardized short interest, a label-shuffle permutation null, a
bootstrap CI, a 1m/3m/6m/12m horizon sweep, and a cost-adjusted net spread that
charges borrow on the short leg. A deterministic synthetic positive control
confirms the sort recovers a planted SI->return slope (with the correct sign)
and reads ~zero on the null.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the bounce-vs-bleed debate, who is heavily shorted, the sort in plain language, and why one snapshot is a coin flip |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | quantile spread + OLS slope, label-shuffle null, bootstrap CI, horizon sweep, borrow-aware costs, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`short_interest/`](short_interest/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
