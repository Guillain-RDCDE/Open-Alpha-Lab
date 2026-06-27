# Study 522 — Percent Operating Accruals

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | On a 40-name large-cap survivor basket (2009–2025): long-short hedge +2.6%/yr, HAC *t* = **+0.84** — below the \|t\|≥2 bar. The quintile sort is non-monotone (Q2 leads, not Q1), the long leg's excess is +0.6%/yr (*t* = +0.29), and a label-shuffle placebo puts the real *t* at *p* ≈ 0.47 (seed-stable) — indistinguishable from random. And this is a *survivorship-biased upper bound*. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Annual rebalancing on large caps is operationally cheap (62%/yr turnover, net hedge +1.9%/yr) — but there is no edge to harvest: net HAC *t* = +0.64. A tradable wrapper around a non-signal. |
| **Does percent-scaling beat Sloan (Study 231)?** | ![Busted](https://img.shields.io/badge/HLVW_sharper_sort-Busted-8b949e?style=flat-square) | HLVW (2011) claim accruals scaled by *earnings* sort returns *more* sharply than Sloan's asset-scaling. On a comparable survivor panel: Study 231 reached HAC *t* = +2.73; this percent-scaled version lands *t* = +0.84. The sharper-sort claim does not replicate on large-cap survivors. |

> **In one sentence:** scaling operating accruals by the magnitude of earnings (Hafzalla-Lundholm-Van Winkle 2011) instead of by total assets (Sloan) does *not* sharpen the long-short sort on a large-cap survivor basket — the hedge is +2.6%/yr gross at HAC *t* = +0.84, indistinguishable from shuffled labels (placebo *p* ≈ 0.47), and weaker than the asset-scaled Sloan version on the same kind of panel.

## What we tested

Hafzalla, Lundholm & Van Winkle (2011) take Sloan's operating accrual — Net Income minus Operating Cash Flow — and scale it by the **absolute value of earnings** rather than by total assets. Their "percent accruals" rank firms by the *fraction* of reported profit that is *not* backed by cash; they argue this sorts future returns more sharply than the asset-scaled Sloan (1996) accrual, especially in the extreme deciles.

We pull annual 10-K Net Income and Operating Cash Flow from **EDGAR** for a fixed 40-name large-cap survivor basket, compute percent operating accruals = (NI − CFO) / |NI|, sort into quintiles each fiscal year, lag one full year (fiscal year y → calendar year y+1 returns), and go long the low-percent-accruals quintile / short the high (Q1 − Q5). We charge transaction costs and short-leg borrow, run a label-shuffle placebo and a random-portfolio null, and verify the engine on a deterministic synthetic panel with a tunable planted premium.

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the percent-accruals recipe in plain English, why earnings-scaling differs from asset-scaling, the survivorship and post-publication-decay caveats, year-by-year results |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | quintile monotonicity, HAC t-stats, label-shuffle placebo distribution, random-portfolio null, costs, the Sloan head-to-head, synthetic positive-control sweep |

## Does scaling accruals by earnings really beat scaling by assets?

On S&P-500-scale survivors from 2009 to 2025: no. The percent-accruals hedge earns +2.6%/yr gross (HAC *t* = +0.84), the sort is non-monotone, and the signal is statistically indistinguishable from randomly shuffled labels (placebo *p* ≈ 0.47). The asset-scaled Sloan version on a comparable panel (Study 231) reached *t* = +2.73 — so on large-cap survivors the earnings-scaling does the opposite of HLVW's claim. Both accrual flavours have decayed on liquid large caps post-2000 and survive mainly in small, illiquid names this basket excludes.

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`percent_operating_accruals/`](percent_operating_accruals/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
