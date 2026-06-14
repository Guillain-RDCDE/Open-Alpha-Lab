# Study 155 — Asset-Turnover

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Top-quintile excess **+1.3%/yr**, HAC *t* = **+1.31** (16 annual obs, survivorship-biased upper bound); not distinguishable from zero — and true live *t* is lower still. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Annual rebalance, concentrated ~40-name portfolio, no robust signal; survivorship bias inflates every number. |
| **Adds over ROA?** | ![No](https://img.shields.io/badge/No-8b949e?style=flat-square) | ROA (NetIncome/Assets) also shows no robust top-quintile premium (*t* = −0.13). Incremental AT over ROA: +1.4%/yr (*t* = +1.76), not significant. |

> **In one sentence:** asset turnover (Revenues / Assets), the DuPont efficiency leg, shows no robust long-only premium on a survivorship-biased S&P 500 panel — the top-quintile excess of +1.3%/yr is noise (*t* = 1.31), ROA is equally absent, and the only significant finding is a long-short (*t* = 2.36) driven mainly by the bottom quintile's drag.

## What we tested

The DuPont identity decomposes return on equity into profit margin × asset turnover × leverage.
The efficiency leg — how many dollars of revenue a firm generates per dollar of assets — is
pitched as a quality signal: capital-efficient managers compound faster and attract multiple
expansion.  We sort the current S&P 500 universe on Revenues / Assets from fiscal year y, go
long the top quintile, and measure calendar-year y+1 returns vs the equal-weight universe.  A
one-year reporting lag ensures the 10-K is public before the position is opened.  The panel is
**survivorship-biased** (current S&P 500 members only, projected back) — all results are upper
bounds.  A head-to-head ROA comparison decides whether AT is an independent factor or a
re-labelling of profitability.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the DuPont story, the year-by-year picture, the head-to-head with ROA in plain language, why neither AT nor ROA produces a robust premium |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stats, random-portfolio control, AT vs ROA independence test, long-short decomposition, synthetic positive control, survivorship accounting |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`asset_turnover/`](asset_turnover/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
