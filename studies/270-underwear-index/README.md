# Study 270 — Underwear-Index

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**Table hook:** *Does men's-underwear sales (Greenspan) signal the economy?*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | As a *leading* indicator the dip is noise: recession-after-dip **25%** vs **13.8%** base, Fisher p = **0.50**, perm p = **0.49**, on only **4 dips**. The coincident p<0.001 is a reconstruction tautology, not a finding. Single index, price-only — no survivorship. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The only implementable rule (dip → cash next year) earns **7.4%/yr** net vs **8.6%/yr** buy-and-hold; HAC t on the spread = **−0.94**. It sits out post-recession rebounds. |
| **Busted?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The famous 2008 dip was *coincident* with the slump, not ahead of it; n = 33 with 4 dips cannot resolve any effect. |

> **In one sentence:** the Men's Underwear Index is a vivid *story about* recessions, not a *predictor of* them — once you forbid look-ahead, a dip's next-year recession lift is statistically indistinguishable from the 15% base rate, and trading it loses to buy-and-hold.

## What we tested

The Greenspan claim: a *dip* in men's-underwear sales flags an oncoming recession. We
hardcode a curated annual men's-underwear index (1992–2024) and the NBER recession
calendar in `data.py`, join them with Shiller S&P 500 annual returns, and separate the
**coincident** view (tautological in a reconstructed series, reported only as a control)
from the **leading** view (the genuine forecasting claim). We run a Fisher exact test on
the 2×2 dip × next-year-recession table, a 10,000-shuffle permutation test, and a
no-look-ahead "dip → flat next year" timing backtest with one-way costs and a Newey-West
HAC t-stat. The synthetic positive control confirms the machinery finds a planted
dip→recession link; the real tape has none.

> **Honesty note.** No clean public men's-underwear unit-sales series exists, so the
> index here is a curated reconstruction (smooth trend + stylised recession dips). The
> NBER calendar and the Shiller S&P 500 tape are real, so the *test* is genuine even
> though the predictor's raw series is stylised — this is flagged throughout.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, coincident-vs-leading, the 4-dip evidence base, the base-rate trap, the failed trade in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | Fisher exact, permutation distribution, HAC t on the timing spread, the n=33/4-dip power note, the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`underwear_index/`](underwear_index/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
