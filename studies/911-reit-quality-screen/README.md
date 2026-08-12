# Study 911 — REIT Quality Screen 🏢

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a durable-income equity-REIT screen beat the broad REIT index, and is the mortgage-REIT carry a trap? | ![Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square) | **Two legs, two answers.** The **durable-income tilt is not certified**: the residential quality sleeve (REZ) edges the broad index (VNQ) on excess Sharpe (**0.36 vs 0.29**) but the monthly spread is only **+8.7 bps/mo at HAC *t* = 0.75**, the bootstrap Sharpe-advantage CI **[−0.07, +0.20] straddles zero**, and it is not era-robust (*t* = 0.08 → 1.09). The **leveraged-carry *trap* is real and structural**: mortgage REITs (REM) earned **−1.13%/yr for 19 years** at excess Sharpe **0.039** — an order of magnitude below the equity sleeves, a mechanical yield trap — but the broad index already excludes it. *Short-history: these sector ETFs are young; magnitudes indicative.* |
| **Tradability** — can you bank an edge over the index you'd otherwise hold? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | The one robust action — *avoid the mortgage-REIT carry* — is **already free** inside any broad equity-REIT ETF (VNQ holds ≈ no mortgage REITs). The incremental quality tilt is too thin to bank: the costed quality book nets **~+0.1–0.2%/yr at *t* ≈ 0.3** over VNQ (costs barely matter — the gross edge is only ~2 bps/mo), and the stronger single REZ tilt runs its Sharpe-advantage CI through zero. A real *distinction*, not a bankable *premium*. |

> **In one sentence:** equity REITs and mortgage REITs really are different animals — the
> mortgage-REIT levered carry is a genuine **total-return trap** (−1.13%/yr for 19 years) —
> but the broad index already screens it out for free, and the incremental "quality" tilt of
> a residential sleeve over the broad index is **too thin to certify or bank**.

## What we tested

**The claim:** a "quality REIT" screen — hold the durable-income equity sleeve (residential
**REZ**, broad **VNQ**/**RWR**), screen *out* the leveraged mortgage-REIT carry (**REM**) —
delivers a better *risk-adjusted, net-of-cost* return than the broad REIT index. We race the
live vehicles **excess-vs-excess** (all minus the **BIL** T-bill) over a **229-month common
sample (2007-06 → 2026-06, yfinance daily total-return closes)**, with **SPY** the equity
control: a Newey-West HAC *t* on monthly spreads, a paired circular-block-bootstrap
Sharpe-advantage CI, a two-era cut (split 2017-01), daily max-drawdowns, a monthly-rebalanced
costed quality book, and a 20-seed synthetic control. Young sector ETFs → short-history
caveat, named on the **Signal** axis. **Dedup:** [207-reits-diversifier](../207-reits-diversifier/)
tests REITs as a **portfolio diversifier** (REITs-vs-other-assets), not quality *within* the
complex; [611-mreit-carry](../611-mreit-carry/) studies the **mortgage-REIT carry trade**
itself (here REM is only the *foil*); [341-mlp-pipelines](../341-mlp-pipelines/) is the
**energy-MLP** income complex, a different asset; [246-defensive-sectors](../246-defensive-sectors/)
is a **low-vol defensive-sector** tilt, not a within-real-estate leverage/income screen.
As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why equity and mortgage REITs are different animals — the durable-income edge that isn't, and the levered-carry trap that is |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the excess-Sharpe race, the HAC spread *t* + bootstrap Sharpe-advantage CI + era cut, the trap decomposition, the drawdowns, the costed book, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`reit_quality/`](reit_quality/). Total-return tape via yfinance (`auto_adjust=True`),
cached under `_cache/`. Young sector ETFs → magnitudes are indicative, named on the Signal axis.
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
