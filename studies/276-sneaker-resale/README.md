# Study 276 — Sneaker-Resale

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**Does the sneaker resale market (StockX) beat stocks?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Sneakers **underperform** stocks: excess return **−2.4%/yr**, t = **−0.57**, HAC t = **−0.58**. The only "edge" is a higher *reported* Sharpe (0.71 vs 0.55), which is a stale-pricing artifact: AR(1) ρ = **+0.63**, and unsmoothing collapses the Sharpe to **0.29**. n = 18. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No investable index exists; physical resale carries ~10–15% one-way marketplace fees, authentication, storage, and no shorting. Net of a 3%/yr friction, sneakers return ~**+4.5%/yr** — roughly half the S&P. |
| **Busted?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The "alternative asset class that beats stocks" is unbuyable as an index and uneconomic as a hustle; its attractive risk profile is just stale appraisal marks. |

> **In one sentence:** the StockX sneaker market did not beat the S&P 500 — it returned *less* (+7.0% vs +8.2% CAGR), and its famously "low-risk" profile is an illusion created by smoothed, infrequent appraisal pricing that hides ~23% true volatility.

## What we tested

We hardcode a curated **annual sneaker-resale price index** (2006–2024, base 100) in
`data.py` — encoding the documented slow-grind → StockX/GOAT boom → 2022–2024 bust —
and join it to **^GSPC** (S&P 500 price index) calendar-year returns from the repo-level
cache. We then ask three honest questions: (1) does it beat stocks on return (one-sample
and Newey-West HAC t-tests on the annual excess); (2) is its high Sharpe real or a
stale-pricing artifact (AR(1) Geltner unsmoothing); (3) could you actually trade it
(one-way frictions, no shorting, no vehicle, a lagged trend overlay). The synthetic
positive control confirms the engine finds a planted alpha when one exists; the real
tape has none.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the pitch, the boom-and-bust chart, the Sharpe illusion, the frictions, in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | excess-return t-stats, Newey-West HAC, AR(1) unsmoothing, the n=18 power calculation, the synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`sneaker_resale/`](sneaker_resale/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
