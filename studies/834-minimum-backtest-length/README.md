# Study 834 — Minimum Backtest Length 📅

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there a real edge to find? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | A synthetic-only method demo on a world we *built* to have **zero** edge (true Sharpe 0). There is nothing real to detect: over a 2-year window **8.4%** of these worthless backtests still post an observed Sharpe ≥ 1.0 (the luckiest hits **2.45**) — a gaudy backtest with no skill behind it. Real free data can never certify "true Sharpe = 0", so the study can never earn `REAL`. |
| **Tradability** — could you trade it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | A backtest shorter than its **MinTRL** is statistically indistinguishable from a coin flip; there is no harvestable edge and costs never enter. Most published strategies (Sharpe 0.3–0.7 on 5–10 years) sit *inside* their MinTRL — a paycheck mirage. |
| **Do short backtests fail to tell skill from luck?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | **Yes.** The required length grows as `(Z/SR)²`: **2.7 yr** at Sharpe 1, **10.8 yr** at 0.5, **43 yr** at 0.25 — and negative skew + fat tails lengthen it a further **×1.65** (monthly, Sharpe 1). The PSR null test is calibrated at its nominal **5%**, yet a *genuine* Sharpe-1 strategy is only **50%** detectable at its 2.71-yr MinTRL. |

> **In one sentence:** a Sharpe ratio without its track-record length is meaningless — Bailey–Borwein–López de Prado's **Minimum Track Record Length** says you need `≈ (Z/SR)²` years before an observed Sharpe clears a 95% bar (43 years at Sharpe 0.25, longer still with fat tails), so the typical 5-year backtest of a Sharpe-0.5 idea is *provably* too short to distinguish skill from luck, which is exactly why almost nothing here survives.

## What we tested

Bailey, Borwein, López de Prado & Zhu (2014), **"Pseudo-Mathematics and Financial Charlatanism"**
(MinTRL formula from Bailey & López de Prado 2012, *The Sharpe Ratio Efficient Frontier*): a Sharpe
estimated over a finite history is noisy, and there is a **minimum track-record length** below which
you cannot reject "true Sharpe ≤ 0" at a chosen confidence — so short backtests cannot tell skill from
luck. Because only a *constructed* world lets us fix the truth, this is a **synthetic/simulation-only**
demonstration (no network, no real market data): a deterministic generator produces tapes with a
**known** annualised Sharpe and a **known** return-distribution shape. We implement the closed-form
MinTRL and Probabilistic Sharpe Ratio, chart how the required length blows up as the Sharpe falls
(`MinTRL ≈ (Z/SR)²`) and how negative skew and fat tails lengthen it, then run a 4,000-path Monte-Carlo
proving (a) short backtests of a *worthless* world routinely post gaudy Sharpes by luck, (b) the PSR
null test is calibrated at 5%, and (c) a *genuine* high-Sharpe series (positive control) is only
reliably confirmed *past* its MinTRL. Survivorship is not in play (clean synthetic data by
construction) — the limitation named on the **Signal** axis is that a synthetic world can never earn
`REAL`. **Dedup:** [344-backtest-overfitting](../344-backtest-overfitting/) is the *multiple-trials*
trap (grid-search inflation, Deflated Sharpe + PBO); [833-deflated-sharpe](../833-deflated-sharpe/)
haircuts a Sharpe for the *trial count*; [345-survivorship](../345-survivorship/) is a
*data-construction* bias — this study isolates the single-strategy **track-record-length** axis (MinTRL
/ PSR). As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a five-year backtest can't prove a mediocre Sharpe, the `(Z/SR)²` blow-up in plain language, and how often a worthless strategy *looks* brilliant by luck |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the MinTRL/PSR closed forms, the skew/kurtosis correction, the calibrated 4,000-path null Monte-Carlo, and the positive-control power curve (MinTRL vs 95%-power length) |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run (fingerprint `c648fb3ad2b5`, as-of 2026-06-30): [docs/results.md](docs/results.md).

---

*Engine: [`min_backtest_length/`](min_backtest_length/) — numpy / pandas / scipy / statsmodels, deterministic and offline. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
