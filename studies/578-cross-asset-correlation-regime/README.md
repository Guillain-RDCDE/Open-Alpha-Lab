# Study 578 — Cross-Asset-Correlation-Regime 🕸️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does rising cross-asset correlation predict *lower* forward returns? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | **Backwards, robustly.** In the HIGH-correlation regime SPY's forward 21-day return averaged **+1.65%** vs **+0.67%** in LOW — a HIGH−LOW spread of **+0.98%** (two-sample *t* **+6.80**, block-placebo *p* **0.048**), the *opposite* of the fragility claim, and **positive in all 7** window/horizon/quantile specs. Forward *vol* is higher in HIGH (17.2% vs 15.6%, *t* +4.62), so co-movement flags turbulence — but not lower returns. Sample is 2007-2026 (one GFC, one COVID). |
| **Tradability** — does a risk-off overlay pay? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Going flat in the HIGH regime (1-day lag, 5 bps/switch) cut the drawdown (−37.5% vs −55.2%) but **halved the return and lowered the Sharpe**: net CAGR **+5.55%** / Sharpe **+0.42** vs buy-and-hold **+12.20%** / **+0.62**. It sat out exactly the rebounds. Strictly worse portfolio. |
| **"Is average correlation a fragility gauge?"** | ![Coincident](https://img.shields.io/badge/Coincident-8b949e?style=flat-square) | **Coincident, not leading.** HIGH-correlation days are exactly the days SPY is *already* in a drawdown (mean contemporaneous DD **−11.7%** vs **−5.3%**; 91% of 2009, 63% of 2020 were HIGH). Correlations peak near *bottoms*, so the forward window catches the rebound. It marks the storm you're in, not the one ahead. |

> **In one sentence:** the "correlations go to one before a crash" folklore has the timing exactly backwards — on a 14-ETF cross-asset panel over 2007-2026, average correlation is a *coincident* stress gauge (it spikes when SPY is already deep in a drawdown and it does forecast higher near-term volatility), but as a *forward* signal the HIGH-correlation regime was followed by **higher** SPY returns (+0.98%/month, *t* +6.80, placebo *p* 0.048, sign stable across all seven specs), because correlation peaks near bottoms — so a risk-off overlay built on it underperforms buy-and-hold (Sharpe +0.42 vs +0.62).

## What we tested

The practitioner maxim that *when everything starts moving together, the market is about to break* —
a high / rising **average cross-asset correlation** read as a **fragility indicator** that should
predict lower forward returns and higher forward volatility (Longin & Solnik 2001; Ang & Chen 2002;
Pollet & Wilson 2010). We compute the trailing-63-day mean off-diagonal pairwise correlation across a
14-ETF cross-asset panel (equities, bonds, credit, gold, commodities, oil, REITs, silver, EM), split
the tape into a HIGH regime (above its expanding 70th percentile, past data only) and a LOW regime,
and test SPY's forward 21-day return and vol across the two: a two-sample *t*, a **block-shuffle
placebo** null (for the overlapping forward windows), a **risk-off overlay** backtest (gross AND net,
one execution lag), a **seven-spec robustness sweep**, and a deterministic, seed-robust synthetic
positive control that plants the fragility effect and proves the engine catches it. *Distinct from
[245 Oil-Equity-Correlation](../245-oil-equity-correlation/) (a single pair) and
[502 Betting-Against-Correlation](../502-betting-against-correlation/) (a cross-sectional stock sort);
this is the **panel-average time-series regime**.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "average correlation" is, why "everything moving together = danger" feels right, and why on the tape it flagged the *rebound* not the crash |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the correlation index, the expanding-quantile regime split, the two-sample *t* + block-placebo, the coincident-drawdown mechanism, the seven-spec sign-stability sweep, the risk-off overlay net of costs, and the seed-robust synthetic positive control |

The fingerprinted real-data run (14-ETF panel 2007-01-04 → 2026-06-26, returns fp `9dbfa917d6ba`,
regime/forward frame fp `0b73f61632f6`, as-of 2026-06-30) is in [docs/results.md](docs/results.md);
the offline machinery proof runs on the deterministic synthetic world in
[`cross_asset_correlation_regime/data.py`](cross_asset_correlation_regime/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`cross_asset_correlation_regime/`](cross_asset_correlation_regime/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
