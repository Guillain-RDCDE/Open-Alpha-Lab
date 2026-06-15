# Study 203 — Golden-Butterfly

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Real (risk-adjusted)](https://img.shields.io/badge/Signal-Real_(risk--adjusted)-2ea44f?style=flat-square) | GB Sharpe **0.682** vs 0.532 (SPY); bootstrap 95% CI [−0.10, +0.39], GB wins 88% of resamples vs SPY. But the PP wins 84% vs GB on Sharpe — the fifth leg adds CAGR at the cost of risk-adjusted quality. Directionally robust vs SPY, not certified at 5%. |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | 5 liquid ETFs, annual rebalance, <1 bp CAGR cost impact. Fragile: IWN amplifies equity drawdowns (−59.5% in GFC vs SPY's −55.2%), the small-cap-value premium has had decade-long droughts, and the bond+gold secular tailwind of 2004-2021 inflates the Sharpe. |
| **Better than PP?** | ![No](https://img.shields.io/badge/vs_Permanent_Portfolio%3F-Worse_on_Sharpe-8b949e?style=flat-square) | GB Sharpe 0.682 vs PP 0.782; PP wins 84% of bootstrap Sharpe resamples. The small-cap-value fifth leg added +0.34 pp/yr CAGR but hurt risk-adjusted quality. The PP remains the cleaner design. |

> **In one sentence:** Tyler's Golden Butterfly (20% each: large-cap / small-cap value / long bonds / short bonds / gold) beats SPY on risk-adjusted terms but loses to the simpler Permanent Portfolio on Sharpe — the small-cap-value fifth leg is a return booster in calm markets and a drawdown amplifier in crashes, leaving the portfolio better than 60/40 and worse than the PP on risk-adjusted terms.

## What we tested

A lazy portfolio recipe popularised by PortfolioCharts.com (Tyler, 2015): divide
savings equally across five asset classes — SPY (US large-cap), IWN (US
small-cap value), TLT (long Treasuries), SHY (short Treasuries/cash), GLD (gold)
— and rebalance once per year. The design is a deliberate extension of Harry
Browne's Permanent Portfolio (Study 144): it adds the small-cap-value premium
(Fama-French HML+SMB) and spreads bond duration risk across short and long
maturities. We run the exact recipe over 21 years (2004-2026, GLD-constrained)
and compare against 100% SPY, 60/40 SPY/TLT, and the Permanent Portfolio with
bootstrap Sharpe CIs, Newey-West HAC t-stats, and a regime crash table.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the equity curve, the crash table showing IWN's drawdown amplification, the return trade-off chart, and why the PP beats the GB on Sharpe |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | bootstrap Sharpe CIs for all four comparisons, HAC t-stats, regime crash table, synthetic positive control with tunable cycle strength |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`golden_butterfly/`](golden_butterfly/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
