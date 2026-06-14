# Study 139 — AI-Powered-ETF

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Signal-WEAK-dab617?style=flat-square) | Jensen alpha **-4.40%/yr**, HAC *t* = **-1.28** — consistently negative but below the significance bar on 8.6 years; power-limited by AIEQ's idiosyncratic vol (~8%/yr). |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Tradability-MIRAGE-c0392b?style=flat-square) | AIEQ CAGR **+9.88%/yr** vs SPY **+14.90%/yr**; Sharpe 0.531 vs 0.826; max DD -39% vs -34%. IBM Watson delivered worse returns *and* worse risk over the full 8.6-year live sample. |
| **AI picks vs buy-and-hold?** | ![Busted](https://img.shields.io/badge/AI_picks_vs_buy--and--hold%3F-Busted-8b949e?style=flat-square) | AIEQ outperformed SPY in only **31%** of rolling 12-month windows; the 0.75% ER adds persistent fee drag on top of negative gross alpha. |

> **In one sentence:** the AI-Powered ETF (IBM Watson) has trailed the S&P 500 by ~5 percentage points per year since launch, underperforming on return, Sharpe, and drawdown, with a HAC t-stat on Jensen alpha of -1.28 — the hype did not survive contact with the market.

## What we tested

AIEQ (ticker: AIEQ), the "AI Powered Equity ETF" launched October 2017 by EquBot LLC
using IBM Watson to select ~30-70 US equities from a universe of 6,000+, rebalanced
daily. The marketing claim: AI processes more data faster than human analysts, identifies
mispriced stocks, and delivers consistent outperformance net of its 0.75% expense ratio.
We take that at face value and run the honest test: AIEQ vs SPY (the S&P 500 tracking
ETF at ~0.09% ER) and IVV, using daily adjusted-close returns over the full 8.6-year live
history (Oct 2017 – Jun 2026). The comparison is CAPM alpha (Jensen), risk-adjusted return
(Sharpe), and drawdown — with HAC Newey-West t-stats and bootstrap confidence intervals.
Power is limited (8.6 years with ~8%/yr idio vol implies a minimum detectable alpha of
~4.5%/yr at 95% confidence) — we name this clearly.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, what Watson promised, the actual track record vs the S&P 500 in plain language, and why fees alone don't explain the whole shortfall |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | Jensen alpha, HAC t-stat, bootstrap Sharpe CI, rolling outperformance windows, synthetic positive control confirming detection works when real alpha exists |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`ai_powered_etf/`](ai_powered_etf/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
