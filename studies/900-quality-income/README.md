# Study 900 — Quality-Income 💎

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does quality beat yield on risk-adjusted return? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | Quality (SCHD+NOBL) edges yield (SPHD+VYM) by +0.9 pp/yr and a **+0.047** excess Sharpe gap, with a genuinely shallower max drawdown (**−22.4%** vs **−27.5%**) from dodging the 2020 yield-sleeve crater (**+11.7%** vs **−4.7%**). But the risk-adjusted edge is **insignificant**: NW *t* **+0.57**, bootstrap 95% CI **[−0.19, +0.24]** straddles zero (P(gap<0)=0.36), no era significant. A real trap-avoidance *profile*, not a certified premium. Short single-regime tape (young ETFs, NOBL-bound to 2013). |
| **Tradability** — can you bank it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | The tilt is trivially buyable — cheap, liquid, long-only, monthly rebalance costs **< 1 bp/yr** (turnover ~0.6%/mo), so nothing erases it (not a Mirage). But there is no significant premium to bank: you buy a shallower-drawdown *risk profile*, not an edge — much like a min-vol wrapper. |
| **"Beat SPY?"** | ![Busted](https://img.shields.io/badge/Beat_SPY%3F-Busted-8b949e?style=flat-square) | **No.** Both dividend sleeves trailed plain SPY by ~3-4 pp/yr and ~0.17-0.21 of Sharpe (quality −2.81%/yr *t*=−1.45; yield −3.62%/yr *t*=−1.44). The dividend debate is quality-vs-yield; on dividend-vs-market, SPY won. |

> **In one sentence:** screening dividends for quality *does* dodge the yield-trap
> blowups — a 5-point-shallower drawdown and a +11.7% vs −4.7% swing in 2020 — but the
> risk-adjusted edge over raw yield is only +0.05 Sharpe (NW *t* 0.57, CI straddles zero),
> a real drawdown cushion rather than a certified premium, and **both** dividend sleeves
> trailed just owning SPY.

## What we tested

Two long-only, equal-weight, **monthly-rebalanced** dividend sleeves built from live ETFs
— **quality** = SCHD + NOBL (durable / growing payers), **yield** = SPHD + VYM (raw
high-yield screens) — raced against each other and **SPY** on monthly **total returns**,
all **excess of cash** (minus BIL, whose monthly return *is* the realized cash return),
over the common window 2013-11 → 2026-06 (152 months, NOBL-bound), as-of 2026-06-30. We
report the excess-of-cash Sharpe race, the quality-minus-yield HAC *t*, a paired
moving-block bootstrap CI on the Sharpe **gap**, max drawdown, a calendar-year table, an
era cut (split 2020-01), and a costed (turnover × spread) net version. A deterministic
synthetic world with a planted quality-over-yield edge proves the machinery. Short-history
/ survivor selection is named on the Signal axis. **Dedup:** siblings
[206-dividend-aristocrats](../206-dividend-aristocrats/) (academic aristocrat signal),
[233-shareholder-yield](../233-shareholder-yield/) (total-yield factor), and
[57-yield-trap](../57-yield-trap/) (the trap phenomenon) grade the *signals*; this study is
the *product race* — quality-dividend wrapper vs high-yield wrapper — built on the
[601-factor-etf-live-test](../601-factor-etf-live-test/) live-ETF template.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "value trap" means, why the quality sleeve rode +11.7% through 2020 while the yield sleeve fell −4.7%, why a shallower drawdown ≠ a market-beating edge, in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the excess-of-cash Sharpe race, quality-minus-yield NW *t*, the block-bootstrap Sharpe-gap CI, the era cut, the vs-SPY races, the costed sleeves, and the planted-edge synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quality_income/`](quality_income/). The audited unit is the LIVE product sleeve
net of its own fee; the only trading is a monthly rebalance to equal weight (one clean
within-month drift, no look-ahead), costed at the ETF spread. **Not investment advice** —
research & education. See [LICENSE](../../LICENSE).*
