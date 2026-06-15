# Study 202 — Fifty-Two-Week-Low

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Q1-Q5 spread = **−7.42 bps/week**, HAC *t* = **−0.67**; negative at every tested horizon (1d to 65d); only the 1-day horizon reaches |*t*| ≥ 2, and it does so in the **wrong direction** (*t* = −2.00, momentum wins). Survivorship-biased — the unbiased result is likely more negative. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Bottom decile earns **+8.5%/yr** against a passive equal-weight basket at **+11.3%/yr** — the strategy lags the market before any transaction costs. Near-low stocks carry wider spreads and lower liquidity, making execution drag worse than for high-proximity names. |
| **Contrarian?** | ![Losers Keep Losing](https://img.shields.io/badge/Contrarian%3F-Losers__Keep__Losing-8b949e?style=flat-square) | The mirror of the 52-week-high momentum anomaly does not pay off. Stocks near their 52-week low systematically underperform stocks near their high in this sample — mild momentum, not mean-reversion. |

> **In one sentence:** buying stocks near their 52-week low as a "contrarian bargain bet" does not work — they underperform stocks near their 52-week high at every tested horizon, lag the equal-weight market before costs, and on a survivorship-biased sample the result is likely rosier than reality.

## What we tested

The mirror of the 52-week-high momentum anomaly (George & Hwang 2004): rank S&P 500 names cross-sectionally by proximity to their 52-week low ((close − 252d-low) / (252d-high − 252d-low)), form equal-weight quintile portfolios, and measure forward returns over 1 to 65 trading days. Q1 (stocks nearest the 52-week low) is the contrarian "bargain buy"; Q5 (stocks nearest the 52-week high) is the momentum long. We also compare a long-only bottom-decile strategy against the equal-weight basket — the fair benchmark for a long-only investor. Universe: 20 representative S&P 500 large-cap names, 2013-01-02 to 2026-06-15 (~13 years, 3,383 panel-days). **Survivorship-biased** — all names still trade in 2026; excluded delisted companies (which hit 52-week lows and kept falling) would worsen the contrarian result.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, why the contrarian story feels right, the quintile chart that runs backwards, why the bounce never materialises, the cost argument |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC *t*-stats per quintile, hold-period sweep (1d to 65d), long-only vs equal-weight comparison, synthetic positive control confirming the engine is a faithful mean-reversion detector |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`fifty_two_week_low/`](fifty_two_week_low/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
