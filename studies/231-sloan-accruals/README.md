# Study 231 — Sloan Accruals

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | On the survivor-biased S&P 500 panel: hedge +5.7%/yr, HAC *t* = +2.73 — but virtually all the hedge is long-side (low-accruals excess +5.7%/yr); high-accruals excess = −0.1%/yr (*t* = −0.04). The t-stat is a survivorship-biased upper bound; the academic anomaly has substantially decayed on large caps post-2000. |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Annual rebalancing on ~80 S&P 500 names is cheap; but the alpha estimate is uncertain (upper bound only), the short side does not contribute meaningfully, and literature shows the Sloan anomaly has been arbitraged away in large-cap liquid stocks. |
| **Accruals anomaly still real on large caps?** | ![Decayed](https://img.shields.io/badge/Accruals_on_LargeCaps-Decayed-8b949e?style=flat-square) | Sloan (1996) documented a ~10%/yr effect on the full US stock universe. Post-publication decay is well-documented (Green et al. 2011); the anomaly survives primarily in small, illiquid, low-coverage stocks — not in S&P 500 names. |

> **In one sentence:** the low-accruals (cash-backed earnings) portfolio outperforms the high-accruals portfolio on the S&P 500 survivor panel by +5.7%/yr, but essentially all the gain is long-side quality tilt — the short leg earns market returns, and the academic anomaly has largely been arbitraged away in large caps since Sloan (1996) was published.

## What we tested

Sloan (1996) showed that firms whose earnings are backed by accounting accruals — rather than real cash flows — earn systematically lower future returns. The mechanism: investors fail to discount the lower persistence of accrual earnings; when accruals mean-revert, earnings disappoint and prices fall.

We compute accruals using the **cash-flow statement method** (Net Income − Operating Cash Flow) / Average Total Assets — the cleaner post-SFAS 95 version used by Richardson et al. (2005). We sort the current S&P 500 survivors into quintiles on this signal, lag fundamentals by one full year (fiscal year y → calendar year y+1 returns), and test whether the low-accruals quintile outperforms the high-accruals quintile. A deterministic synthetic tape with tunable accruals premium serves as the positive control.

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the accruals recipe in plain English, the survivor-bias and post-publication-decay caveats, year-by-year results |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | quintile return monotonicity, HAC t-stats, random-portfolio null distribution, synthetic positive control sweep |

## Do high-accrual firms really underperform the cash-earners?

On S&P 500 survivors from 2009 to 2026: the low-accruals quintile earns +23.8%/yr vs the high-accruals quintile's +18.1%/yr — a hedge of +5.7%/yr. But the high-accruals quintile earns almost exactly market return (+18.2%/yr equal-weight). The entire hedge is long-side: cash-backed earnings (low accruals) predict high quality and future returns; accrual-heavy earnings on surviving large caps do not predict dramatic underperformance. On the original broader (non-biased) universe of all US stocks, Sloan (1996) found the anomaly was large; on large liquid names, it has been documented to have decayed substantially after publication.

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`sloan_accruals/`](sloan_accruals/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
