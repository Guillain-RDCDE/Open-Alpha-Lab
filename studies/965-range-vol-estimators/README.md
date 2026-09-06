# Study 965 — The Range Estimators 📏

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the textbook efficiency gain real? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | On simulated bars where the day's true sigma is known, Parkinson's estimator has **4.9x** the efficiency of close-to-close against the truth (Garman-Klass 6.8x, Rogers-Satchell 5.5x) — the textbook claim survives, in the textbook's own world. On the real tape that world does not exist: the overnight gap carries **33%** of SPY's daily variance and none of the three can see it, so they report **69%** of the close-to-close level. |
| **Tradability** — does it improve a forecast you would actually use? | ![Useful](https://img.shields.io/badge/Useful-2ea44f?style=flat-square) | Rescaled to remove that level error, the best range estimator (Yang-Zhang (2000)) beat close-to-close on QLIKE on **5 of 5** tapes with a pooled Diebold-Mariano *t* of **+3.73** — a real but modest improvement in forecasting the next month's realised variance (22.4% lower QLIKE on SPY). Yang-Zhang, the only gap-aware estimator, needs no rescaling and is the honest default. |

> **In one sentence:** The five-to-one efficiency of the range estimators is real in the model they were derived in and misleading on a market that gaps: they miss the **33%** of daily variance that arrives overnight, and once that level error is corrected the remaining forecasting gain is genuine but small — which is why Yang-Zhang, not Parkinson, is the one to use.

## What we tested

Every volatility textbook carries the same table: Parkinson's high-low estimator is
about **five times** as efficient as squared close-to-close returns, Garman-Klass better
still, Rogers-Satchell robust to drift. The claim is a theorem, and the theorem assumes a
driftless continuous diffusion that trades **twenty-four hours a day**. Real markets close.
We test the claim in two places: first on simulated bars with a *known* daily sigma (the only
setting where "efficiency" can be measured at all), then on **SPY, QQQ, IWM, GLD and TLT**,
where the question becomes what the estimators *miss* — the overnight gap, which none of
Parkinson, Garman-Klass or Rogers-Satchell can see — and whether, once that level error is
removed by a burn-in-only rescaling, the extra information in the bar improves a forecast of
the next month's realised variance. Scored with QLIKE and MSE, compared with a HAC-corrected
Diebold-Mariano test, swept across rolling windows.
**Dedup:** distinct from **966-har-vs-garch** (competing *models* of tomorrow's volatility
from a close-only series; this study is about estimating *today's* from one bar),
**817-realized-volatility-trend** and **992-vol-clustering-halflife** (the dynamics of vol,
not its measurement), **374-vol-of-vol** and **130-vol-risk-premium** (implied vs realised),
and **788-overnight-intraday-tug-of-war** (the return in each leg, not the variance
estimator's blindness to one of them).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the high and the low know more than the close, the five-times claim tested where the answer is known, the hole every textbook version has, and what it means for the volatility number you quote |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | closed-form estimators, MSE efficiency against a known sigma with and without gaps, level bias on five tapes, QLIKE/MSE forecast race with burn-in-only rescaling, HAC Diebold-Mariano and a window sweep |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`range_vol/`](range_vol/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
