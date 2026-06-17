# Study 267 — M2-Growth

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Lagged M2 YoY has no robust predictive content for forward ^GSPC returns: 1-month HAC **t ≈ 0.0**; the 12-month naive OLS **t = −2.0 evaporates to HAC t = −0.6** under the overlapping-window correction; the high-minus-low monthly spread is **−1.2pp** (t = −1.87) — *negative*, the wrong sign for the bullish claim. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The long/flat "M2 above trailing median" rule earns **+2.8%/yr** net vs **+9.3%/yr** for passive buy-and-hold, at half the Sharpe (0.32 vs 0.67), in the market only 40% of the time. |
| **Busted?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The famous M2-vs-S&P overlay is *contemporaneous*: money and stocks both surge when the Fed eases into a downturn. Lag the (already-lagged) M2 release and demand a HAC t-stat and the edge vanishes. |

> **In one sentence:** money-supply growth does not drive next month's stock market — the seductive overlay chart is the business cycle moving both series at once, and once you trade it with a proper execution lag and an honest Newey-West t-stat, there is nothing there.

## What we tested

The monetarist claim that **M2 growth drives equities**. We hardcode a monthly
M2 year-over-year growth series (FRED M2SL contour, 1960–2025, including the +27%
COVID spike and the 2023 negative prints) in `data.py`, join it to ^GSPC monthly
price returns (yfinance, cache-only), and impose a one-month execution lag (M2 is
itself released weeks late). We then run a **predictive HAC regression** of
forward 1-month and 12-month returns on lagged M2 growth, a **quantile sort** of
high-money vs low-money months, and a **tradable long/flat rule with costs** vs
buy-and-hold. A synthetic positive control confirms the engine detects a planted
M2 premium when one exists; the real tape has none (and the sign is wrong).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the overlay chart, why it deceives, the lag, the plain-English verdict |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC vs OLS on overlapping windows, quantile sort, the long/flat backtest, the positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`m2_growth/`](m2_growth/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
