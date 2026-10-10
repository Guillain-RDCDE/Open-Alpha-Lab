# Study 1026 — Gibson's Paradox 🪙

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do long yields follow the price level rather than inflation? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) — present in the US changes · absent in Germany | **US 1957-2018:** in first differences the detrended price level beats 12-month inflation (Newey-West t **+3.56** vs +0.38) and improves real-time forecasts of next year's yield change (Clark-West p < 0.001). **Germany 1972-98:** nothing (t +0.07). And the US half is not the paradox it looks like: yields are **not** cointegrated with the price gap (Engle-Granger p 0.19, despite a naive levels t of +36.8) but **are** with a slow adaptive inflation expectation (p 0.009) — Fisher's own 1930 resolution. A 10-year-rolling detrend kills the forecasting edge (p 0.82). No price index before 1957 on these tapes, so the gold standard itself is untested. |
| **Tradability** — can it time duration? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | A 0.5×/1.5× duration overlay on a 20-year AAA par bond earns **+1.95% a year net over constant duration** (HAC t +2.63; net excess Sharpe 0.54 vs 0.50; 5 bp one-way, CPI lagged a month, one execution lag). But it sits at 1.43× duration on average, trades 0.13× NAV a year, made −0.05% a year in 1965-81 against +2.44% in 1982-2018, and its alpha after the extra duration is +0.65% (t +1.01). One secular long-duration call, not a repeatable rule. |

> **In one sentence:** Under fiat money Gibson's paradox survives only as a shadow of Fisher — the detrended price level beats 12-month inflation on the US tape but not in Germany, it is not cointegrated with yields while long-memory inflation expectations are, and the duration timer it implies earns +1.95% a year net from what is really one long bet on the 1982-2018 bond bull.

## What we tested

Keynes (1930, *A Treatise on Money*) named it **Gibson's paradox**: for over a century British
long yields moved with the **price level**, not with inflation as Fisher's theory requires; Barsky
& Summers (1988, *JPE*) explained it as a gold-standard phenomenon. The investor's version is
"yields follow prices, so the price level tells you where yields go". We take it to Moody's AAA
yields against US core CPI (1957-2018), US headline CPI (1959-2009) and a German long rate and
deflator (1972-98) — all frozen inside the `arch` / `statsmodels` wheels — through unit roots,
Engle-Granger cointegration, a first-difference horse race against inflation, an adaptive-
expectations confound, expanding-window forecasts with Clark-West, and a costed duration overlay,
with a synthetic Fisher-world / Gibson-world control. **Dedup:** **152-inflation-hedge** asks
whether *stocks* hedge inflation, **119-real-rate-regime** times equities on real rates,
**625-starting-yield-bond-decade** and **581-term-premium** forecast bond returns from the yield
itself; none tests whether yields follow the price *level* versus the inflation *rate*.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a beautiful chart of yields and prices proves nothing, the three tests that would, and what a "price-level" bond timer really did |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | ADF/KPSS, Engle-Granger, the Newey-West horse race by regime and tape, the adaptive-expectations grid, Clark-West, the overlay with cost sweep and alpha, and the two-world synthetic control |

Reproducible headline run: [docs/results.md](docs/results.md) (as-of 2018-11-30, fingerprinted; `python examples/verify.py`).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/) + [`gibson/`](gibson/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
