# Study 260 -- Margin-Debt

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The contrarian *sign* is correct and stable in every sub-period (record-margin years average **+6.6%** forward vs **+11.1%** otherwise; YoY-growth regression slope negative throughout), and it rhymes with the 2000/2007/2021 tops. But the full-sample HAC *t* is only **-1.35** (regression), **-0.67** (terciles), **-1.22** (record event) -- nothing clears the *t* = 2 bar. Folklore + right-but-insignificant tape = **Weak**, not Real. Price-only S&P (dividends excluded), which is the *conservative* direction for a "sell" claim. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A "go to cash after a margin surge" rule *underperforms* passive buy-and-hold by **~1pp/yr** net of 10 bps/switch -- it sat out 8 of 65 forward-years and missed more up-years than crashes it dodged. Margin debt is a coincident by-product of rising prices, not a leading indicator of falling ones. |

> **In one sentence:** record/fast-rising NYSE-FINRA margin debt carries the *right contrarian sign* -- record years really do average lower forward returns, echoing the famous post-record crashes -- but on 65 non-overlapping years the effect never clears *t* = 2 in any test, and a tradable contrarian timing rule loses to buy-and-hold, making this a *weak, untradable* piece of folklore rather than a real timing signal.

## The claim

> *Is record NYSE margin debt a contrarian sell signal?*

## What we tested

We hardcode the December (year-end) NYSE/FINRA customer **margin-debt** series
(1959-2025) in `data.py` and pair it with month-end S&P 500 closes (price-only).
For each year we compute the year-over-year change in margin debt and, after a
one-month reporting/execution lag, the forward 12-month return. We test the
contrarian claim ("more leverage -> lower forward returns") four ways: (a) a HAC
OLS predictive regression, (b) high vs low YoY-growth terciles, (c) a literal
fresh-all-time-high "record" event test against the unconditional up-rate, and
(d) a tradable "cash after a surge" timing rule vs buy-and-hold, net of one-way
costs on NAV. A deterministic synthetic positive control confirms the regression
recovers a *real* planted contrarian signal at *t* = -5 to -11.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the viral chart, why the level trends with the market, the record-high bar chart, the famous post-record crashes, and why the timing rule loses |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC regression, YoY terciles, record-high event test, sub-period robustness, the buy-and-hold equity curve, and the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`margin_debt/`](margin_debt/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
