# References — Study 896 (Risk-Parity + Trend)

## The claim's source

Combining **risk parity** with **trend following** is a staple of the multi-strategy /
"crisis-alpha" allocation literature and of practitioner decks: risk parity earns a
diversified premium in normal times, trend following de-risks in sustained bear markets,
and stacking the two is pitched as improving both the risk-adjusted return *and* the
tail. This study tests the simplest concrete version — an inverse-vol risk-parity book
across SPY/TLT/GLD/DBC with a per-sleeve 200-day trend gate that parks a falling sleeve in
T-bills — and asks whether adding trend to risk-parity really improves the excess-of-cash
Sharpe and drawdown, net of costs.

## Key papers

- **Qian, E. (2005)** — *Risk Parity Portfolios: Efficient Portfolios Through True
  Diversification*, PanAgora. The foundational statement of risk (not dollar) budgeting —
  the inverse-vol book this study bolts trend onto.
- **Asness, C., Frazzini, A. & Pedersen, L. (2012)** — *Leverage Aversion and Risk
  Parity*, **Financial Analysts Journal** 68(1), 47–59. Why an unlevered risk-parity book
  keeps a smaller drawdown but a lower return than 60/40 — the trade-off the trend gate
  then reshapes. <https://doi.org/10.2469/faj.v68.n1.1>
- **Moskowitz, T., Ooi, Y. H. & Pedersen, L. (2012)** — *Time Series Momentum*, **Journal
  of Financial Economics** 104(2), 228–250. The evidence that a trailing trend signal
  (their 12-month; here the 200-day SMA gate) predicts each asset's own future return and
  de-risks in downtrends. <https://doi.org/10.1016/j.jfineco.2011.11.003>
- **Faber, M. (2007)** — *A Quantitative Approach to Tactical Asset Allocation*, **Journal
  of Wealth Management** 9(4). The 10-month / 200-day moving-average timing rule this
  study uses as its per-sleeve gate; the direct ancestor of study 110.
  <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=962461>
- **Hurst, B., Ooi, Y. H. & Pedersen, L. (2017)** — *A Century of Evidence on
  Trend-Following Investing*, AQR. Trend following as a diversifying, crisis-robust
  overlay across a century — the steelman for stacking it onto risk parity.
- **Baltas, N. (2015)** — *Trend-Following, Risk-Parity and the Influence of
  Correlations*, in *Risk-Based and Factor Investing* (Elsevier). Directly on the
  interaction this study measures — how a trend overlay reshapes a risk-parity book.
- **Newey, W. K. & West, K. D. (1987)** — *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, **Econometrica**
  55(3), 703–708. The HAC *t*-statistics used throughout.
- **Politis, D. N. & Romano, J. P. (1994)** — *The Stationary Bootstrap*, **JASA** 89(428),
  1303–1313. The block bootstrap behind the paired Sharpe-difference confidence interval.

## Named siblings on this desk (the dedup guard)

- [68-all-weather](../../68-all-weather/) — the **plain** inverse-vol risk-parity book
  (SPY/IEF/GLD/DBC), no trend gate. This study *is* 68's book plus a 200-day per-sleeve
  trend gate, and the head-to-head race is precisely RP+trend **vs** that plain RP.
- [110-faber-timing](../../110-faber-timing/) — the **single-asset** 10-month / 200-day
  SMA timer (in-or-out of one index). Here the same moving-average rule is applied
  **per sleeve inside a diversified risk-parity budget**, and de-risked sleeves fund a
  T-bill leg rather than exiting the whole book.
- [595-managed-futures-allocation](../../595-managed-futures-allocation/) — a **managed-
  futures / trend sleeve added to a portfolio** as a separate allocation; this study
  instead gates the existing sleeves in place, never adding a new asset.
- [894-trend-6040](../../894-trend-6040/) — a trend overlay on a **fixed 60/40** book; the
  budget there is dollar-weighted 60/40, here it is the **inverse-vol risk-parity** budget
  across four sleeves, so the diversification baseline and the gating unit differ.
- [656-dragon-portfolio](../../656-dragon-portfolio/) — a fixed offense/defense **cocktail**
  (Cole's "Dragon") that *includes* a trend allocation among other sleeves; this study is
  not a static mix but a **trend gate applied in place** to a risk-parity budget.

## Data sources

- **SPY / TLT / GLD / DBC / IEF / BIL daily total-return closes** — Yahoo! Finance via
  `yfinance` (public, no key), `auto_adjust=True` (dividends & coupons reinvested).
  2007-05-30 → 2026-06-30, cached under
  [`_cache/rp_trend_prices.parquet`](../_cache/). BIL's 2007 inception is the binding
  common-sample start; TLT/GLD/DBC list in 2002–2006, so the tape covers 2008, 2020 and
  2022 but no earlier crisis.

## Shared method citations

- Desk house style, inference bar and shared protocol: [`METHODOLOGY.md`](../../../METHODOLOGY.md).
- Reproducibility stamp (as-of + fingerprint): [`quantlab/repro.py`](../../../quantlab/repro.py).
