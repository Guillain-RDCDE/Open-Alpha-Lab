# References — Study 623 (IPO Long-Run Underperformance)

## The claim's source

- **Ritter, J. R. (1991)** — *The Long-Run Performance of Initial Public Offerings*,
  **Journal of Finance 46(1), 3-27**. The origin of the claim: 1,526 IPOs from 1975-84
  underperform matched seasoned firms by ~29% over the three years after listing.
  https://doi.org/10.1111/j.1540-6261.1991.tb03743.x
- **Loughran, T. & Ritter, J. R. (1995)** — *The New Issues Puzzle*, **Journal of Finance
  50(1), 23-51**. Extends the drift to five years and to SEOs; "investing in firms issuing
  stock is hazardous to your wealth." https://doi.org/10.1111/j.1540-6261.1995.tb05166.x
- **Ritter, J. R. & Welch, I. (2002)** — *A Review of IPO Activity, Pricing, and
  Allocations*, **Journal of Finance 57(4), 1795-1828**. Table I is the cohort table this
  study re-runs (in its continuously updated form).
  https://doi.org/10.1111/1540-6261.00478

## The counter-literature (the third axis)

- **Brav, A. & Gompers, P. A. (1997)** — *Myth or Reality? The Long-Run Underperformance
  of IPOs*, **Journal of Finance 52(5), 1791-1821**. The style rebuttal: matched on size and
  book-to-market, IPOs do *not* underperform — the drag lives in small growth firms
  generally, IPO or not. https://doi.org/10.1111/j.1540-6261.1997.tb02742.x
- **Fama, E. F. (1998)** — *Market Efficiency, Long-Term Returns, and Behavioral Finance*,
  **Journal of Financial Economics 49(3), 283-306**. The methodological warning: long-run
  BHAR "anomalies" are fragile to the benchmark model and weighting scheme.
  https://doi.org/10.1016/S0304-405X(98)00026-9
- **Ritter's own style adjustment** (Table 19, style-adjusted column) concedes the point
  quantitatively: −20.5% market-adjusted shrinks to −8.9% style-adjusted, 1980-2024.

## Data sources

- **Ritter's IPO data page** — https://site.warrington.ufl.edu/ritter/ipo-data/ ; the cohort
  table used here is Table 19 of *Initial Public Offerings: Updated Long-run Statistics*
  (April 7, 2026 edition; Table 19 updated February 16, 2026),
  https://site.warrington.ufl.edu/ritter/files/IPOs-long-run-returns-on-IPOs.pdf —
  downloaded once, cached in `_cache/`, and hardcoded in
  [`data.py`](../ipo_long_run_underperformance/data.py) with the source comment.
- **yfinance** (public, no key) — daily adjusted total-return closes for **IPO**
  (Renaissance IPO ETF — a rules-based basket of recent IPOs held ~2-3 years post-listing,
  i.e. exactly the aftermarket window Ritter measures, in investable form), **SPY**,
  **IWM**, **IWO** (Russell 2000 Growth, the small-growth style benchmark) and **^IRX**
  (13-week T-bill rate). Cached under `_cache/ipo_tape.csv`.
- **Renaissance Capital, IPO ETF methodology** — https://www.renaissancecapital.com/IPO-Investing/US-IPO-ETF
  (Renaissance IPO Index: new listings added on a fast-entry/quarterly basis, removed after
  ~3 years of seasoning).

## Named siblings on this desk (the dedup guard)

- [`219-ipo-pop`](../../219-ipo-pop/) — the **day-1** effect: the first-day pop that goes
  to allocants, not to you. This study starts *after* that pop (all long-run returns are
  measured from the first closing market price, never the offer price).
- [`265-ipo-volume`](../../265-ipo-volume/) — IPO **volume as a timing signal** for the
  market. This study is neither the pop nor the timing signal: it is the **3-5 year
  aftermarket drift** of the IPOs themselves — the bagholder's leg.

## Shared method citations

- **Newey, W. K. & West, K. D. (1987)** — *A Simple, Positive Semi-definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, **Econometrica
  55(3), 703-708**. The HAC t used on the monthly ETF series (lag 6) and across overlapping
  cohort years (lag 3).
- Desk-wide protocol: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar
  (REAL needs t ≥ 2 on the real tape; literature alone reads WEAK), one documented
  execution lag, one-way costs × NAV with shorts paying borrow, seed-averaged baselines.
