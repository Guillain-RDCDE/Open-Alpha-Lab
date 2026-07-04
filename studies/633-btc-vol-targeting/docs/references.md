# References — Study 633 (BTC Vol Targeting)

## The claim's source

Constant-volatility ("vol targeting") overlays on Bitcoin are a staple of crypto asset-management
pitches and quant-twitter folklore: *"target 30% vol and you keep the Bitcoin ride while cutting
the −80% drawdowns in half."* The pitch is a direct port of the institutional vol-targeting
literature to a single ultra-volatile asset. This study tests that ported claim on the tape.

## Key papers

- **Moreira, A. & Muir, T. (2017)** — *Volatility-Managed Portfolios*, **Journal of Finance**
  72(4), 1611–1644. The canonical result: scaling exposure by inverse realized variance raises
  Sharpe ratios and earns alpha on equity factors, because variance is forecastable while the
  conditional mean is not. <https://doi.org/10.1111/jofi.12513>
- **Harvey, C. R., Hoyle, E., Korgaonkar, R., Rattray, S., Sargaison, M. & van Hemert, O.
  (2018)** — *The Impact of Volatility Targeting*, **Journal of Portfolio Management** 45(1),
  14–33. Vol targeting cuts left tails and smooths the path for risk assets; the Sharpe
  improvement concentrates in assets with a strong leverage effect.
  <https://doi.org/10.3905/jpm.2018.45.1.014>
- **Dreyer, A. & Hubrich, S. (2019)** — *Tail-Risk Mitigation with Managed Volatility
  Strategies*, **Journal of Investment Strategies** 8(1). Managed-vol overlays as drawdown
  control rather than return enhancement — exactly the split this study finds on BTC.
  <https://doi.org/10.21314/JOIS.2019.105>
- **Bollerslev, T. (1986)** — *Generalized Autoregressive Conditional Heteroskedasticity*,
  **Journal of Econometrics** 31(3), 307–327. Why trailing realized vol forecasts tomorrow's
  vol at all (volatility clustering) — the raw material the overlay feeds on.
- **Newey, W. K. & West, K. D. (1987)** — *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, **Econometrica** 55(3),
  703–708. The HAC t-statistics used throughout.

## Named siblings on this desk (the dedup guard)

- [210-crypto-trend](../../210-crypto-trend/) — **SMA timing** on BTC (in-or-out on the 200-day
  moving average; graded Real/Fragile). This study is the **vol-SIZING overlay**: never a
  directional signal, only *how much* BTC to hold, scaled continuously by trailing realized vol.
- [591-vol-managed-portfolio](../../591-vol-managed-portfolio/) — the Moreira-Muir **1/RV
  scaling on equity ETFs** (monthly, SPY/QQQ/EFA/IWM; graded Mixed/Fragile). Same family of
  rule, different asset class and rebalance frequency; study 633 asks whether the recipe ports
  to a single asset with 66% vol and −83% drawdowns.

*(Inter-study links use `../NNN-slug/` per house style.)*

## Data sources

- **BTC-USD daily closes** — Yahoo! Finance via `yfinance` (public, no key),
  <https://finance.yahoo.com/quote/BTC-USD/>. 2014-09-17 → 2026-06-30, cached under
  [`_cache/btc_usd.csv`](../_cache/btc_usd.csv). Price-only equals total-return for BTC (no
  distributions).

## Shared method citations

- Desk house style, inference bar and shared protocol: [`METHODOLOGY.md`](../../../METHODOLOGY.md).
- Reproducibility stamp (as-of + fingerprint): [`quantlab/repro.py`](../../../quantlab/repro.py).
