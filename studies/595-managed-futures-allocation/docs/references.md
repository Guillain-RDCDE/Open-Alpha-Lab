# References — Study 595 (Managed-Futures Sleeve)

## The claim's source

The allocator's pitch — "put 10-20% of a 60/40 into managed futures: near-zero correlation,
crisis alpha" — is the marketing core of the post-2022 managed-futures-ETF wave:

- **iMGP / DBi, *DBMF — Managed Futures Strategy ETF*** — fund literature pitching the
  "diversifier with crisis alpha" allocation case. <https://imgpfunds.com/im-dbi-managed-futures-strategy-etf/>
- **Mount Lucas / KFA, *KMLM — Mount Lucas Managed Futures Index Strategy ETF*.**
  <https://kfafunds.com/kmlm/>
- **AQR, *You Can't Always Trend When You Want* (2018) and the AQR managed-futures allocation
  notes** — the honest version of the sleeve pitch, including the drought caveat.
  <https://www.aqr.com/Insights/Research/White-Papers/You-Cant-Always-Trend-When-You-Want>

## Key papers

- **Moskowitz, T., Ooi, Y.H., Pedersen, L.H. (2012), "Time Series Momentum",** *Journal of
  Financial Economics* 104(2) — the 12-month own-sign construction and inverse-vol sizing the
  replication book copies. <https://doi.org/10.1016/j.jfineco.2011.11.003>
- **Hurst, B., Ooi, Y.H., Pedersen, L.H. (2017), "A Century of Evidence on Trend-Following
  Investing",** *Journal of Portfolio Management* 44(1) — the long-sample case that trend pays
  most in equity drawdowns (the "crisis alpha" evidence).
  <https://doi.org/10.3905/jpm.2017.44.1.015>
- **Fung, W., Hsieh, D. (2001), "The Risk in Hedge Fund Strategies: Theory and Evidence from
  Trend Followers",** *Review of Financial Studies* 14(2) — trend followers as long-straddle
  payoffs (why the sleeve pays in crises). <https://doi.org/10.1093/rfs/14.2.313>
- **Kaminski, K. (2011), "In Search of Crisis Alpha"** — the term "crisis alpha" itself.
  <https://www.cmegroup.com/education/files/in-search-of-crisis-alpha.pdf>
- **Ledoit, O., Wolf, M. (2008), "Robust Performance Hypothesis Testing with the Sharpe
  Ratio",** *Journal of Empirical Finance* 15 — why Sharpe differences need block-bootstrap
  inference (our ΔSharpe CI follows this spirit). <https://doi.org/10.1016/j.jempfin.2008.03.002>

## Desk siblings (dedup guard)

- [**31-trade-winds**](../../31-trade-winds/) — the standalone cross-asset TSMOM book on the
  same 18-futures panel: Signal **Weak** (blend HAC t ≈ 1.6), crisis-alpha third axis
  Confirmed. This study inherits its shared futures cache and does **not** re-litigate the
  standalone premium.
- [**518-time-series-momentum**](../../518-time-series-momentum/) — MOP TSMOM on a modern ETF
  basket: Signal **Weak** (12-mo one-sample t = 1.61, fragile to lookback). Here the unit
  under test is the **portfolio benefit** of a 10-20% sleeve, not the premium.

## Data sources

- **Shared futures panel** — `_cache/trade_winds_futures.parquet` (built by study 31 from
  yfinance continuous front-month contracts, daily returns, 18 contracts, 2000-07 → 2026-06).
- **yfinance** (public, no key) — SPY, VBMFX (Vanguard Total Bond Index, the long-history
  bond leg), DBMF, KMLM total-return closes; ^IRX 13-week T-bill discount yield as the
  risk-free leg. <https://github.com/ranaroussi/yfinance>
- Method citations shared by the desk: Newey-West (1987) HAC errors; Fisher (1921) z-CI for
  correlations; Efron/Künsch moving-block bootstrap.
