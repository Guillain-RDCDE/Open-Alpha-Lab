# References & literature map — Study 56 (Tide-Table)

## The signal and its source

- **Campbell, J., & Shiller, R. (1988, 1998).** *Stock Prices, Earnings, and Expected Dividends* /
  *Valuation Ratios and the Long-Run Stock Market Outlook* — CAPE and long-horizon return predictability.
- **Shiller, R. (2000).** *Irrational Exuberance* — CAPE popularised; the data series we use.
- **Vendor entry** — [paperswithbacktest / awesome-systematic-trading](https://github.com/paperswithbacktest/awesome-systematic-trading),
  *"Value Factor – CAPE Effect within Countries"* (listed Sharpe `0.351`) — the cross-country cousin;
  this study tests CAPE's *time-series* forecasting on the US. Backlog:
  [`docs/pwb_strategies_inventory.md`](../../../docs/pwb_strategies_inventory.md).

## On CAPE as a forecaster, not a timer

- **Asness, C., Ilmanen, A., & Maloney, T. (2017).** *Market Timing: Sin a Little.* Journal of Portfolio
  Management — valuation timing works only weakly and slowly; the horizon caveat at the heart of this study.
- **Welch, I., & Goyal, A. (2008).** *A Comprehensive Look at… Equity Premium Prediction* — most
  predictors fail out of sample; CAPE's long-horizon power is one of the few that survives.

## Data

- **Robert Shiller's** monthly U.S. series (price, dividend, earnings, CPI, **PE10/CAPE**), 1871–present,
  via the key-free **datahub.io** mirror. Forward real total return reinvests real dividends. The offline
  synthetic world makes CAPE forecast forward returns negatively (and a null) so the result is provable offline.

*The honest counterpart to [47 Paper-Moon](../../47-paper-moon/): valuation (E/P, CAPE) carries the
signal; the Fed Model's bond-yield comparison does not.*
