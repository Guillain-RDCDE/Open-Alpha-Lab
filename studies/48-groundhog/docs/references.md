# References & literature map — Study 48 (Groundhog)

## The effect and its source

- **Heston, S., & Sadka, R. (2008).** *Seasonality in the Cross-Section of Stock Returns.* Journal of
  Financial Economics 87(2), 418–445 — the foundational result: a stock's same-calendar-month history
  forecasts its future same-month return, and the effect is specific to the same month.
- **Heston, S., & Sadka, R. (2010).** *Seasonality in the cross-section of stock returns: the
  international evidence* — replication out of the US.
- **Vendor entry** — [paperswithbacktest / awesome-systematic-trading](https://github.com/paperswithbacktest/awesome-systematic-trading),
  *"12 Month Cycle in Cross-Section of Stocks Returns"* (listed Sharpe `0.340`), with a QuantConnect
  implementation. Backlog triage: [`docs/pwb_strategies_inventory.md`](../../../docs/pwb_strategies_inventory.md).

## On whether it's tradable / what drives it

- **Keloharju, M., Linnainmaa, J., & Nyberg, P. (2016).** *Return Seasonalities.* Journal of Finance —
  seasonalities are pervasive and tied to common factors; capacity and turnover are the practical limits.
- **Novy-Marx, R. (2014).** *Predicting anomaly performance with politics, the weather…* — a caution on
  over-reading seasonal regularities; our same-month-vs-other-month control is the guard against that.

## Data

- **Yahoo! Finance** — monthly total returns for current S&P 500 members with ≥20 years of history.
  Honest caveats: the universe is **survivorship-biased twice over** (current membership × the
  long-history filter) and large-cap — `groundhog/data.py` raises a `SurvivorshipBiasError` unless the
  caller opts in with `allow_survivorship_bias=True`, and the Signal verdict carries the magnitude
  caveat. The effect is documented across the size spectrum and is typically stronger in smaller names
  — the untradable end. The offline synthetic panel injects a fixed (stock × calendar-month) bias (and
  a null) so the same-month-vs-control logic is provable offline.

*The one `REAL`/`CONFIRMED` of the recent bench — contrast with the mirages
[44 Growth-Spurt](../../44-growth-spurt/), [45 Vanishing-Act](../../45-vanishing-act/),
[47 Paper-Moon](../../47-paper-moon/).*
