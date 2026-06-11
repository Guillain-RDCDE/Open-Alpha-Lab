# References & literature map — Study 47 (Paper-Moon)

## The model and its critique

- **The "Fed Model"** — popularised after a 1997 Federal Reserve *Humphrey-Hawkins* report noted the
  S&P forward earnings yield tracked the 10-year Treasury yield. Never an official Fed position; the
  name stuck.
- **Asness, C. (2003).** *Fight the Fed Model.* Journal of Portfolio Management 30(1) — the definitive
  takedown: the model confuses a **real** earnings yield with a **nominal** bond yield, so its apparent
  signal is the inflation co-movement, not value. The intellectual spine of this study.
- **Modigliani, F., & Cohn, R. (1979).** *Inflation, Rational Valuation and the Market.* Financial
  Analysts Journal — the original "inflation illusion" in equity valuation.
- **Vendor entry** — [paperswithbacktest / awesome-systematic-trading](https://github.com/paperswithbacktest/awesome-systematic-trading),
  *"FED Model"* (listed Sharpe `0.369`), with a QuantConnect implementation. Backlog triage:
  [`docs/pwb_strategies_inventory.md`](../../../docs/pwb_strategies_inventory.md).

## On valuation (E/P / CAPE) as the thing that actually works

- **Campbell, J., & Shiller, R. (1988, 1998).** *Valuation Ratios and the Long-Run Stock Market
  Outlook* — earnings yield / CAPE has mild long-horizon forecasting power; the *level*, not the
  comparison to bonds, is what carries it.

## Data

- **Robert Shiller's** monthly U.S. stock data (price, dividend, earnings, CPI, 10-year yield), 1871–
  present, via the key-free **datahub.io** mirror (`datasets/s-and-p-500`). Total return ≈ price change
  + dividend/12; E/P = trailing earnings / price. The offline synthetic world makes only E/P
  informative (the bond yield independent), so the "bond term is inert" result is provable offline.

*The debunk bench: [40 Paper-Tiger](../../40-paper-tiger/) … [46 Bargain-Bin](../../46-bargain-bin/) —
this one is the "the logic itself is broken" entry.*
