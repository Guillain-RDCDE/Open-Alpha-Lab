# References & literature map — Study 41 (Hangover)

## The claim and its source

- **Hirsch, Y. (1972, and the annual *Stock Trader's Almanac*).** The original "January Barometer":
  *as the S&P goes in January, so goes the year.* The omen the desk replicates and scores.
- **Vendor entry** — [paperswithbacktest / awesome-systematic-trading](https://github.com/paperswithbacktest/awesome-systematic-trading),
  *"January Barometer"* (listed Sharpe `0.365`), with a QuantConnect implementation. This study is the
  desk's independent test. Backlog triage: [`docs/pwb_strategies_inventory.md`](../../../docs/pwb_strategies_inventory.md).

## Why the base rate is the right benchmark

- **Cooper, M., McConnell, J., & Ovtchinnikov, A. (2006).** *The Other January Effect.* Journal of
  Financial Economics 82(2) — finds predictive content, but the debate hinges on the unconditional
  drift; we make that explicit by scoring against the base rate.
- **Stivers, C., Sun, L., & Sun, Y. (2009).** *The Other January Effect: International, style, and
  subperiod evidence* — the effect is concentrated and sub-period-dependent (our decay split).
- **McLean, R. D., & Pontiff, J. (2016).** *Does Academic Research Destroy Stock Return
  Predictability?* Journal of Finance 71(1) — the post-publication-decay frame.

## Data

- **Yahoo! Finance** — S&P 500 (^GSPC), daily auto-adjusted close resampled to month-end total return,
  1950–2025. (Daily ^GSPC reaches 1927; the monthly endpoint is truncated to ~1986, so we resample
  from daily.) The offline synthetic year-world exercises the machinery without the network.

*Sibling in spirit: [40 Paper-Tiger](../../40-paper-tiger/) — another "beats the market" headline that
dissolves against the right benchmark.*
