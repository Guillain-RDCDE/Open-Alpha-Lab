# References & literature map — Study 60 (Long-Shot)

## The effect and its source

- **Fuertes, A.-M., Miffre, J., & Fernandez-Perez, A. (2015).** *Commodity Strategies Based on
  Momentum, Term Structure, and Idiosyncratic Volatility.* Journal of Futures Markets — and the
  skewness strand: low-skewness commodities out-earn high-skewness ones (lottery preference).
- **Bali, T., Cakici, N., & Whitelaw, R. (2011).** *Maxing Out* — the equity lottery/MAX effect
  ([Study 53 Jackpot](../../53-jackpot/)); skewness is the commodity analogue.
- **Vendor entry** — *"Skewness Effect in Commodities"* on
  [paperswithbacktest](https://github.com/paperswithbacktest/awesome-systematic-trading) (listed Sharpe
  `0.482`); backlog: [`docs/pwb_strategies_inventory.md`](../../../docs/pwb_strategies_inventory.md).

## Why a small ETF basket under-powers it

- The literature uses ~20–30 commodity *futures*; an ETF basket of 14 (with quintile/tercile sub-baskets
  of ~5) starves the cross-sectional t-stat of breadth.
- **Open-Alpha-Lab [Study 35 Contango](../../35-contango/)** — commodity ETFs carry roll-yield drag that
  erodes long-short commodity strategies.

## Data

- **Yahoo! Finance** — 14 commodity ETFs (metals, energy, agriculture), daily, 2009–2026. The offline
  synthetic panel makes high-skew assets underperform (positive jumps + drift penalty) and a null, so
  the machinery is provable offline.

*The commodity analogue of the equity lottery studies [53 Jackpot](../../53-jackpot/) /
[54 Static](../../54-static/) — but here the effect points the right way.*
