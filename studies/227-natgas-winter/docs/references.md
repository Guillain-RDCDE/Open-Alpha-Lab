# References & literature map — Study 227 (Natgas-Winter)

## The claim and its sources

- **Erb, C. B., & Harvey, C. R. (2006).** *The Strategic and Tactical Value of Commodity Futures.*
  Financial Analysts Journal 62(2), 69–97 — foundational commodities paper documenting roll yield as
  the dominant return component for futures-based vehicles; seasonal patterns in natgas noted.
- **Gorton, G., & Rouwenhorst, K. G. (2006).** *Facts and Fantasies about Commodity Futures.* Financial
  Analysts Journal 62(2), 47–68 — commodity futures diversification and roll-yield economics; contango
  vs backwardation environments. The paper notes that long commodities in contango are a structural
  drag.
- **Siegel, L. B. (2008).** *Alternatives and Liquidity: Will Spending and Capital Calls Eat Your
  "Free Lunch"?* Journal of Portfolio Management — note on commodity ETP roll drag for retail investors.
- **Folk belief** — The "winter natgas spike" is widely cited in retail trading forums and commodity
  newsletters: buy UNG in October, sell in March. The desk treats this as a folklore claim with no
  peer-reviewed backing and tests it on the live UNG history.

## On roll yield drag in commodity ETPs

- **Cheng, I. H., & Xiong, W. (2014).** *Financialization of Commodity Markets.* Annual Review of
  Financial Economics 6, 419–441 — documents how financialization changed commodity futures dynamics,
  including roll-yield regimes.
- **Bhardwaj, G., Gorton, G., & Rouwenhorst, K. G. (2015).** *Facts and Fantasies about Commodity
  Futures Ten Years Later.* NBER Working Paper 21243 — updates the 2006 facts paper; confirms roll-yield
  drag remains the dominant return component for long commodity futures positions.
- **ProShares / United States Commodity Funds (2007-2026).** UNG prospectus disclosures — warn
  explicitly of contango roll drag: "investing in UNG may be significantly different from investing in
  natural gas itself."

## On natural-gas seasonality (fundamental backdrop)

- **EIA (U.S. Energy Information Administration).** *Natural Gas Weekly Update* (various issues) —
  heating demand peaks December–February; injections peak April–October. The fundamental seasonal
  pattern is real; the question is whether it is already priced into the futures curve (we find it is).
- **Hamilton, J. D. (2009).** *Understanding Crude Oil Prices.* Energy Journal 30(2), 179–206 — oil
  and gas price dynamics; storage arbitrage forces seasonal patterns into futures spreads, not spot
  returns.

## Data

- **Yahoo! Finance** — UNG (United States Natural Gas Fund LP), 2007-05 → 2026-05, **daily** closes
  resampled to month-end. UNG began trading April 18, 2007; first complete calendar month is May 2007.
  The study-local cache lives at `_cache/natgas_winter_ung.parquet` (gitignored). The offline synthetic
  world injects a tunable winter premium and a null; the real tape always overrules the synthetic in
  interpretation.

*Companion studies in spirit: [92-easy-money](../../92-easy-money/) (commodity carry), [85-dr-copper](../../85-dr-copper/)
(cross-asset commodities signal). Engine: [`quantlab/`](../../quantlab/).*
