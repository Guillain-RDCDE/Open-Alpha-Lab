# References & literature map — Study 307 (Coffee-Seasonality)

## The claim and its agronomy

- **Coffee-trade folklore / brokerage seasonality charts** — Arabica is a Southern-Hemisphere crop
  (Brazil ~35–40% of world output), so the "buy ahead of the Brazilian frost season (Jun–Aug winter),
  sell into the May–Sep harvest" calendar is a perennial commodity-desk talking point. It is the
  steelman this study tears down. See e.g. seasonal-pattern services (MRCI, Seasonalgo) that publish
  KC=F monthly seasonality composites.
- **U.S. Department of Agriculture (USDA), Foreign Agricultural Service** — *Coffee: World Markets and
  Trade* (semi-annual) — documents the Brazilian harvest calendar (main crop May–September) and the
  frost-risk window in the southern-hemisphere winter; the supply backdrop the calendar story rests on.

## Coffee-price dynamics and weather shocks

- **Bastian, M., Bekkerman, A., et al.** and the broader agricultural-economics literature on coffee:
  frost and drought events (notably the catastrophic Brazilian frosts of 1975, 1994, and the 2021
  frost) drive most of the variance in Arabica prices — i.e. the price action is **event-driven, not
  calendar-driven**, which is precisely why a fixed-month rule fails.
- **Deaton, A., & Laroque, G. (1992).** *On the Behaviour of Commodity Prices.* Review of Economic
  Studies 59(1), 1–23 — the canonical storage model: commodity prices are dominated by stockout-driven
  spikes (fat right tails), not smooth seasonal cycles, so seasonal means are swamped by tail events.
- **Gorton, G., & Rouwenhorst, K. G. (2006).** *Facts and Fantasies about Commodity Futures.*
  Financial Analysts Journal 62(2), 47–68 — seasonal variation exists in many commodity futures, but
  the softs (coffee, cocoa, sugar) are among the noisiest, with weather risk dominating.

## On multiple testing and seasonality data-mining

- **Sullivan, R., Timmermann, A., & White, H. (2001).** *Dangers of Data Mining: The Case of Calendar
  Effects in Stock Returns.* Journal of Econometrics 105(1), 249–286 — the direct warning that calendar
  effects found in exploratory screens rarely survive; Bonferroni / Reality-Check correction is the
  minimum due diligence (here: 12 monthly tests ⇒ effective |t| ≈ 3).
- **McLean, R. D., & Pontiff, J. (2016).** *Does Academic Research Destroy Stock Return
  Predictability?* Journal of Finance 71(1), 5–32 — post-publication decay applies to seasonal patterns
  as much as to factor premia.
- **Lo, A. W. (2002).** *The Statistics of Sharpe Ratios.* Financial Analysts Journal 58(4), 36–52, and
  **Newey, W. K., & West, K. D. (1987).** *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix.* Econometrica 55(3), 703–708 — the HAC inference and
  autocorrelation-robust Sharpe SE used throughout the desk.
- **Politis, D. N., & Romano, J. P. (1994).** *The Stationary Bootstrap.* JASA 89(428), 1303–1313 — the
  block-bootstrap family behind the 95% CI on the frost-minus-harvest spread (resampling 12-month blocks
  to respect the annual seasonal structure).

## Data

- **Yahoo! Finance** — Arabica coffee front future (KC=F, ICE) and the 13-week T-bill (^IRX, the cash
  leg), 2000–2026, **daily** closes resampled to month-end (316 months). KC=F is a price-only,
  roll-naive front contract — labeled accordingly; it is **not** a total-return roll index. The
  study-local cache lives at `_cache/coffee_seasonality.parquet` (gitignored).

*Companion studies on the bench: [226 Crude-Seasonality](../../226-crude-seasonality/) (driving-season
oil calendar), [133 Crypto-Seasonality](../../133-crypto-seasonality/), [247 Bond-Seasonality](../../247-bond-seasonality/),
and [223 Same-Month Seasonality](../../223-same-month-seasonality/) — the calendar-effect family.*
