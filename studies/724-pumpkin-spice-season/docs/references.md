# References & literature map — Study 724 (Pumpkin-Spice-Season)

## The claim and its cultural source

- **Starbucks Corporation — Pumpkin Spice Latte (PSL).** Introduced 2003; now Starbucks' best-selling
  seasonal beverage, with the annual launch pulled progressively earlier (late **August** in recent
  years) to capture demand. Starbucks' own press releases date each year's launch and market it as the
  start of "fall." The steelman this study tears down: the PSL launch kicks off an **Aug–Nov**
  "pumpkin-spice season" in which SBUX *beats the market*. See Starbucks Stories & News (annual "Pumpkin
  Spice Latte returns" announcements) and the trade-press coverage each August (CNBC, Bloomberg,
  QSR Magazine) that frames the launch as a revenue and stock catalyst.
- **"Pumpkin spice economy" business-press pieces** — recurring seasonal features (e.g. Nielsen /
  NielsenIQ pumpkin-category sales round-ups; Forbes/CNBC "how much the PSL is worth to Starbucks"
  explainers) that assert the autumn pumpkin ritual is a material commercial event. These are the
  believers' framing; the study asks whether the *market-relative stock return* carries any of it.

## Calendar effects and the data-mining problem

- **Sullivan, R., Timmermann, A., & White, H. (2001).** *Dangers of Data Mining: The Case of Calendar
  Effects in Stock Returns.* Journal of Econometrics 105(1), 249–286 — the direct warning that calendar
  effects found in exploratory screens rarely survive a data-snooping correction. The 12-window placebo
  here is exactly this discipline: Aug–Nov is one slice among twelve, and a "winning" window chosen
  after the fact is significant by construction.
- **Bouman, S., & Jacobsen, B. (2002).** *The Halloween Indicator, "Sell in May and Go Away."*
  American Economic Review 92(5), 1618–1635 — the canonical Nov–Apr vs May–Oct seasonal; the
  methodological template (and cautionary tale) for splitting the calendar into a "good" and "bad"
  window and testing the mean difference. Our Aug–Nov window overlaps its "bad" half, which is part of
  why a market-relative single-name framing is the right object of study.
- **Lakonishok, J., & Smidt, S. (1988).** *Are Seasonal Anomalies Real? A Ninety-Year Perspective.*
  Review of Financial Studies 1(4), 403–425 — the long-horizon audit of calendar anomalies; most shrink
  or vanish out of sample.

## Survivorship / single-name selection

- **Brown, S. J., Goetzmann, W., Ibbotson, R. G., & Ross, S. A. (1992).** *Survivorship Bias in
  Performance Studies.* Review of Financial Studies 5(4), 553–580 — why testing a *hand-picked
  survivor* (SBUX, one of the great stocks of its era) manufactures apparent edges. Named on the Signal
  axis here: any "buy the beloved brand for the season" result is one survivor away from a mirage.
- **McLean, R. D., & Pontiff, J. (2016).** *Does Academic Research Destroy Stock Return
  Predictability?* Journal of Finance 71(1), 5–32 — post-publication/post-popularization decay; a widely
  told seasonal folklore is exactly the kind of pattern that arbitrages away (or was never there).

## Shared method (the desk engine)

- **Newey, W. K., & West, K. D. (1987).** *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix.* Econometrica 55(3), 703–708 — the HAC standard errors
  behind every per-month and season t-stat.
- **Lo, A. W. (2002).** *The Statistics of Sharpe Ratios.* Financial Analysts Journal 58(4), 36–52 —
  the autocorrelation-robust Sharpe inference used in the rotation race.
- **Politis, D. N., & Romano, J. P. (1994).** *The Stationary Bootstrap.* JASA 89(428), 1303–1313 — the
  block-bootstrap family behind the 95% CI on the season-minus-off spread (12-month blocks to respect
  the annual seasonal structure).

## Data

- **Yahoo! Finance** — Starbucks (SBUX), the S&P 500 ETF (SPY) and the 13-week T-bill (^IRX, the cash
  leg), 1993–2026, **daily** auto-adjusted closes resampled to month-end (400 months). SBUX and SPY are
  **total-return** (dividends & splits reinvested via `auto_adjust`) — labeled accordingly. The object
  of study is the **excess** SBUX − SPY. A robustness leg uses an equal-weight coffee/QSR basket
  (SBUX, MCD, YUM, CMG) starting 2006 (CMG's IPO). The study-local caches live at
  `_cache/pumpkin_spice_season.parquet` and `_cache/pumpkin_spice_basket.parquet` (gitignored).

*Companion studies on the bench: [307 Coffee-Seasonality](../../307-coffee-seasonality/) (the
commodity behind the cup), and the calendar-effect family — the same "is this month special?" question
asked of a different tape.*
