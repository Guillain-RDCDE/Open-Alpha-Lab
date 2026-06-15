# References & literature map — Study 180 (TRIX)

## The claim under test

- **The folk recipe.** Popularised by Jack Hutson in *Technical Analysis of Stocks and
  Commodities* magazine (1983): *"TRIX — a triple-smoothed exponential moving average. The
  triple smoothing filters out short-term cycles and focuses the trader on the major trend.
  When TRIX crosses zero from below, buy; when it crosses from above, sell."*  The recipe is
  widely replicated in trading software (TradingView, MetaTrader, Bloomberg) and retail
  tutorials.  The testable hypothesis: *the zero-line cross of the triple-smoothed EMA
  rate-of-change direction carries enough medium-term momentum to beat a random-direction
  entry, net of costs, on daily bars.*

## Why the steelman is almost coherent — the real effect it leans on

- **Intermediate-term price momentum.**  Jegadeesh & Titman (1993), *Returns to Buying
  Winners and Selling Losers: Implications for Stock Market Efficiency* (Journal of Finance),
  document 3–12 month cross-sectional momentum — the empirical bed TRIX is designed to ride.
  If individual names trend over months, a slow oscillator that confirms direction after triple
  smoothing *should* have a positive expected return on the confirmed side.  The failure is
  in the lag, not in the underlying effect.
- **MA-based trend-following on indices.**  Faber (2007), *A Quantitative Approach to Tactical
  Asset Allocation* (Journal of Wealth Management), shows 10-month SMA timing works on asset
  classes over long horizons.  A triple-smoothed 15-day EMA (warmup ≈ 45 days) sits between
  the very short-term (where mean reversion dominates) and the long-term (where the Faber effect
  operates) — a zone with weak historical signal.
- **Moving-average crossovers as trend detectors.**  Brock, Lakonishok & LeBaron (1992),
  *Simple Technical Trading Rules and the Stochastic Properties of Stock Returns* (Journal of
  Finance), found MA-based rules predictive on early DJIA data.  Park & Irwin (2007), *What Do
  We Know About the Profitability of Technical Analysis?* (Journal of Economic Surveys), catalogue
  the post-publication decay and data-snooping concerns.  Sullivan, Timmermann & White (1999),
  *Data-Snooping, Technical Trading Rule Performance, and the Bootstrap* (Journal of Finance),
  show that MA rules that looked profitable in Brock et al. no longer survived a proper
  multiple-comparison correction — the same failure this study replicates with Bonferroni
  across four TRIX period variants (max |*t*| = 1.27, threshold = 2.57).

## The structural weakness — lag from triple smoothing

- **Exponential smoothing and lag.**  Gardner (1985), *Exponential Smoothing: The State of the
  Art* (Journal of Forecasting), provides the theoretical framework: each EMA introduces a
  delay of approximately ``span / 2`` bars; triple smoothing accumulates three such delays,
  giving TRIX(15) an effective lag of ≈ 22 bars (one calendar month).  By the time TRIX
  confirms a zero-line cross, the trend that triggered it is typically more than halfway
  through its cycle — the information content is stale.
- **Price of information timeliness.**  Lo & MacKinlay (1999), *A Non-Random Walk Down Wall
  Street* (Princeton University Press), Chapter 2, discuss the speed at which information is
  incorporated into prices: at the daily bar level, momentum from individual stocks is largely
  priced within a month.  An indicator that lags by a month on a monthly momentum cycle
  arrives at the party after most of the profit has already been claimed.

## Related desk studies

- **[Study 21 — Fools-Gold](../../21-fools-gold/)** — the daily 50/200 SMA golden cross.
  Same moving-average-crossover family, longer periods, same fundamental verdict.
- **[Study 72 — Loaded-Dice](../../72-loaded-dice/)** — SMA(5/10) crossover at the 5-minute
  scale.  Shows the same coin-flip result applies at intraday resolution; the synthetic control
  confirms the engine detects momentum when it is actually planted.
- **[Study 106 — Supertrend](../../106-supertrend/)** — ATR-envelope trend indicator on daily
  bars.  Found REAL/FRAGILE; represents the high end of what trend-following on daily bars can
  achieve.  TRIX lags further and achieves less.
- **[Study 127 — Williams-R](../../127-williams-r/)** — mean-reversion oscillator on daily bars,
  the *opposite* bet from TRIX; tests the oversold/overbought bounce rather than trend continuation.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.**  Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.summarize`](../trix/strategy.py) and [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Sharpe with robust SE.**  Lo (2002), *The Statistics of Sharpe Ratios* (Financial Analysts
  Journal) — [`quantlab.analytics.sharpe_with_se`](../../../quantlab/analytics.py).
- **Block bootstrap CI.**  Politis & Romano (1994), *The Stationary Bootstrap* (JASA) —
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).
- **Multiple-comparisons correction.**  Bonferroni (1936) — applied over four TRIX period
  variants (9, 15, 21, 30) with adjusted threshold |*t*| ≥ 2.57.
- **Reproducibility stamp.**  [`quantlab/repro.py`](../../../quantlab/repro.py) — the as-of
  freeze and content fingerprint each headline run carries.

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`), ~15 years of daily data across five liquid
  instruments (SPY, QQQ, IWM, AAPL, MSFT).  The long window (3,772 bars per instrument)
  ensures TRIX(15) has ample warmup (~45 bars) and enough independent zero-line cross events
  (~89 per instrument) for meaningful inference.  Cache: `studies/180-trix/_cache/`.
