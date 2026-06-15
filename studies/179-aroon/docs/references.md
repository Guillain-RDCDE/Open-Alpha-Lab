# References & literature map -- Study 179 (Aroon)

## The indicator's origin

- **Chande, T. S. (1995).**  *The New Technical Trader: Boost Your Profit by Plugging into
  the Latest Indicators.*  Wiley.  The original Aroon paper and name ("Aroon" from Sanskrit
  for "dawn's early light").  Chande designed it specifically to detect the *beginning* of a
  new trend: Aroon-Up near 100 means the most recent high was near the current bar; Aroon-Down
  near 100 means the most recent low was recent -- together they signal a strong developing
  trend.

- **Chande, T. S. & Kroll, S. (1994).**  *The New Technical Trader.*  Wiley.  Earlier
  precursor work; the "Aroon" name and indicator appeared in *Technical Analysis of Stocks &
  Commodities*, June 1995.

## Why it might work -- the theoretical basis

- **Momentum in equity returns.**  Jegadeesh & Titman (1993), *Returns to Buying Winners and
  Selling Losers: Implications for Stock Market Efficiency* (Journal of Finance, 48(1), 65-91)
  -- the landmark paper on cross-sectional momentum.  Aroon is a time-series (within-instrument)
  momentum indicator: when the highest high in the past 25 days is recent, recent price action
  has been upward -- a candidate predictor of short-horizon continuation.

- **Time-series momentum.**  Moskowitz, Ooi & Pedersen (2012), *Time Series Momentum* (Journal
  of Financial Economics, 104(2), 228-250) -- documents strong directional persistence across
  asset classes at monthly horizons.  Aroon's 25-day lookback is shorter, operating in the zone
  where such effects are weaker and noisier.

- **Trend following at daily frequency.**  Hurst, Ooi & Pedersen (2017), *A Century of Evidence
  on Trend-Following Investing* (Journal of Portfolio Management, 44(1), 15-29) -- broad
  evidence that simple trend signals carry a risk premium across long horizons; the literature
  is mixed at the shorter (5-25 bar) window this study focuses on.

## Why it likely fails -- the counter-evidence

- **Technical analysis out-of-sample.**  Park & Irwin (2007), *What Do We Know About the
  Profitability of Technical Analysis?* (Journal of Economic Surveys, 21(4), 786-826) -- a
  comprehensive review showing most technical-indicator profits evaporate after data-snooping
  corrections, transaction costs, and out-of-sample testing.

- **Look-back bias in indicator design.**  Sullivan, Timmermann & White (1999), *Data-Snooping,
  Technical Trading Rule Performance, and the Bootstrap* (Journal of Finance, 54(5), 1647-1691)
  -- demonstrates how cherry-picked parameters (here: period=25, hold=5) inflate apparent
  backtested t-stats; our desk controls for this with an explicit random-direction baseline.

- **Bull-market survivorship and regime dependency.**  The 2011-2026 window used here is
  predominantly a US-equity bull market.  A long-biased trend-following signal (like Aroon
  crossovers in practice, where shorts are noise) will look strong in such a window;
  Dichev & Zhong (2022), *Long-run return reversal: evidence from international market indices*
  (Journal of International Financial Markets, Institutions & Money) highlight how window
  selection distorts verdict.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.**  Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica, 55(3),
  703-708) -- [`strategy.summarize`](../aroon/strategy.py).

- **Block bootstrap CI.**  Politis & Romano (1994), *The Stationary Bootstrap* (Journal of the
  American Statistical Association, 89(428), 1303-1313) --
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).

- **Multiple-comparisons correction.**  Bonferroni (1936) -- with 3 tickers x 5 hold periods
  = 15 simultaneous tests, the 5%-level threshold rises to |*t*| > 2.57; the pooled t=3.25
  clears this bar but individual instrument results (SPY t=0.74, QQQ t=1.48) do not.

## Data sources

- **Yahoo! Finance daily bars** (via `yfinance`), 15-year window ending 2026-06-15, across
  three US-equity ETFs: SPY (large-cap), QQQ (Nasdaq-100), IWM (small-cap).  All content
  fingerprinted in [`docs/results.md`](results.md).

## Related desk studies

- **[Study 21 -- Fools-Gold](../../21-fools-gold/)**: the 50/200 golden cross on the same daily
  bars -- same moving-average family, applied to much slower trend detection.
- **[Study 72 -- Loaded-Dice](../../72-loaded-dice/)**: the SMA(5/10) crossover on 5-minute bars --
  the intraday version of the crossover family.  Verdict: None.  Daily bars (here) give a very
  different answer.
- **[Study 78 -- Crossed-Wires](../../78-crossed-wires/)**: another daily MA crossover -- direct
  comparison for the family's behaviour at this timeframe.
- **[Study 127 -- Williams-R](../../127-williams-r/)**: a 14-period oscillator on daily bars --
  same infrastructure, complementary oscillator family (mean-reversion vs trend-following).
- **[Study 106 -- Supertrend](../../106-supertrend/)**: ATR-based trend filter on daily data --
  another competitor in the short-horizon trend-detection space.
