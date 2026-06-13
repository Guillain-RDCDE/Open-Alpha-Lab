# References & literature map -- Study 104 (Bollinger-Reversion)

## The claim under test

- **The folk recipe.**  Popularised in Bollinger (2001) *Bollinger on Bollinger Bands*
  (McGraw-Hill) and endlessly retold in trading forums: *"When the daily close pierces
  the lower Bollinger Band (20-SMA minus 2 sigma), buy; price always returns to the
  middle band or beyond -- that's the free lunch."*  A complementary variant claims the
  opposite: *"A close above the upper band is a breakout -- buy the momentum."*  Both
  cannot be simultaneously attributable to the bands; we steelman them as the sharpest
  testable forms -- (a) the lower-band pierce is a *statistically* better entry than a
  random day, and (b) the upper-band pierce is a worse one -- and measure both against a
  random-day baseline over 21 years of US equity data.

## The real effect the claim leans on -- short-term mean reversion

- **Mean reversion in equity prices.**  De Bondt & Thaler (1985), *Does the Stock Market
  Overreact?* (Journal of Finance) -- multi-year contrarian effect: extreme losers
  outperform over 3-5 years.  Our daily/20-day horizon is a much shorter version of the
  same idea.
- **Short-term reversal at weekly horizons.**  Jegadeesh (1990), *Evidence of Predictable
  Behavior of Security Returns* (Journal of Finance), and Lehmann (1990), *Fads, Martingales,
  and Market Efficiency* (Quarterly Journal of Economics) -- weekly return reversals exist
  at short horizons, driven partly by microstructure and partly by over-reaction.
- **Bollinger Bands as a volatility envelope.**  Keltner (1960) *How to Make Money in
  Commodities* and Bollinger (2001) -- the original rationale is that 95%+ of prices lie
  within ±2 sigma of a rolling mean by construction (for a normal distribution), so a
  pierce is a local extreme.  The study shows this mechanical property does not by itself
  generate an incremental trading edge once the market trend is accounted for.

## Why the steelman is partially coherent -- but the attribution is wrong

- **"Price always returns" is (mostly) drift.**  In a rising market (S&P 500 +10%/yr
  2005-2026), almost any buy-and-hold for 20 days earns a positive return.  Our
  random-day control earns +141 bps/20 days at HAC t = +6.20 -- comparable to the lower-band
  entry's +193 bps.  The incremental signal from the band (delta ~ +52 bps, t ~ +0.63) is
  real but not statistically robust.
- **The contradiction: both band extremes work.**  In a trending market the lower-band
  entry (buy the dip) AND the upper-band entry (buy the breakout) earn positive returns,
  but for the same reason -- they're both just buying a rising market.  Brock, Lakonishok
  & LeBaron (1992), *Simple Technical Trading Rules and the Stochastic Properties of Stock
  Returns* (Journal of Finance), documented similar survivorship patterns for moving-average
  rules.
- **Out-of-sample decay.**  Sullivan, Timmermann & White (1999), *Data-Snooping, Technical
  Trading Rule Performance, and the Bootstrap* (Journal of Finance), and Park & Irwin
  (2007), *What Do We Know About the Profitability of Technical Analysis?* (Journal of
  Economic Surveys) -- technical-rule backtests on long bull markets routinely show high
  t-stats that decay or reverse when tested out-of-sample or in bear markets.

## The two traps this study exposes

- **Survivor drift inflates every "buy the dip" number.**  Testing on 2005-2026 includes
  the 2009-2026 bull run; the 2007-2009 crisis is present but dwarfed by the recovery.
  A strategy that is really just "buy and hold" will look prescient here.  The correct
  inference bar is: does the band pierce add something the random day does not?  Our
  delta (t ~ +0.63) says: barely.
- **Win-rate inflated by long bull market.**  The 64.6% win-rate (price reaches SMA within
  20 days) versus 61.5% for random sounds like a meaningful difference but is merely
  3 percentage points of an already-high base rate.  Taleb (2004), *Fooled by Randomness*,
  and the gambler's-ruin literature (Feller, *An Introduction to Probability Theory*,
  Vol. 1) caution against treating a high win-rate as evidence of skill when the base
  rate is driven by market regime.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.**  Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) --
  [`strategy.summarize`](../bollinger_reversion/strategy.py).
- **Bollinger Bands definition.**  Bollinger (2001) -- SMA(20) +/- 2 * rolling_std(20,
  ddof=1), exactly as implemented in [`strategy.bollinger_bands`](../bollinger_reversion/strategy.py).
- **Average true range as risk unit.**  Wilder (1978), *New Concepts in Technical Trading
  Systems* -- used in the desk's barrier engine; here we use a fixed 20-day horizon
  instead, which is horizon-fair for the random-day comparison.
- **Forward-return study design.**  The "enter at next open, exit after N days" framework
  follows Jegadeesh & Titman (1993), *Returns to Buying Winners and Selling Losers*
  (Journal of Finance), extended to single-name vs random baselines.
- **Reproducibility stamp.**  [`quantlab/repro.py`](../../../quantlab/repro.py) -- the
  as-of freeze and content fingerprint each headline run carries.

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`), adjusted close (auto_adjust=True), back
  to 2005-01-03 for six US equity instruments: SPY, QQQ, AAPL, MSFT, JPM, XLE.  Daily
  history goes back 20+ years, giving a meaningful power budget.  The offline reproducible
  core and the test-suite run on the deterministic
  [`data.synthetic_daily`](../bollinger_reversion/data.py) generator, never the network.

## Related desk studies

- **[Study 21 -- Fools-Gold](../../21-fools-gold/)**: the daily 50/200 golden cross --
  the same "the trend is your friend" family; also works in a bull market and dies when
  tested against a buy-and-hold baseline.
- **[Study 72 -- Loaded-Dice](../../72-loaded-dice/)**: the 5-minute SMA crossover scalp
  -- the intraday cousin of this family; found to be a fair coin even on the trending
  intraday tape.
- **[Study 86 -- Tail-Radar](../../86-tail-radar/)**: volatility-signal entries using a
  vol-index threshold; shares the "band as extreme detector" logic.
- **[Study 78 -- Crossed-Wires](../../78-crossed-wires/)**: another MA/indicator rule
  study that isolates the mechanical vs informative components of a technical signal.
