# References — Study 185 (Chande-Momentum)

## Primary source

**Chande, T. S. & Kroll, S. (1994).** *The New Technical Trader: Boost Your Profit by
Plugging into the Latest Indicators.* Wiley.  The original CMO definition appears in
Chapter 3 ("Momentum Indicators").  Chande defines CMO as 100 × (Su − Sd) / (Su + Sd)
over a look-back period; the key innovation vs RSI is the denominator includes *all*
movement, making the oscillator symmetric.

**Chande, T. S. (1997).** *Beyond Technical Analysis: How to Develop and Implement a
Winning Trading System.* Wiley.  Follow-up treatment discussing the CMO's use as a trend
filter (high |CMO| = trend; low |CMO| = chop), overbought/oversold framings, and
combination with adaptive moving averages (VIDYA).

## Independent replications and survey articles

**Kirkpatrick, C. D. & Dahlquist, J. R. (2011).** *Technical Analysis: The Complete
Resource for Financial Market Technicians* (2nd ed.). FT Press.  Chapter 12 surveys
momentum oscillators; CMO is discussed alongside RSI and stochastics as a member of the
normalised-momentum family.

**Pring, M. J. (2002).** *Technical Analysis Explained* (4th ed.). McGraw-Hill.  Pring
benchmarks CMO against RSI and %K on historical US equity data.  No rigorous inference
is applied, and the comparison does not control for the random-direction baseline.

## Related academic work on oscillator-based signals

**Brock, W., Lakonishok, J. & LeBaron, B. (1992).** "Simple technical trading rules and
the stochastic properties of stock returns." *Journal of Finance*, 47(5), 1731–1764.
Seminal paper testing moving-average and oscillator signals on DJIA; finds positive
in-sample returns that diminish significantly after transaction costs.

**Ready, M. J. (2002).** "Profits from technical trading rules." *Financial Management*,
31(3), 43–61.  Out-of-sample replication of Brock et al. (1992); finds that most
moving-average and oscillator profits disappear in the post-1987 period when controls for
transaction costs and risk are applied.

**Menkhoff, L. & Taylor, M. P. (2007).** "The obstinate passion of foreign exchange
professionals: Technical analysis." *Journal of Economic Literature*, 45(4), 936–972.
Cross-market survey finding that oscillator signals are most commonly used by FX
practitioners for short-horizon timing, but that evidence for consistent profitability
is weak.

## Methodological references used in this study

**Newey, W. K. & West, K. D. (1987).** "A simple, positive semi-definite,
heteroskedasticity and autocorrelation consistent covariance matrix." *Econometrica*,
55(3), 703–708.  The HAC estimator used for the t-statistics throughout.

**White, H. (2000).** "A reality check for data snooping." *Econometrica*, 68(5),
1097–1126.  Motivation for using a random-direction baseline as the honest null: the
coin flip correctly accounts for the multiple-comparisons hazard inherent in searching
over oscillator thresholds and hold periods.

## House context

This study sits in the same family as:
- **Study 72 (Loaded-Dice)** — SMA(5/10) intraday crossover vs random direction.
- **Study 106 (Supertrend)** — ATR-band trend-following vs random direction.
- **Study 127 (Williams-R)** — %R overbought/oversold vs random direction (same framing
  as CMO Framing 1, different normalisation).
- **Study 78 (Crossed-Wires)** — MACD zero-cross vs random direction (same framing as
  CMO Framing 2).

All four reach the same verdict: technical oscillators in this family do not clear the
inference bar on daily data over 10-year windows.
