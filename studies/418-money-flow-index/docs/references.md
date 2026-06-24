# References — Study 418 (Money Flow Index)

## The claim & its source

- **Quong, G. & Soudack, A. (1989).** *"Volume-Weighted RSI: Money Flow."* Technical
  Analysis of Stocks & Commodities, 7(3). The original introduction of the Money Flow
  Index as a volume-weighted refinement of Wilder's RSI.
- **Investopedia — "Money Flow Index (MFI)."** The canonical retail statement of the
  folk claim: the MFI "incorporates volume, whereas the RSI does not," and is therefore
  a sharper overbought/oversold gauge. https://www.investopedia.com/terms/m/mfi.asp
- **StockCharts ChartSchool — "Money Flow Index (MFI)."** Standard 14-period
  construction, 20/80 (and 30/70) thresholds, divergence usage.

## The indicator it is compared to

- **Wilder, J. W. (1978).** *New Concepts in Technical Trading Systems.* Trend Research.
  The original Relative Strength Index and Wilder's smoothing — the plain-RSI benchmark
  this study races the MFI against.

## On whether oscillator timing adds value

- **Brock, W., Lakonishok, J. & LeBaron, B. (1992).** *"Simple Technical Trading Rules
  and the Stochastic Properties of Stock Returns."* Journal of Finance, 47(5). The
  classic demonstration that technical rules look good in-sample but largely capture
  drift / fail out-of-sample once benchmarked properly.
- **Sullivan, R., Timmermann, A. & White, H. (1999).** *"Data-Snooping, Technical
  Trading Rule Performance, and the Bootstrap."* Journal of Finance, 54(5). Why a
  positive Sharpe from a screened rule needs a snooping-aware test, not a bare *t*.
- **De Bondt, W. & Thaler, R. (1985).** *"Does the Stock Market Overreact?"* Journal of
  Finance, 40(3). The overreaction premise the contrarian oversold-buy rule leans on.

## Shared method (desk engine)

- **Newey, W. & West, K. (1987).** *"A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."* Econometrica,
  55(3). The HAC *t*-stat used on every daily excess-return book here.
- **Politis, D. & Romano, J. (1994).** *"The Stationary Bootstrap."* JASA, 89(428). The
  block-resampling logic behind our 21-day block bootstrap on the MFI-minus-RSI
  difference and the block-permutation placebo.

## Related desk studies

- [`../../104-bollinger-reversion/`](../../104-bollinger-reversion/) — the same "buy the
  extreme" contrarian logic on Bollinger bands, with the same drift-baseline honesty.
- [`../../301-triple-rsi/`](../../301-triple-rsi/) — a viral RSI recipe torn down with
  the same one-sample-vs-benchmark protocol.
- [`../../107-stochastic-oscillator/`](../../107-stochastic-oscillator/) — another bounded
  oscillator turned into a long/flat timer and raced against buy-and-hold.
