# References & literature map — Study 183 (Fisher-Transform)

## The claim under test

- **Ehlers (2002).** John F. Ehlers, *Cybernetic Analysis for Stocks and Futures: Cutting-Edge
  DSP Technology to Improve Your Trading* (Wiley, 2004); the Fisher Transform indicator
  originally appeared in his 2002 *Stocks & Commodities* article. Ehlers argues that mapping
  the normalised price midpoint through the atanh function produces a "more Gaussian" output
  where "turning points are sharper and easier to see." The folk rule: buy when the Fisher line
  crosses above its trigger (the prior bar's Fisher value); sell/short when it crosses below.
  We steelman it as the sharpest testable version: *the Fisher/Trigger crossover carries
  directional information that beats a random-entry control, net of costs, on daily bars.*

## Why the transform cannot add information — the monotonicity argument

- **Monotone functions and information content.** A strictly monotone function $f$ applied to a
  scalar $x$ preserves the ranking of observations: $x_1 < x_2 \Leftrightarrow f(x_1) < f(x_2)$.
  The crossover signal depends only on the *sign* of $(F_t - F_{t-1})$, which equals the sign of
  $(x_t - x_{t-1})$ wherever $f$ is monotone. The Fisher transform $\mathrm{atanh}(\cdot)$ is
  strictly monotone on $(-1, +1)$, so the Fisher crossover is **logically equivalent** to the
  raw-normalised-price crossover — confirmed empirically at 100% coincidence on every tested
  tape. See *Cover & Thomas (2006), Elements of Information Theory* (Wiley) for the general
  principle: invertible transformations of a sufficient statistic cannot increase information.
- **The atanh mapping.** The Fisher Transform is $F = 0.5 \ln\!\bigl(\tfrac{1+x}{1-x}\bigr) =
  \mathrm{atanh}(x)$ where $x \in (-1, 1)$ is the close's normalised position within the rolling
  high-low range. The function maps $(-1,1) \to (-\infty, +\infty)$ monotonically. It is designed
  to stretch the tails of the distribution toward Gaussian, but the *crossover logic* only uses the
  *sign of the difference*, which is invariant to any monotone transformation.

## The signal structure — what the crossover actually detects

- **Momentum vs mean-reversion.** The Fisher/Trigger crossover fires when the normalised price
  midpoint shifts from one side to the other of its one-bar-lagged position. This is a
  *momentum* (trend-continuation) signal: it goes long when the normalised price is rising, short
  when it is falling. The synthetic positive control (Study 183) confirms: the crossover beats a
  coin when *momentum* (positive AR(1)) is planted, and loses to the coin when *mean-reversion*
  (negative AR(1)) is planted. This is the opposite of the "Gaussian turning points" framing in
  the Ehlers literature.
- **Weak-form efficiency and daily returns.** Fama (1970), *Efficient Capital Markets* (Journal
  of Finance) — daily prices are close to a martingale at short horizons. Lo & MacKinlay (1988),
  *Stock Market Prices Do Not Follow Random Walks* (Review of Financial Studies) — there is *some*
  weekly autocorrelation in small stocks, but the effect is small and largely arbitraged away.
  Neither supports a reliable daily Fisher-crossover edge on liquid large-cap ETFs and stocks.
- **Technical trading rules: out-of-sample performance.** Brock, Lakonishok & LeBaron (1992),
  *Simple Technical Trading Rules and the Stochastic Properties of Stock Returns* (Journal of
  Finance) — found in-sample evidence for MA rules; Park & Irwin (2007), *What Do We Know About
  the Profitability of Technical Analysis?* (Journal of Economic Surveys) — document substantial
  out-of-sample deterioration, data-snooping bias, and cost sensitivity. The Fisher crossover is
  in the same family.
- **Short-term momentum on daily bars.** Jegadeesh (1990), *Evidence of Predictable Behavior of
  Security Returns* (Journal of Finance) — documents 1-month *reversal*, not continuation, for
  individual stocks. Jegadeesh & Titman (1993), *Returns to Buying Winners and Selling Losers*
  (Journal of Finance) — momentum exists at 3-12 month horizons, not 1-5 day. The Fisher
  crossover's 1-day result (-4.72 bps, *t* = -2.29) is consistent with the short-horizon
  reversal literature — the signal fires *with* a day's move and the next day reverses.

## The high-frequency version of the same problem

- **5-minute SMA crossover (Study 72 — Loaded-Dice).** The desk's own daily-bar equivalent
  shows the Fisher Transform on a daily tape behaves identically to a moving-average cross:
  it is a fair die on a martingale, finds a momentum signal when momentum exists in the
  synthetic tape, and finds nothing on the real tape.
- **Intraday momentum in specific windows.** Gao, Han, Li & Zhou (2018), *Market Intraday
  Momentum* (Journal of Financial Economics) — find the first half-hour return predicts the
  last half-hour; this is a window-specific effect, not a "every Fisher cross" signal.

## Cost and turnover

- **Novy-Marx & Velikov (2016).** *A Taxonomy of Anomalies and Their Trading Costs* (Review of
  Financial Studies) — at the Fisher crossover's natural turnover (~550 crossovers/year/ticker),
  even a tiny per-trade cost kills a borderline edge. Since the gross edge is already negative
  in this study, the break-even cost is undefined.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.summarize`](../fisher_transform/strategy.py).
- **Block bootstrap CI.** Politis & Romano (1994), *The Stationary Bootstrap* (JASA) —
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).
- **Reproducibility stamp.** Content fingerprints in `docs/results.md` pin the tape to a
  specific as-of date; the offline core and test-suite run on the deterministic
  [`data.synthetic_daily`](../fisher_transform/data.py) generator without network access.

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`), 10-year window through 2026-06-15, on six
  liquid tickers: SPY, QQQ, IWM, AAPL, TSLA, NVDA. The offline core and tests run on the
  deterministic [`data.synthetic_daily`](../fisher_transform/data.py) generator.

## Related desk studies

- **[Study 72 — Loaded-Dice](../../72-loaded-dice/)**: the 5-minute SMA(5/10) crossover, tested
  the same way — same verdict (NONE/MIRAGE), same random-direction control discipline.
- **[Study 127 — Williams-R](../../127-williams-r/)**: Williams %R oscillator on daily bars —
  another normalised price oscillator, same family and conclusion.
- **[Study 106 — Supertrend](../../106-supertrend/)**: ATR-based trend filter, related family,
  same honest-baseline discipline.
- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: daily 50/200 SMA golden cross — same
  moving-average-crossover family, one step removed from oscillator to raw MA.
