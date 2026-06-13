# References & literature map — Study 88 (Dogs-of-the-Dow)

## The claim under test

The **Dogs of the Dow**: each January, buy the **ten highest-dividend-yield** stocks in
the Dow Jones Industrial Average, **equal-weight**, hold for a year, then rebalance into
that year's new ten. The strong, sold-at-full-strength version is that this trivially
simple rule **beats the Dow itself** — a free lunch a retail investor can run with a
brokerage account and a spreadsheet.

- Michael O'Higgins & John Downes, *Beating the Dow* (HarperCollins, 1991) — the book that
  launched the strategy; the "high-yield ten" is its headline tactic.
- Popular write-ups keep it alive: the *Dogs of the Dow* website
  (<https://www.dogsofthedow.com/>), Investopedia, *"Dogs of the Dow"*
  (<https://www.investopedia.com/terms/d/dogsofthedow.asp>), and an annual cycle of
  finance-media articles every January.

## Why the steelman is almost coherent

- **The value / high-yield premium is real and documented.** High dividend yield is a
  classic value proxy, and value has earned a long-run premium (Fama & French, *The
  Cross-Section of Expected Stock Returns*, JF 1992; *Common Risk Factors*, JFE 1993).
  Buying the cheapest-by-yield names is a crude value tilt, so it is not pure superstition.
- **Dividend-tilt strategies are defensive.** High-yield blue chips tend to have **below-1
  market beta**, so a high-yield basket can post a smoother ride and a better risk-adjusted
  number than the cap-weighted parent even when raw returns are similar.

## Why it is likely to fail *as stated* ("beats the Dow")

- **Whatever edge exists is mostly the value/high-yield tilt — beta you were always paid
  for**, not a Dow-specific secret. An honest test must run **alpha-vs-beta** (regress the
  Dogs' excess return on the Dow's) and ask whether the intercept survives. (See the
  general critique of mechanical-screen strategies in McQueen, Shields & Thorley, *Does the
  "Dow-10 Investment Strategy" Beat the Dow Statistically and Economically?*, FAJ 1997 — an
  early, careful refutation that the headline gap clears a significance bar once you account
  for risk, size and taxes.)
- **The trailing-yield screen is a dividend-trap magnet.** A stock that has crashed shows a
  *mechanically* high trailing yield right before it cuts the payout — the January-2009 Dogs
  basket on this tape was led by Citigroup (22.8% trailing yield) and Bank of America
  (20.8%), names whose dividends were about to be slashed. The "highest yield" is sometimes
  the market pricing in a cut.
- **Taxes and turnover.** A high-yield basket throws off a large, fully-taxable dividend
  stream every year; a taxable investor pays that drag annually, and the January rebalance
  realizes short-term-adjacent gains. The headline backtest usually ignores both.

## Survivorship / selection — named on the Signal axis

The Dow's **membership changes** (~12 reshuffles in our window). Using today's 30 names back
to 2000 is textbook survivorship bias: removed losers (GM, Eastman Kodak) vanish and added
winners (Apple, Nvidia) get back-dated into a portfolio that never held them — a bias that
can *manufacture* a result. This study encodes a **point-in-time** membership timeline
(`dogs_of_the_dow/data.py`) and selects each January's Dogs only from the members **as of
that January**. Residual selection risk is named in the verdict: two members have no
recoverable Yahoo tape (Eastman Kodak 2000–04, Walgreens 2019–24) and are left out — 29 of
30 priced each year. Both were plausible high-yielders, so their absence is a real, named
limitation, not a silent cap.

## Method lineage

- **Newey–West HAC standard errors** for the mean of an autocorrelated annual excess:
  Newey & West (1987), *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix*, Econometrica.
- **Circular block bootstrap** for the CI on the annual excess (a 26-observation series is
  short; the bootstrap is honest about that): Politis & Romano (1992/1994).
- **Synthetic panel control** — a deterministic offline panel where high yield either does
  or does not carry a planted forward-return edge, so the harness's positive/negative
  controls are explicit (a pipeline that can't bank a planted signal proves nothing).

## Data sources used

- **Dow components**, daily, **total-return adjusted** (`yfinance`, `auto_adjust=True`)
  plus the per-share **dividend** stream (`yf.Ticker(t).dividends`), cached to parquet under
  `_cache/`. UTX→RTX and Kraft (KFT)→Mondelez (MDLZ) are price continuations of one listing.
- **Benchmark: DIA** (SPDR Dow Jones Industrial Average ETF), total return — the same-basis
  yardstick (total return vs total return; no price-only-vs-total-return race).

## Related desk studies

- [Study 18 — Dull-Roar](../../18-dull-roar/) — the low-volatility anomaly as a defensive
  low-beta tilt: the same "is it alpha or just beta you were paid for?" lens.
- [Study 28 — Carousel](../../28-carousel/) — another mechanical-rotation rule judged
  against equal-weight holding.
- [Study 91 — Death-Cross](../../91-death-cross/) — the template for this study's
  alpha-vs-beta / matched-control construction.
