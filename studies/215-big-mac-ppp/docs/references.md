# References & literature map — Study 215 (Big-Mac-PPP)

## The claim under test

- **The Economist Big Mac index (1986-present)** — *"The Big Mac index"*,
  The Economist (annual, first published September 1986). The original source: the
  Economist's Big Mac index compares the local price of a McDonald's Big Mac to the
  US price to infer an implied PPP exchange rate. If the implied rate differs from
  the market rate, the currency is deemed over- or under-valued. The claim: this
  cheap, intuitive signal predicts mean-reversion in FX over the medium term.

## The PPP theory behind the claim

- **Cassel, G. (1918)** — *"Abnormal deviations in international exchanges"*,
  Economic Journal 28(112), pp. 413-415. The foundational statement of purchasing
  power parity: exchange rates should in the long run equalize the price of a basket
  of goods across countries. Deviations are temporary; reversion is expected.
- **Froot, K.A. & Rogoff, K. (1995)** — *"Perspectives on PPP and long-run real
  exchange rates"*, Handbook of International Economics Vol. 3. The canonical survey:
  PPP holds in the very long run (decades) but is strongly rejected at 1-3 year
  horizons. Half-lives of real exchange rate deviations are estimated at 3-5 years.
  This is the key empirical regularity that limits the Big Mac strategy: even if the
  signal is correct, reversion is too slow to produce annual alpha.
- **Rogoff, K. (1996)** — *"The purchasing power parity puzzle"*, Journal of Economic
  Literature 34(2), pp. 647-668. Documents the PPP puzzle: short-run exchange rates
  are driven by financial and monetary shocks, not goods-market arbitrage. Structural
  models imply much slower reversion than commodity arbitrage would suggest.

## Big Mac PPP as a forecasting signal

- **Cumby, R.E. (1996)** — *"Forecasting exchange rates and relative prices with the
  Hamburger standard: Is what you want what you get with McParity?"*, NBER Working
  Paper 5675. First rigorous statistical test of the Big Mac index as a predictor.
  Finds weak evidence that the Big Mac mis-valuation predicts future spot rates at
  12-24 month horizons, but the effects are small and statistically fragile.
- **Pakko, M.R. & Pollard, P.S. (1996)** — *"For Here or To Go? Purchasing Power
  Parity and the Big Mac"*, Federal Reserve Bank of St. Louis Review 78(1), pp. 3-21.
  Examines whether the Big Mac PPP is a useful exchange rate guide; finds it useful
  as a rough long-run anchor but unreliable for tactical currency positioning.
- **Clements, K.W., Lan, Y. & Seah, S.P. (2012)** — *"The Big Mac Index Two Decades
  On: An Evaluation of Burgernomics"*, The B.E. Journal of Macroeconomics 12(1).
  A thorough post-2000 evaluation. Finds that Big Mac mis-valuations have some
  long-run mean-reversion predictive power over 3-5 year horizons, but the one-year
  horizon is too short for statistically reliable signals.

## Why the edge fails at short horizons

- **Meese, R.A. & Rogoff, K. (1983)** — *"Empirical exchange rate models of the
  seventies: Do they fit out of sample?"*, Journal of International Economics 14(1-2),
  pp. 3-24. The seminal "random walk" result: structural FX models fail to outperform
  a naive random walk at 1-year horizons. Fundamental-based signals (including PPP)
  are overwhelmed by short-run noise.
- **Engel, C. & West, K.D. (2005)** — *"Exchange rates and fundamentals"*, Journal
  of Political Economy 113(3), pp. 485-517. Shows that if fundamentals are I(1), the
  exchange rate is also I(1) and nearly a random walk — PPP reversion is slow because
  the fundamental itself wanders.
- **Obstfeld, M. & Rogoff, K. (2000)** — *"The six major puzzles in international
  macroeconomics: Is there a common cause?"*, NBER Macroeconomics Annual 15, pp. 339-412.
  The PPP puzzle is one of six. Goods-market frictions (transport costs, non-tradeables,
  price stickiness) imply slow reversion that cannot be exploited at annual frequencies.

## Post-publication context

- **Chen, Y-C. & Rogoff, K. (2003)** — *"Commodity currencies"*, Journal of
  International Economics 60(1), pp. 133-160. Documents that commodity-exporting
  country FX (AUD, CAD, NZD) is driven by commodity prices, not PPP. Big Mac
  mis-valuation for these currencies partly reflects commodity cycles, not
  structural over/under-valuation.
- **Ito, T. & Chinn, M. (2009)** — *"East Asia and global imbalances: Saving,
  investment, and financial development"*, University of Chicago Press. Discusses
  persistent real exchange rate misalignments in managed-float regimes (JPY, CNY)
  where PPP reversion is suppressed by policy.

## Method lineage (the desk's shared engine)

- **OLS cross-sectional regression.** Fama (1984), *"Forward and spot exchange
  rates"*, Journal of Monetary Economics — the template for cross-sectional FX
  predictability tests. The PPP regression (mis_val -> fwd_ret) is the structural
  cousin of the UIP test.
- **HAC t-stat.** Newey & West (1987), *"A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix"*, Econometrica
  — used for inference on the annual portfolio return series.

## Data sources used here

- **The Economist Big Mac index** (hardcoded annual July snapshots, 2000-2023):
  raw-dollar method (local Big Mac price / US Big Mac price vs spot rate).
  Ten currencies: EUR, GBP, JPY, CAD, AUD, CHF, SEK, NOK, MXN, BRL.
- **Yahoo Finance FX spot rates** (via `yfinance`): annual log-returns measured
  Aug(T) to Jul(T+1) to match the Big Mac publication timing.

## Related desk studies

- **[Study 147 — FX-Momentum](../147-fx-momentum/)**: cross-sectional FX momentum
  (rank by past 12-month return vs USD). An orthogonal signal to Big Mac PPP: momentum
  bets on trend continuation, PPP bets on reversion. Both fail in the post-2010 sample.
- **[Study 36 — Greenback](../../36-greenback/)**: FX carry (interest rate
  differential). Carry differs from PPP: it exploits the UIP puzzle (high-yield
  currencies don't depreciate as theory predicts), not goods-market mis-pricing.
- **[Study 85 — Dr Copper](../../85-dr-copper/)**: commodity price as macro indicator.
  Like the Big Mac, a real-goods proxy for macro fundamentals; also shows limited
  short-run predictive power.
