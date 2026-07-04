# References — Study 638 (Value-Momentum-Everywhere)

## The claim's source

- **Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013). "Value and Momentum
  Everywhere."** *Journal of Finance*, 68(3), 929–985.
  <https://doi.org/10.1111/jofi.12021> — the claim under test: value and momentum premia
  exist **in every asset class** (individual stocks, country equity indices, currencies,
  government bonds, commodities), they are **negatively correlated** with each other
  everywhere (~−0.5 within classes), and the **50/50 combination** is far stronger than
  either alone — combo Sharpes near 1 in their 1972–2011 sample. For non-stock asset
  classes their value measure is the **negative of the long-run past return**
  (the 5-year reversal, skipping the most recent year) — exactly the proxy we use.

## Key papers

- **Jegadeesh, N., & Titman, S. (1993).** "Returns to Buying Winners and Selling Losers."
  *Journal of Finance*, 48(1), 65–91 — the 12-1 cross-sectional momentum construction.
- **De Bondt, W. F. M., & Thaler, R. (1985).** "Does the Stock Market Overreact?"
  *Journal of Finance*, 40(3), 793–805 — the long-horizon (5-year) reversal that AMP use
  as the universal value proxy; torn down on single stocks in
  [study 196 — long-term-reversal](../../196-long-term-reversal/).
- **Menkhoff, L., Sarno, L., Schmeling, M., & Schrimpf, A. (2012).** "Currency Momentum
  Strategies." *Journal of Financial Economics*, 106(3), 660–684 — FX momentum, torn down
  on the same G10 spot tape in [study 147 — fx-momentum](../../147-fx-momentum/) (Weak /
  Mirage post-publication).
- **McLean, R. D., & Pontiff, J. (2016).** "Does Academic Research Destroy Stock Return
  Predictability?" *Journal of Finance*, 71(1), 5–32 — the post-publication decay pattern
  our sub-period split is designed to detect.
- **Grinold, R. C., & Kahn, R. N. (2000).** *Active Portfolio Management* — the
  fundamental law behind the third axis: combining signals lifts the ratio only through
  breadth/correlation arithmetic. The null twin is
  [study 401 — signal-stacking](../../401-signal-stacking/): the *same* arithmetic
  applied to **noise** ingredients multiplies zero. Here the ingredients are the two most
  famous *real* premia in the literature — and the arithmetic still can only deliver
  what the ingredients carry on the tape.
- **Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012).** "Time Series Momentum."
  *JFE*, 104(2), 228–250 — the time-series cousin, torn down in
  [study 31 — trade-winds](../../31-trade-winds/) (Weak on our tape; crisis alpha
  confirmed).

## Data sources

- **yfinance** (public, no key) — country equity ETFs (SPY + 12 single-country iShares
  MSCI funds, dividend-adjusted **total-return** closes, 1996→) and G10 **spot** FX pairs
  vs USD (**price-only — no carry**; Yahoo spot histories are usable from late 1996 /
  Dec 2003 depending on the pair). <https://finance.yahoo.com/>
- **Shared repo futures cache** `_cache/_cache/trade_winds_futures.parquet` — daily
  returns of Yahoo continuous front-month futures (built once by
  [study 31 — trade-winds](../../31-trade-winds/); ±25% clip, roll-gap caveat documented
  there): US Treasury futures ZF/ZN/ZB (the BOND sleeve) and 8 commodity futures
  CL/GC/SI/HG/NG/ZC/ZS/ZW (the CMD sleeve). Continuous front-month **price** returns
  carry roll noise — stated, not hidden.

## Shared method citations

- **Newey, W. K., & West, K. D. (1987).** "A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica*,
  55(3), 703–708 — every mean/t on monthly L/S series is HAC (Bartlett kernel).
- **House rules** — [METHODOLOGY.md](../../../METHODOLOGY.md): the inference bar
  (REAL needs robust *t* ≥ 2 on the real tape), one documented execution lag, costs
  one-way × traded notional with the ETF short leg paying borrow, excess-vs-excess
  Sharpe races, random baselines averaged over ≥ 20 seeds, synthetic controls never
  cited as market evidence.
