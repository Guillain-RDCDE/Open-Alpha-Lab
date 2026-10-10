# Sources & literature map — Study 1027 (Oil Is Equity in a Mirror)

## The claim and its equity original

- **Black, F. (1976), "Studies of Stock Price Volatility Changes", *Proceedings of the 1976
  Meetings of the American Statistical Association, Business and Economic Statistics Section*,
  177-181.** The observation that stock volatility rises after price falls, and the
  debt-to-equity explanation that gave the "leverage effect" its name.
- **Christie, A. A. (1982), "The Stochastic Behavior of Common Stock Variances: Value, Leverage
  and Interest Rate Effects", *Journal of Financial Economics* 10(4), 407-432.** The classic
  empirical documentation of the negative return-volatility elasticity in equities — the
  pattern the S&P 500 leg here reproduces.
- **Baur, D. G. (2012), "Asymmetric Volatility in the Gold Market", *The Journal of Alternative
  Investments*.** Reports an *inverted* asymmetry in gold — positive shocks raising volatility
  more than negative ones — explained by gold's safe-haven role. This is the commodity
  "mirror" in its best-documented form; this study does not test gold.
- **Kristoufek, L. (2014), "Leverage effect in energy futures", *Energy Economics*.** Studies
  the return-volatility relation in energy futures — the energy-specific setting of the claim.
  We cite it for the question, not for a specific estimate: our test uses spot WTI and its own
  estimators, and we do not reproduce his.
- The wider commodity literature on asymmetric volatility is mixed in sign by market and
  sample; we cite only the works above, whose details we are confident of, and describe the
  rest without specific citation.

## Mechanism — supply versus demand shocks in oil

- **Kilian, L. (2009), "Not All Oil Price Shocks Are Alike: Disentangling Demand and Supply
  Shocks in the Crude Oil Market", *American Economic Review* 99(3), 1053-1069.** The
  decomposition that makes the claim's mechanism testable: if the mirror comes from supply
  shocks, it should track identified supply shocks rather than calendar eras. Listed as the
  natural next fork (beat 7).

## Method

- **Glosten, L. R., Jagannathan, R. & Runkle, D. E. (1993), "On the Relation between the
  Expected Value and the Volatility of the Nominal Excess Return on Stocks", *Journal of
  Finance* 48(5), 1779-1801.** The GJR-GARCH model.
- **Nelson, D. B. (1991), "Conditional Heteroskedasticity in Asset Returns: A New Approach",
  *Econometrica* 59(2), 347-370.** EGARCH.
- **Engle, R. F. & Ng, V. K. (1993), "Measuring and Testing the Impact of News on Volatility",
  *Journal of Finance* 48(5), 1749-1778.** The news-impact curve and sign-bias tests; the
  framing for "down/up impact" in §1.
- **Bollerslev, T. & Wooldridge, J. M. (1992), "Quasi-Maximum Likelihood Estimation and
  Inference in Dynamic Models with Time-Varying Covariances", *Econometric Reviews* 11(2),
  143-172.** The robust (sandwich) standard errors used for every GARCH coefficient.
- **Newey, W. K. & West, K. D. (1987), "A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix", *Econometrica* 55(3), 703-708.** HAC errors for
  the overlapping forward-volatility regressions.
- **Politis, D. N. & Romano, J. P. (1992), "A Circular Block-Resampling Procedure for
  Stationary Data", in *Exploring the Limits of Bootstrap*, Wiley.** The circular block
  bootstrap behind every interval on correlations, skewness and Sharpe gains.
- **Working, H. (1960), "Note on the Correlation of First Differences of Averages in a Random
  Chain", *Econometrica* 28(4), 916-918.** Why the monthly-*average* Brent tape is a
  direction-only cross-check.
- **Moreira, A. & Muir, T. (2017), "Volatility-Managed Portfolios", *Journal of Finance* 72(4),
  1611-1644.** The vol-target overlay whose behaviour on oil is the study's consequence test.
- **Sheppard, K., `arch` — ARCH models in Python** (the estimation library, and the package
  that ships the frozen tapes).

## Data provenance

All tapes are read from the `arch` wheel through [`quantlab/bundled.py`](../../../quantlab/bundled.py),
which verifies each file against a pinned SHA-256 before use. Pins and content fingerprints are
printed in [results.md §0](results.md).

| Tape | Source as shipped | Used as |
|---|---|---|
| `wti` | FRED `DCOILWTICO`, WTI Cushing **spot**, daily | the oil tape, 1986-01-02 → 2018-12-28 (partial Jan-2019 dropped). Spot, not investable. |
| `sp500` | S&P 500 index, daily OHLC/Adj Close | the equity mirror, **price index** (no dividends), 1999-2018 |
| `crude` | monthly Brent and WTI | **monthly averages** (verified against the daily tape), 1987-05 → 2019-12 |
| `frenchdata` | Ken French data library, monthly RF | T-bill rate for the S&P excess return, to 2018-11 |

## Neighbours on this desk

- **993-leverage-effect-asymmetry** — the equity leverage effect itself (SPY), with gold and
  Bitcoin as balance-sheet-free controls. This study takes the opposite starting claim (an
  *inverse* effect in oil) on a different tape, adds the regime question and the overlay.
- **245-oil-equity-correlation** — whether oil *leads* equities (it doesn't); a return-
  forecasting question, not a volatility-asymmetry one.
- **16-storm-shy**, **591-vol-managed-portfolio**, **898-managed-vol-equity** — the
  vol-target / volatility-managed overlay on equities. Here the overlay is the consequence
  test, run on oil against an equity control.
