# References & literature map — Study 503 (Expected Idiosyncratic Skewness)

## The claim under test

- **The anomaly (Boyer, Mitton & Vorkink).** Brian Boyer, Todd Mitton & Keith Vorkink,
  *Expected Idiosyncratic Skewness* (Review of Financial Studies, 2010). They build a forecast
  of each stock's **idiosyncratic skewness** — the asymmetry of its own-return distribution,
  orthogonal to the market — and show that high *expected* idio-skew predicts **low** subsequent
  returns in the cross-section. Investors over-pay for the small chance of a large idiosyncratic
  upside (a lottery-like right tail), so positively-skewed names are bid above fair value and
  underperform. The natural trade is **long low expected idio-skew, short high**.
- **The behavioural mechanism.** Skewness preference rests on **probability weighting** and a
  taste for positively-skewed payoffs: Tversky & Kahneman, *Advances in Prospect Theory* (1992);
  Barberis & Huang, *Stocks as Lotteries* (American Economic Review, 2008); Brunnermeier,
  Gollier & Parker, *Optimal Beliefs, Asset Prices, and the Preference for Skewed Returns*
  (American Economic Review P&P, 2007). Investors over-weight small probabilities of large gains
  and so over-pay for idiosyncratically skewed, lottery-like stocks.
- **The proxy we use.** *Expected* idio-skew is a forecast; the simplest honest predictor is
  **trailing realised residual skew** — regress a name's daily returns on the market over a
  rolling window and take the skewness of the residuals (Boyer-Mitton-Vorkink show realised
  idio-skew is the dominant predictor of its own future value). We use a trailing 12-month daily
  market-model residual skew, recomputed monthly.

## Distinct from its neighbours (the brief flags both)

- **MAX (one-day pop), Study 365.** Bali, Cakici & Whitelaw, *Maxing Out* (Journal of Financial
  Economics, 2011), sort on the single **highest daily return** over the prior month. MAX is one
  *point* of the distribution; idiosyncratic skewness is the **shape** of the whole residual
  distribution (its third standardised moment). A name can have a modest single-day max yet a
  fat, persistent right tail — and vice versa. See [Study 365 — Lottery-MAX](../365-lottery-max-effect/).
- **Coskewness (systematic).** Harvey & Siddique, *Conditional Skewness in Asset Pricing Tests*
  (Journal of Finance, 2000), price **coskewness** — a stock's contribution to the *market's*
  skewness (a systematic, undiversifiable tail). Boyer-Mitton-Vorkink's object is the opposite:
  the **idiosyncratic**, market-orthogonal tail, which a diversified investor *could* diversify
  away but, behaviourally, pays up for instead. We strip the market out by construction (residual
  skew), so this is the idiosyncratic axis, not coskewness.

## Why our tape can *invert* the published result — survivorship and universe

- **The published effect lives in the small-cap / lottery tail.** Boyer-Mitton-Vorkink (and the
  related Kumar, *Who Gambles in the Stock Market?*, Journal of Finance, 2009) find the
  skewness-preference premium is strongest among small, low-priced, retail-held names — exactly
  the segment a free large-cap feed excludes. We run the sort on a fixed **S&P-100-style
  large-cap basket** (yfinance adjusted closes), an explicit **proxy** that is the *least*
  lottery-prone slice of the market.
- **On survivors, high idio-skew can flag winners, not losers.** A fixed surviving-large-cap
  basket carries **survivorship bias**: a positively-skewed residual on a survivor often belongs
  to a high-beta growth name that subsequently led the 2009–2026 bull market. The desk's rule is
  to **name survivorship on the Signal axis** and reason about its direction explicitly
  (METHODOLOGY → *Survivorship is named on the Signal axis*). A current-membership / surviving
  panel can mute or invert an anomaly outright.
- **Anomalies weaken outside micro-caps.** Hou, Xue & Zhang, *Digesting Anomalies* (Review of
  Financial Studies, 2015), document that many cross-sectional anomalies fade once micro-caps are
  excluded — the same mechanism that flattens a skewness sort on large survivors.

## The inference bar (why literature support alone is never `REAL`)

- **The bar.** `REAL` is earned only by an autocorrelation-robust statistic clearing **t = 2**
  *in the claimed direction* on the real tape, surviving a placebo null (METHODOLOGY → *The
  inference bar*). We test the long-low / short-high mean with a **Newey-West (HAC) t-stat**
  (Newey & West, 1987, *A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation
  Consistent Covariance Matrix*, Econometrica) and a **sign-flip placebo** null (Fisher's
  randomization logic; Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993).
- **Selection / multiple testing on a famous factor.** Harvey, Liu & Zhu (2016), *…and the
  Cross-Section of Expected Returns* (Review of Financial Studies), and McLean & Pontiff (2016),
  *Does Academic Research Destroy Stock Return Predictability?* (Journal of Finance) — published
  anomalies decay (and sometimes reverse) out-of-sample and out-of-universe. A skewness sort
  failing to certify on large-cap survivors is consistent with both effects.

## Method lineage (the desk's shared engine)

- **Trailing market-model residual skew.** [`data.build_panel`](../expected_idiosyncratic_skewness/data.py)
  regresses each name's daily returns on SPY over a rolling 12-month window and takes the
  Fisher-Pearson skewness of the residuals — the transparent proxy for expected idio-skew.
- **Quintile sort + long-short.** [`strategy.quintile_returns`](../expected_idiosyncratic_skewness/strategy.py)
  ranks the cross-section by idio-skew each month and earns each quintile's next-month return
  (one execution lag, baked into the panel: month-*t* skew pairs with month *t+1*'s return);
  [`strategy.long_short`](../expected_idiosyncratic_skewness/strategy.py) forms Q1 − Q5 net of
  one-way costs × turnover and short borrow.
- **Robust inference.** [`strategy.hac_tstat`](../expected_idiosyncratic_skewness/strategy.py)
  (Newey-West) and [`strategy.placebo_pvalue`](../expected_idiosyncratic_skewness/strategy.py)
  (sign-flip null) — the Signal-axis tests on the spread mean.
- **Deterministic synthetic control.**
  [`data.synthetic_panel`](../expected_idiosyncratic_skewness/data.py) plants a known skewness
  penalty (high-skew names made to underperform next month); the offline core runs with no
  network. [`strategy.seed_robust_synth`](../expected_idiosyncratic_skewness/strategy.py) averages
  the control's HAC *t* over 20 seeds, per the house seed-robustness bar. The control confirms the
  sort+inference recover a *real* low-minus-high edge when present and find **nothing** when the
  edge is zero — so the real-tape *t* is a genuine universe feature, not a coding artefact.

## Data sources used here

- **yfinance** daily adjusted closes for a fixed S&P-100-style large-cap basket **plus SPY**
  (the market proxy), 2005-01-03 → 2026-05-29, cached under `_cache/basket_prices.csv`. All
  headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 365 — Lottery-MAX-Effect](../365-lottery-max-effect/)**: the one-day-pop cousin; the
  same survivor-universe caveat, a different (single-point vs whole-shape) signal.
- **[Study 330 — Low-Volatility-Anomaly](../330-low-volatility-anomaly/)**: calm beats wild
  *risk-adjusted*; same 2009–2026 bull window flattering the high-beta / lottery leg.
- **[Study 238 — Betting-Against-Beta](../238-betting-against-beta/)**: the beta sort that a
  large-cap lottery sort tends to collapse into once the genuine lottery tail (small, illiquid)
  is excluded.
