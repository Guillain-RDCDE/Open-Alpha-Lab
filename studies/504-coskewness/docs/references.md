# References & literature map — Study 504 (Coskewness)

## The claim under test

- **The factor (Harvey & Siddique).** Campbell R. Harvey & Akhtar Siddique, *Conditional Skewness
  in Asset Pricing Tests* (Journal of Finance, 2000). They show that **systematic (co-)skewness**
  — a stock's contribution to the *market portfolio's* skewness — is **priced** in the
  cross-section, beyond the CAPM beta and the Fama-French factors. A stock with **negative
  coskewness** tends to deliver low returns precisely when market volatility spikes / the market
  falls hard; it makes the investor's portfolio *more* left-skewed, so it is a poor hedge and must
  offer a **premium**. A stock with positive coskewness is insurance-like and earns less. The
  natural trade is **long low (negative) coskewness, short high coskewness**.
- **The earlier theory.** The idea that the *third moment* of returns is priced predates
  Harvey-Siddique: Robert A. Kraus & Robert H. Litzenberger, *Skewness Preference and the Valuation
  of Risk Assets* (Journal of Finance, 1976) derive a three-moment CAPM in which coskewness with
  the market carries a risk premium. Harvey-Siddique make it **conditional** (time-varying) and
  test it empirically.
- **The measure we use.** Harvey-Siddique's **direct standardised coskewness**,
  ``β_SKD = E[ε_i·ε_m²] / ( √E[ε_i²] · E[ε_m²] )``, where ``ε_i`` and ``ε_m`` are the demeaned
  daily stock and market returns over a rolling window. It is the standardised covariance of the
  stock's return with the *squared* market return — exactly "how the name moves when the market
  moves a lot." We compute it on a trailing 12-month daily window, recomputed monthly.

## Distinct from its neighbour (the brief flags it)

- **Idiosyncratic skewness, Study 503.** Boyer, Mitton & Vorkink, *Expected Idiosyncratic
  Skewness* (Review of Financial Studies, 2010) price the **idiosyncratic** tail — the skew of the
  market-model *residual*, the part a diversified investor *could* diversify away but
  behaviourally over-pays for (lottery demand → high idio-skew **underperforms**). Coskewness is
  the **opposite axis of the same regression**: the *systematic* co-movement of the name's tail
  with the market's tail, which a diversified investor *cannot* shed and is *rewarded* for bearing
  (low coskew → **out**performs). Same daily regression, orthogonal moments, opposite predicted
  sign. See [Study 503 — Expected Idiosyncratic Skewness](../503-expected-idiosyncratic-skewness/).
- **Tail-co-movement cousins.** Lower-tail dependence and downside beta (Ang, Chen & Xing,
  *Downside Risk*, Review of Financial Studies, 2006) price *correlation* in the left tail;
  coskewness prices the *third-moment* co-movement specifically. The desk's downside-beta sort is
  [Study 332 — Downside-Beta](../332-downside-beta/).

## Why our tape can *mute* the published result — survivorship and universe

- **The premium is small and dispersion-driven.** The coskewness premium is modest even in the
  full universe and is identified off the **cross-sectional dispersion** in crash-sensitivity —
  widest among smaller, more cyclical, more leveraged names. A fixed **S&P-100-style large-cap
  basket** compresses that dispersion: mega-cap survivors cluster near market-like coskewness, so
  the long-short spread shrinks toward zero.
- **Survivorship trims the realised-risk names.** A current-membership surviving basket excludes
  the very low-coskew names whose crash risk *materialised* — the ones that fell hardest and
  de-listed. That is exactly the left tail the premium is meant to compensate, so a survivor panel
  systematically understates it. The desk's rule is to **name survivorship on the Signal axis** and
  reason about its direction (METHODOLOGY → *Survivorship is named on the Signal axis*).
- **Anomalies weaken outside the broad universe.** Hou, Xue & Zhang, *Digesting Anomalies* (Review
  of Financial Studies, 2015), document that many priced cross-sectional effects fade once
  micro-caps are excluded — the same mechanism that flattens a coskewness sort on large survivors.

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
  factors decay (and shrink) out-of-sample and out-of-universe. A coskewness sort failing to
  certify on large-cap survivors is consistent with both effects.

## Method lineage (the desk's shared engine)

- **Trailing direct coskewness.** [`data.build_panel`](../coskewness/data.py) computes the
  Harvey-Siddique standardised coskewness ``E[e_i·e_m²]/(√E[e_i²]·E[e_m²])`` of each name's daily
  returns with SPY over a rolling 12-month window — the transparent systematic-skew signal.
- **Quintile sort + long-short.** [`strategy.quintile_returns`](../coskewness/strategy.py)
  ranks the cross-section by coskewness each month and earns each quintile's next-month return
  (one execution lag, baked into the panel: month-*t* coskewness pairs with month *t+1*'s return);
  [`strategy.long_short`](../coskewness/strategy.py) forms Q1 − Q5 net of one-way costs × turnover
  and short borrow.
- **Robust inference.** [`strategy.hac_tstat`](../coskewness/strategy.py)
  (Newey-West) and [`strategy.placebo_pvalue`](../coskewness/strategy.py)
  (sign-flip null) — the Signal-axis tests on the spread mean.
- **Deterministic synthetic control.**
  [`data.synthetic_panel`](../coskewness/data.py) plants a known coskewness premium (low-coskew
  names made to out-perform next month); the offline core runs with no network.
  [`strategy.seed_robust_synth`](../coskewness/strategy.py) averages the control's HAC *t* over 20
  seeds, per the house seed-robustness bar. The control confirms the sort+inference recover a
  *real* low-minus-high premium when present and find **nothing** when the edge is zero — so the
  real-tape *t* is a genuine universe feature, not a coding artefact.

## Data sources used here

- **yfinance** daily adjusted closes for a fixed S&P-100-style large-cap basket **plus SPY**
  (the market proxy), 2005-01-03 → 2026-05-29, cached under `_cache/basket_prices.csv`. All
  headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 503 — Expected-Idiosyncratic-Skewness](../503-expected-idiosyncratic-skewness/)**: the
  *idiosyncratic* (diversifiable, behavioural-lottery) tail — the opposite axis of the same
  regression, with the opposite predicted sign.
- **[Study 332 — Downside-Beta](../332-downside-beta/)**: a neighbouring left-tail-co-movement
  risk measure (correlation in down markets rather than third-moment coskewness).
- **[Study 238 — Betting-Against-Beta](../238-betting-against-beta/)** and
  **[Study 330 — Low-Volatility-Anomaly](../330-low-volatility-anomaly/)**: the beta / vol sorts a
  large-cap risk-factor sort tends to collapse into once the genuine dispersion (small, illiquid)
  is excluded.
