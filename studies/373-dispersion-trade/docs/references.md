# References & literature map — Study 373 (Dispersion-Trade)

## The claim under test

- **The trade.** The *dispersion trade* (a.k.a. correlation trade): **sell index
  volatility / buy single-name volatility**, harvesting the gap between the implied
  volatility of an index option and the (correlation-weighted) implied vols of its
  constituents. Desk lore frames it as "index vol is reliably cheap versus the basket" —
  i.e. *implied correlation trades rich*, so a short-index-vol / long-single-name-vol book
  earns a persistent carry. See Bossu, *Introduction to Variance Swaps* (Wilmott, 2006) and
  Bossu, *A New Approach For Modelling and Pricing Correlation Swaps* (JPMorgan, 2007) for
  the canonical statement of the index-vs-basket variance identity.
- **The identity it rests on.** Index variance is a correlation-weighted average of
  single-name variances: σ²_index = Σ_i Σ_j w_i w_j ρ_ij σ_i σ_j. With pairwise correlation
  ρ < 1 the index vol is **strictly below** the weighted-average single-name vol — the
  *subadditivity of volatility*. This is a no-arbitrage identity (Jensen / Cauchy–Schwarz on
  the covariance matrix), **not** an empirical edge; the dispersion seller is fundamentally
  **short realized correlation**.

## Why true (implied) dispersion is not on yfinance — and what we do instead

- **Implied vols / variance swaps.** The real trade prices *implied* index variance against
  *implied* single-name variance (OTC variance swaps, or replicated from option strips). None
  of that is on the free yfinance endpoint, which serves per-ticker OHLCV only. We therefore
  build a transparent **realized-vol proxy**: rolling 21-day realized vol of SPY vs. the
  equal-weight average realized vol of a fixed 40-name large-cap basket, and the gap
  `avg_single_vol − index_vol`. This measures the *realized* dispersion the trade is
  ultimately settling against, labelled a proxy on the Signal axis. The implied-realized
  spread (the actual carry source) is *not* captured — which is exactly why a positive
  realized gap cannot, by itself, certify a tradable edge.
- **Realized-vol estimation.** Andersen, Bollerslev, Diebold & Labys (2003), *Modeling and
  Forecasting Realized Volatility* (Econometrica) — the rolling-window realized-vol estimator
  and its properties; Barndorff-Nielsen & Shephard (2002) on realized variance.

## Why a positive gap is not a free lunch — the statistics

- **Correlation risk premium.** Driessen, Maenhout & Vilkov (2009), *The Price of
  Correlation Risk: Evidence from Equity Options* (Journal of Finance) — index options are
  expensive relative to individual options precisely because **correlation carries a risk
  premium**; the dispersion seller is *paid* for bearing correlation-spike risk, not handed a
  free lunch. Buraschi, Kosowski & Trojani (2014), *When There Is No Place to Hide:
  Correlation Risk and the Cross-Section of Hedge Fund Returns* (RFS) — correlation risk is a
  priced, systematic factor that detonates in crises.
- **HAC inference on autocorrelated carry.** Overlapping rolling-vol windows make the daily
  carry strongly autocorrelated; the naive *t* overstates significance. We use a
  **Newey–West (HAC)** standard error (Newey & West, 1987, *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica) and a
  **block-sign-flip placebo** that preserves within-block structure (Politis & Romano, 1994,
  stationary bootstrap; Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993).
- **Skew / tail.** The dispersion seller's P&L is **negatively skewed in correlation** (long
  convexity in vol-of-dispersion that bleeds, occasional large losses when correlation pins
  near 1). Carr & Wu (2009), *Variance Risk Premia* (RFS) on the variance-swap carry's tail;
  Bakshi & Kapadia (2003) on the volatility risk premium.

## Method lineage (the desk's shared engine)

- **Dispersion proxy + implied correlation.**
  [`data.dispersion_proxy`](../dispersion_trade/data.py) builds index vol, average
  single-name vol, the gap, and the one-factor implied correlation `(σ_idx/avg_σ)²`.
- **Carry book + HAC / placebo.** [`strategy.carry_pnl`](../dispersion_trade/strategy.py)
  is the long-single / short-index variance carry struck at a trailing reference;
  [`strategy.hac_t_vs_zero`](../dispersion_trade/strategy.py) and
  [`strategy.placebo_pvalue`](../dispersion_trade/strategy.py) are the Signal-axis tests.
- **Deterministic synthetic control.**
  [`data.synthetic_basket`](../dispersion_trade/data.py) builds a one-factor correlated
  basket with a **planted carry knob**; the offline core runs with no network. The control
  confirms the engine recovers a planted edge **and** that the mechanical (subadditivity) gap
  manufactures **zero** carry when the true edge is zero.

## Data sources used here

- **yfinance** daily adjusted closes for SPY + a fixed 40-name large-cap basket, 2005-01-03 →
  2026-06-18, cached under `_cache/basket_prices.csv`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **Study 03 — Fear-Gauge**: the VIX as the implied-vol object the index leg of a dispersion
  trade is actually short.
- **Study 05 — Twin-Spread** and the broader **Options & volatility** bench family: other
  volatility-premium structures where a "reliable" carry turns out to be payment for a tail.
