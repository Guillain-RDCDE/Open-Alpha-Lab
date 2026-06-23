# References & literature map — Study 374 (Vol-of-Vol 🌀)

## The claim under test

- **VVIX (the "vol of vol").** CBOE, *VVIX White Paper* / VVIX methodology — VVIX applies
  the VIX model-free implied-variance construction to **VIX options**, producing the
  expected 30-day volatility *of the VIX index itself*. History begins in 2007. The pitch
  in trader lore and sell-side notes: a VVIX spike means option markets are pricing a tail
  the *level* of vol (VIX) hasn't yet shown, so high VVIX should lead weak / drawdown-y
  forward equity returns — and do so **incrementally to the VIX**.
- **The folklore framing.** "When the vol-of-vol blows out, get out" — VVIX as a
  *sharper*, more forward-looking risk clock than the VIX. The whole interest of the claim
  is the word **incremental**: nobody disputes that VVIX co-moves with VIX; the question is
  whether it *adds* timing information beyond the VIX you already watch for free.

## Does vol-of-vol carry priced information? — the literature

- **Vol-of-vol as a priced factor.** Baltussen, Van Bekkum & Van der Grient (2018),
  *Unknown Unknowns: Uncertainty About Risk and Stock Returns* (Journal of Financial and
  Quantitative Analysis) — cross-sectional vol-of-vol commands a return premium. Cboe and
  practitioner studies (e.g. on the **VVIX/VIX ratio**) argue VVIX leads regime shifts.
- **The variance-of-variance / VIX-options literature.** Mencía & Sentana (2013),
  *Valuation of VIX derivatives* (Journal of Financial Economics); Park (2015), *The
  effects of the VVIX*; Huang & Shaliastovich on volatility-of-volatility risk premia —
  vol-of-vol is a genuine, priced object in option markets. **But priced cross-sectionally
  or in option prices is not the same as *timing the equity index*** — which is the
  specific, weaker claim this study tests.
- **VIX as the benchmark to beat.** Whaley (2000, 2009), *The Investor Fear Gauge* /
  *Understanding the VIX*; the broad finding that the VIX *level* is a weak, mostly
  contrarian predictor of forward equity returns (high VIX → high *subsequent* returns,
  because VIX peaks at bottoms). Any VVIX claim must out-predict this — our decisive test
  puts both in one regression.

## Why the right test is *incremental*, with HAC inference

- **Overlapping windows ⇒ serial correlation.** Forward 21-/63-day returns sampled daily
  are heavily overlapping, so OLS standard errors are badly understated. We use
  **Newey-West (1987)**, *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix* (Econometrica), with the lag set to the
  overlap length — the desk's standard HAC treatment for forward-return regressions.
- **Multicollinearity is the whole point.** With corr(VIX, VVIX) ≈ 0.4–0.7, a *univariate*
  VVIX regression is confounded by VIX. The bivariate coefficient on VVIX, **conditional on
  VIX**, is the only honest estimate of vol-of-vol's *marginal* information — the classic
  partial-effect logic (Frisch–Waugh–Lovell). A raw signal that evaporates when the
  correlated control is added is **redundant**, not incremental.
- **Base rates and the win-rate illusion.** US equities rise in most rolling windows, so a
  high post-signal win-rate is *expected under the null*; the right comparison is the
  **excess** over the unconditional base rate (Kahneman & Tversky, 1973, *On the psychology
  of prediction*). We also report forward **drawdowns**, where the signal genuinely (if
  weakly) bites.

## Method lineage (the desk's shared engine)

- **Incremental HAC regression.**
  [`strategy.incremental_regression`](../vol_of_vol/strategy.py) +
  [`strategy.newey_west_se`](../vol_of_vol/strategy.py) — VIX-only, VVIX-only, and the
  decisive VIX+VVIX bivariate fit, all with Newey-West SEs at the overlap lag.
- **Welch t + block placebo null.** [`strategy.welch_t`](../vol_of_vol/strategy.py) and
  [`strategy.placebo_pvalue`](../vol_of_vol/strategy.py) — conditional vs unconditional
  forward returns, and a 20,000-draw block-resampled randomization null that preserves the
  high-VVIX state's clustering.
- **Deterministic synthetic control.**
  [`data.synthetic_tape`](../vol_of_vol/data.py) builds VVIX as a VIX component **plus an
  independent vol-of-vol component**, and plants an edge **only** in the VIX-orthogonal
  part — so the control proves the engine isolates *incremental* signal (edge = 0 must not
  light up; a large edge must).
- **Execution lag + costs.**
  [`strategy.timing_backtest`](../vol_of_vol/strategy.py) — a 1-day-lagged long/flat rule
  with a one-way 2-bps cost per position change, raced gross/net against buy-and-hold.

## Data sources used here

- **yfinance** daily closes for `^VVIX`, `^VIX`, `SPY` (auto-adjusted), 2007-01-03 →
  2026-06-18, cached under `_cache/vol_tape.csv`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 111 — VIX-term-structure](../111-vix-term-structure/)**: the *shape* of the VIX
  curve (contango/backwardation) as a timer — a sibling "second-order vol" claim.
- **[Study 130 — Vol-risk-premium](../130-vol-risk-premium/)**: implied-minus-realized vol,
  the priced gap VVIX-style signals sit on top of.
- **[Study 330 — Low-volatility-anomaly](../330-low-volatility-anomaly/)**: the
  cross-sectional cousin — does *low* realized vol pay, and is it just beta in disguise?
