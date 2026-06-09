# References & literature map — Study 16 (Storm-Shy)

## The claim under test — and why it's the desk's first *green*

- **Volatility-managed portfolios.** Alan Moreira & Tyler Muir, *"Volatility-Managed Portfolios"*,
  **Journal of Finance** 72(4), 2017. The headline result we steelman and test: scaling exposure by
  the inverse of recent realized variance — ``w_t = c / σ̂²_{t−1}``, a position set from *past*
  information only — **raises the Sharpe ratio** of the market and most equity factors, and earns a
  significant **spanning alpha** (the managed factor is not replicable by a static position). The
  mechanism is the one this study isolates: expected returns are roughly *unrelated* to recent
  volatility, while volatility itself is highly persistent — so cutting risk when vol is high (and
  adding it when vol is low) avoids the worst risk-adjusted periods without forecasting returns at
  all. We size by ``1/σ`` (constant-risk / vol-targeting) rather than ``1/σ²``; the two are the same
  idea and our offline core proves the lift comes from the vol *clustering*, not the exponent.

- **Why this one survives where fifteen mirages didn't.** The desk's standing verdict is "most
  ideas are mirages." This one is different on the two axes that usually kill an edge: it needs no
  return forecast (only a *variance* forecast, the most forecastable object in markets), it has
  **low turnover** (the forecast moves slowly, so costs are a few bps against a break-even far
  above), and it sizes the **most liquid instrument on earth** (an index), so capacity is enormous
  — the opposite of the ~\$10M capacity walls that sank earlier studies. That is what earns the
  rare **`INVESTABLE`** stamp.

## Why the steelman must be *bounded*, not swallowed — the honest counter

- **The certainty-equivalent critique.** Scott Cederburg, Michael O'Doherty, Feifei Wang & Xuemin
  (Sterling) Yan, *"On the performance of volatility-managed portfolios"*, **Journal of Financial
  Economics** 138(1), 2020. The essential caution we price in: a Sharpe-ratio gain is **not** the
  same as an investor being better off. A vol-managed strategy must take **leverage in calm
  periods** to hold its risk target, and once you evaluate it by a risk-averse investor's
  direct **mean–variance / CRRA certainty-equivalent at matched risk**, the gain shrinks — and for
  several factors becomes statistically insignificant out of sample. Our [`decompose.certainty_equivalent`](../storm_shy/decompose.py)
  runs exactly this test; the study reports the smaller, real number, not just the headline Sharpe.
  For the **market** factor specifically the effect is among the more robust, which is why the
  verdict lands `REAL` rather than `WEAK` — but the third axis, *"Free lunch? → `RISK-MANAGED`,"*
  keeps the desk honest: this is a risk-management gain, not alpha from nothing.

- **Vol-timing more broadly.** Fleming, Kirby & Ostdiek, *"The economic value of volatility timing"*
  (JF 2001) — that conditioning on volatility forecasts has real economic value — is the earlier
  pillar; Barroso & Santa-Clara, *"Momentum has its moments"* (JFE 2015) is the same idea applied to
  taming momentum-crash risk. The common thread the study leans on: timing *risk* works because risk
  is predictable, even where timing *returns* does not.

## The engine: why volatility is forecastable at all

- **Volatility clustering** is the most replicated stylized fact in empirical finance: large moves
  follow large moves. Benoit Mandelbrot, *"The variation of certain speculative prices"* (J. Business
  1963) first named it; Robert Engle's **ARCH** (*Econometrica* 1982, Nobel 2003) and Tim
  Bollerslev's **GARCH** (*J. Econometrics* 1986) made it a forecast. Our offline tape bakes in a
  persistent two-state vol regime so realized variance is autocorrelated by construction (the AR(1)
  ρ the diagnostics recover), and a **flat-vol null** where ρ ≈ 0 confirms that with nothing to
  forecast the overlay correctly adds nothing.

## Method lineage (the desk's shared engine)

- **Robust inference.** The spanning alpha carries a **Newey–West (HAC)** *t*-stat (Newey & West
  1987) via [`decompose._ols_nw`](../storm_shy/decompose.py) — the regression-intercept analogue of
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py). The Sharpe gain carries a
  paired **bootstrap** CI ([`decompose.sharpe_gain_bootstrap`](../storm_shy/decompose.py), cf.
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py)), and decay/capacity lean on
  [`quantlab.analytics`](../../../quantlab/analytics.py) (rolling Sharpe; square-root market impact).
- **Matched-risk fairness.** Because the Sharpe ratio is invariant to constant leverage, the Sharpe
  comparison is already fair to average exposure; the drawdown/return and utility comparisons lever
  **both** books to the same unconditional vol first ([`decompose.equal_risk_return`](../storm_shy/decompose.py),
  [`decompose.certainty_equivalent`](../storm_shy/decompose.py)) so neither wins by simply running
  more or less risk.
- **Reproducibility.** Every headline real run carries an `as_of` freeze + content fingerprint
  ([`quantlab/repro.py`](../../../quantlab/repro.py)).

## Data sources used here

- **Yahoo! Finance daily closes** (via `yfinance`, `auto_adjust=True` — split/dividend-adjusted, a
  data-choice stated per the house rules: total-return closes are the right input for a long-horizon
  risk-premium harvest). SPY (since 1993) & QQQ (since 1999), `period="max"`, for the real run; the
  offline core needs no network and the window is pinned with `as_of` + a fingerprint.

## Related desk studies

- **Study 12 — Paper-Prophet** is the direct ancestor: it found an ARIMA+GARCH forecasting stack was
  "vol-targeting in a trenchcoat" — the overlay was the *only* real thing inside the fake one. This
  study takes that residual and makes it the **hero**, run through the full protocol on its own
  terms. **Study 14 — Gamma-Gospel** ("the VIX in a trenchcoat") and **Study 06 — Clockwork-Vol**
  (can you *time* the VIX on a cycle? — no) are the desk's other volatility studies; Storm-Shy is the
  one where the volatility angle finally *pays*.
- The contrast with the desk's mirages is the point: where earlier edges died to costs, capacity, or
  disguised beta, this one survives all three — the honest "yes" that makes the honest "no"s
  credible.
