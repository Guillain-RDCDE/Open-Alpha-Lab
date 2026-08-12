# References & literature map — Study 893 (Vol-Target 60/40)

## The claim under test

- **The idea.** Run a **volatility thermostat** on the balanced 60/40 book: scale total exposure
  *down* when the realized *portfolio* volatility spikes and *up* when it is calm, holding risk
  roughly constant at a target. The pitch — a *risk overlay, not alpha* — is that the re-timed book
  earns a better **excess-of-cash Sharpe** and a shallower **drawdown** than the static 60/40, net of
  the extra turnover.

- **The mechanism (why it can work).** **Moreira & Muir**, *"Volatility-Managed Portfolios"*
  (Journal of Finance, 2017): because expected returns move little with recent volatility while
  volatility itself is strongly **persistent** (it clusters — Mandelbrot 1963; Engle 1982, ARCH),
  scaling exposure by `σ_target / σ̂` — a *past-only* position, no return forecast — can raise the
  Sharpe and earn a spanning alpha. The same logic applied to *equities* is this desk's
  [Study 16 (Storm-Shy)](../../16-storm-shy/); here it runs on the diversified 60/40 book.

- **The specific test here.** Static **60% SPY / 40% IEF** (total return), annual rebalance, vs the
  same blend scaled to a constant target by its trailing-21-day realized portfolio vol (one lag, a
  2.0× cap). We race **excess-of-cash vs excess-of-cash** (both minus BIL), set the target to the
  static's own realized vol so the drawdown comparison is at matched risk, and grade the edge with a
  leverage-clean spanning alpha, a bootstrap Sharpe-difference CI, a two-era cut, a cost + borrow
  sweep, and a seeded synthetic control.

## What we measure, and the honesty rails

- **Past-only sizing, one documented lag.** The vol forecast is the trailing realized portfolio vol
  **lagged one bar** (`.shift(1)`): the weight applied on day `t` uses only returns through `t−1`.
  Zero look-ahead.
- **Excess-vs-excess, total return.** Both books are measured net of the BIL cash return, so the
  Sharpe race is fair to the leverage each carries; `auto_adjust=True` gives total-return closes.
- **Leverage-clean significance.** Because the thermostat's average exposure sits slightly above 1
  (Jensen on `1/σ̂`), a plain *t* on the return difference is contaminated by that level tilt. The
  headline significance is therefore the **Moreira–Muir spanning alpha** (regress the managed book on
  the static book; the HAC-*t* intercept is the pickup that survives matching the static's beta) and
  a **circular block bootstrap** CI on the leverage-invariant Sharpe *difference*.
- **Short single-cycle history — named on the Signal axis.** BIL lists **2007-05-30**; the joint
  window is ~19 years that *begins inside the GFC*. Magnitudes are one draw, not a law.
- **The timer is graded separately.** One-way cost × overlay turnover **plus** a borrow spread on the
  levered fraction — the honest test of whether a thin, leverage-financed edge survives friction.

## Shared method citations

- **Moreira, A. & Muir, T. (2017)** — *Volatility-Managed Portfolios*, Journal of Finance 72(4). The
  spanning-alpha test and the "scale by inverse realized variance" rule.
- **Engle, R. (1982)** — Autoregressive Conditional Heteroskedasticity (ARCH); **Mandelbrot, B.
  (1963)** — volatility clustering. Why realized variance is forecastable at all.
- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent (HAC)
  covariance — the *t* on the spanning-alpha intercept and on the return difference.
- **Politis, D. & Romano, J. (1994)** — the block bootstrap used for the Sharpe-difference CI (blocks
  preserve the volatility clustering an i.i.d. resample would destroy).
- **Wilson, E. B. (1927)** — score interval for a binomial share.

## Data sources

- **yfinance daily closes** (`auto_adjust=True`, total return): SPY, IEF, AGG, BIL, 2007-05-31 →
  2026-06-30, cached per ticker under this study's own `_cache/`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py) (fingerprint `a874e54fa109`).

## Related desk studies (the dedup map — what this study is NOT)

- [16-storm-shy](../../16-storm-shy/) — the same inverse-vol overlay on **equities** (SPY/QQQ/EFA),
  not the balanced book. This study runs the thermostat on the **60/40 portfolio's own realized
  vol**, so the diversification of the static blend is already in the baseline.
- [97-balancing-act](../../97-balancing-act/) — the **static** 60/40 (SPY/IEF), the baseline this
  study tries to beat. Study 97 grades the fixed blend Real/Investable; here we ask whether a vol
  overlay *improves* on it.
- [591-vol-managed-portfolio](../../591-vol-managed-portfolio/) — Moreira–Muir `c/RV` scaling on a
  **single index** (SPY/QQQ/EFA/IWM), monthly, with an expanding normaliser. This study targets the
  **portfolio** vol of a two-asset 60/40 with a daily trailing window and a matched-risk target.
- [68-all-weather](../../68-all-weather/) — **risk parity** (weight *each asset* by its own inverse
  vol across SPY/IEF/GLD/DBC). That re-weights *between* assets; this study leaves the 60/40 mix fixed
  and scales the *whole book's* exposure up and down through time.

None of the siblings run a **constant-vol thermostat on the 60/40 book's realized portfolio
volatility** — this study's own axis.
