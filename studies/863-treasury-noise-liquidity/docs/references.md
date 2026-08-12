# References & literature map — Study 863 (Treasury Noise Liquidity)

## The claim under test

- **The source paper.** Grace Xing **Hu, Jun Pan & Jiang Wang**, *"Noise as Information for
  Illiquidity"* (Journal of Finance, 2013, vol. 68). They construct a market-wide illiquidity
  measure from the U.S. Treasury yield curve: on each day they fit a smooth zero-coupon curve
  (a Svensson / spline model) to the universe of outstanding notes and bonds, and take the
  **root-mean-square deviation of observed yields from the fitted curve** — the *"noise"*. The
  economic logic: when arbitrage capital is abundant, relative-value desks trade any off-curve
  bond back onto the curve, so residuals stay tiny; when capital is **scarce** (a funding
  squeeze, a crisis), the smoothing force weakens and yields scatter, so noise rises. The
  measure spikes in 1987, LTCM 1998, and the 2007–2009 crisis, and it **prices the
  cross-section of hedge-fund and currency-carry returns** as a systematic illiquidity factor.
- **The behavioural / institutional reading.** Noise is a real-time gauge of the *shadow price
  of arbitrage capital*. High noise = intermediaries are constrained = risk premia are about to
  be repriced upward, so risky assets (equities, credit) are said to earn **lower** returns
  going into the stress and wider spreads.
- **The specific test here.** We take a **self-contained daily** version anyone can rebuild from
  free data: four constant-maturity Treasury (CMT) par yields (13-week, 5-year, 10-year,
  30-year), a **quadratic-in-maturity** smooth fit (three parameters, four points — one degree
  of freedom for a residual), and `noise_t = RMS(residual)`. We then run *time-series predictive
  regressions* of the **forward SPY return** and the **forward HYG − IEF credit-excess return**
  on the noise level, with a Newey-West slope *t*, a block-rotation placebo, a two-era cut, a
  costed regime timer, and a seeded synthetic positive control. (Four CMT points is a far
  coarser curve than the paper's full CUSIP-level fit, so the magnitudes here are a floor;
  what survives is the *sign* and its era-dependence.)

## What we measure, and the honesty rails

- **Roughness, no free model.** One fixed residual-projection matrix
  `P⊥ = I − M(MᵀM)⁻¹Mᵀ` for the quadratic design `M = [1, m, m²]`; `noise = RMS(P⊥ · y)`. It is
  exactly zero on any perfectly quadratic curve and grows with cross-maturity scatter — the
  whole daily series is a single matmul.
- **Point-in-time, one documented lag.** The noise is **known at the close of day `t`**; the
  forward return is held `t → t+h` (`.shift(−1)` on the return leg). Zero look-ahead.
- **Robust inference.** Newey-West (HAC, Bartlett, lags = ⌈1.5·h⌉) *t* on the regression slope —
  an overlapping `h`-day forward return is strongly serially correlated, so a plain OLS *t*
  (also reported) badly overstates significance. A **3,000-draw block-rotation placebo** breaks
  the noise → forward-return link while preserving the overlap autocorrelation, to confirm the
  slope is not a lucky alignment.
- **No survivorship inflation.** The four CMT indices and the ETFs are continuously listed; the
  noise measure is cross-*maturity*, not cross-*name*, so there is no delisting bias. The only
  modelling choice — the risk-free leg proxied at 0 — is **named on the Signal axis** and moves
  the intercept, not the slope.
- **The timer is graded separately.** One-way × NAV per switch, long/flat — the honest test of
  whether the shallow daily edge survives friction.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the regression slope).
- **Newey, W. & West, K. (1994)** — the automatic lag-length rule used when a horizon is not
  supplied.
- **Wilson, E. B. (1927)** — score interval for a binomial share.
- **Gürkaynak, R., Sack, B. & Wright, J. (2007)** — *"The U.S. Treasury Yield Curve: 1961 to
  the Present"* (the Fed's fitted-curve methodology behind CMT / smooth-curve residuals).

## Data sources

- **yfinance daily closes** (`auto_adjust=True`): `^IRX` / `^FVX` / `^TNX` / `^TYX` (CMT yields,
  %-points) and `SPY` / `HYG` / `IEF` / `LQD` / `TLT`, 2007-04-12 → 2026-06-30, cached under
  `_cache/` as one parquet.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [112-move-index](../../112-move-index/) — **MOVE**, the option-*implied volatility* of
  Treasuries (a forward-looking, risk-neutral vol gauge). This study uses the *realized
  cross-maturity roughness* of the yield **levels** themselves — no options, no implied vol.
- [383-sofr-repo-stress](../../383-sofr-repo-stress/) — discrete **repo / SOFR funding spikes**,
  a short-rate money-market dislocation measured from a curated event table. This study is a
  continuous whole-curve **fitting residual**, not a money-market spread or an event flag.
- [386-nfci-conditions](../../386-nfci-conditions/) — a broad **financial-conditions index**
  proxy blending equity vol, rates vol, a credit spread and the dollar. This study is a
  **Treasury-only** curve residual — one instrument class, one number.
- [581-term-premium](../../581-term-premium/) — the **level** of the term premium (the risk
  compensation embedded in the curve's *shape*, à la Adrian-Crump-Moench). This study measures
  the **deviation *from* a smooth shape** (the fitting residual), which is orthogonal to the
  term premium's level.

None of the siblings measure the **cross-maturity fitting residual of the Treasury yield curve**
— the Hu-Pan-Wang noise signal — which is this study's own axis.
