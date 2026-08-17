# References & literature map — Study 950 (Zero-Coupon Convexity)

## The claim under test

- **The convexity-pickup thesis.** A bond's price is a convex function of its yield, so the
  second-order term `+½·C·(Δy)²` in the price expansion is a *gift*: for the same duration,
  a more convex position gains more when yields fall than it loses when they rise. The
  retail-adjacent version of this argument says that a **zero-coupon Treasury fund** — EDV
  (Vanguard Extended Duration Treasury, 20-30y STRIPS) or ZROZ (PIMCO 25+ Year Zero-Coupon)
  — is the cleanest way to own that convexity, and that it should therefore out-earn a
  duration-matched holding of an ordinary coupon long-bond fund **in months when rates move
  a lot**, at the cost of a small give-up when they do not.
- **What makes it testable.** The claim is an explicit *asymmetry*, not an average. So the
  test is a regression of the duration-matched spread on the **squared** rate move (and on
  the realised variance of daily rate moves), where the theory pins the sign of both the
  quadratic coefficient (positive) and the intercept (negative — convexity is not free).
  An average-return race alone cannot distinguish "convexity is paid for" from "convexity
  does not exist here".
- **The steelman, and the catch.** A zero *does* have the highest convexity of any bond at
  its own maturity point, and the STRIPS funds sit further out the curve than TLT, so per
  unit of duration they genuinely carry more `C/D`. The catch is that for a *given*
  duration a zero has the **lowest** convexity of any cash-flow pattern — dispersion of cash
  flows is what buys convexity — so the pickup here comes from the curve point, not from the
  zero-ness, and it drags a 20s-versus-30s curve exposure along with it. The tape confirms
  precisely that: a residual +0.72 years of duration survives the match.

## The mechanics

- **Macaulay (1938)** and the standard duration/convexity expansion; **Fabozzi, *Bond
  Markets, Analysis and Strategies*** — the canonical treatment of `ΔP/P ≈ -D·Δy + ½·C·Δy²`,
  of why convexity is worth most when volatility is high, and of the barbell-versus-bullet
  arithmetic that a duration-matched comparison rests on.
- **Ilmanen (1995), *Convexity Bias and the Yield Curve*, Salomon Brothers** (and Ilmanen,
  *Expected Returns*, 2011, ch. on bond risk premia) — the argument that convexity is
  **priced**: the forward curve embeds a "convexity bias" that pays the more convex position
  a lower yield, so the expected gain from `½·C·Δy²` should be roughly offset by carry. Our
  negative intercept (−9 to −13 bp/month, *t* = −1.5 / −1.9) is the shape of that
  compensation; our positive-but-insignificant `b2` is the shape of the offsetting gain.
- **Litterman & Scheinkman (1991), *Common Factors Affecting Bond Returns*, Journal of Fixed
  Income** — level/slope/curvature. Why a single-factor (30-year yield) hedge cannot fully
  neutralise two funds at different curve points, and why the residual `b1` we report is a
  *curvature* exposure rather than a nuisance to be swept under the carpet.
- **Brown & Schaefer (1994)** on estimating the term structure and the sensitivity of any
  convexity measurement to the fitted curve — a caution that the "convexity" recovered from
  fund returns is a realised, model-free object, not the analytic `C` of a stripped bond.

## Why it can fail on this tape

- **The effect is second-order and the sample is one lifetime of months.** With ~208 months,
  a monthly spread vol of ~1.4% and a fitted pickup worth ~7 bp for a 25 bp move, the
  signal-to-noise ratio is hopeless by construction. Our synthetic control makes this
  concrete: the same harness resolves a *planted* pickup 2.5x larger at *t* = +6.3, and
  cannot resolve anything close to the real magnitude.
- **Convexity is sold, not given.** If the market prices convexity efficiently (Ilmanen's
  bias), the correct null is that the quadratic gain and the carry give-up cancel — which is
  exactly what a spread of −4 bp/month with a 28 bp breakeven move looks like.
- **Fund frictions and mandate drift.** EDV and ZROZ track index families with different
  maturity bands (20-30y vs 25y+) and different roll conventions; TLT's own duration has
  drifted with coupon levels since 2009. Any long-horizon "duration-matched" claim inherits
  that drift, which is why the hedge is re-solved every month rather than fixed once.

## Related desk studies (dedup)

- **[Study 884 — Convexity-Barbell](../../884-convexity-barbell/)**: the *same physical
  quantity* approached from the opposite direction — a duration-matched **SHY+TLT barbell**
  versus an **IEF bullet**, all coupon funds, where the barbell is the *more* convex side.
  Study 950 tests the **zero-coupon** funds against a **levered coupon** mix, where the
  convexity edge comes from the curve point rather than from cash-flow dispersion, and —
  crucially — it tests the **asymmetry explicitly** (a regression on the squared rate move
  and on realised variance) rather than reading it off an average. 884 concluded the
  barbell's convexity was exactly paid for; 950 finds the zero's convexity is not even
  measurable.
- **[Study 826 — Treasury-Duration-BAB](../../826-treasury-duration-bab/)**: betting against
  beta *across* the Treasury maturity curve — a cross-sectional risk-adjusted-return claim,
  levered to unit beta. Study 950 is not a factor sort: it is a two-arm, duration-matched
  race whose whole question is the *shape* of one spread's payoff in the rate move.
- **[Study 924 — Cut-Cycle-Duration-Extension](../../924-cut-cycle-duration-extension/)**:
  *timing* duration around a hardcoded list of first-cut FOMC dates. Study 950 has no timing
  device and no event list — the position is static and always on.
- **[Study 380 — Curve-Roll-Down](../../380-curve-roll-down/)** and
  **[Study 864 — Yield-Curve-Twist](../../864-yield-curve-twist/)**: first-order curve
  effects (carry, roll, slope reshaping). Those are the *`b1`* of this study — the term we
  hedge out and then report honestly when the hedge leaks.
- **[Study 625 — Starting-Yield-Bond-Decade](../../625-starting-yield-bond-decade/)**: the
  mechanical identity between starting yield and subsequent bond returns; a level story, not
  a curvature one.

## Method lineage

- **HAC / Newey-West standard errors.** Newey & West (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*,
  Econometrica — [`strategy.newey_west_t`](../zero_convexity/strategy.py),
  [`strategy.hac_ols`](../zero_convexity/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Circular block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap*, JASA —
  [`strategy.block_bootstrap_ci`](../zero_convexity/strategy.py) and
  [`strategy.bootstrap_b2_ci`](../zero_convexity/strategy.py), which resamples whole blocks
  of consecutive months (design rows and outcomes together) so the rate-regime persistence
  is preserved.
- **Realised variance as the gamma-P&L regressor.** Andersen, Bollerslev, Diebold & Labys
  (2003), *Modeling and Forecasting Realized Volatility*, Econometrica — the reason we run
  the asymmetry regression twice, once on the squared net monthly move and once on
  `sum(dy_t²)`: a daily-marked fund accrues convexity along the path, not only between
  month-end marks.
- **Reproducibility stamp.** [`quantlab/repro.py`](../../../quantlab/repro.py) — the as-of
  slice and the content fingerprint printed above every headline table.

## Data sources

- **EDV** (Vanguard Extended Duration Treasury, 20-30y STRIPS), **ZROZ** (PIMCO 25+ Year
  Zero-Coupon US Treasury), **TLT** (iShares 20+ Year Treasury), **BIL** (SPDR 1-3 Month
  T-Bill, the cash leg) — daily **total-return** closes via `yfinance` (`auto_adjust=True`).
  Total return is not optional here: EDV and ZROZ distribute the accreted coupon of the
  STRIPS they hold, so their price-only series drifts down against the coupon fund for
  purely mechanical reasons.
- **`^TYX`** — the 30-year constant-maturity Treasury yield, a **level in percentage
  points** (not a price), used as the shared rate factor for the duration match and as the
  regressor in the asymmetry test.
- **Expense ratios are already inside the total-return tapes** (EDV 0.05-0.06%, ZROZ 0.15%,
  TLT 0.15%), so the race is net of them; that small differential mildly favours EDV and is
  left in rather than adjusted away.
- **The financing spread over bills on the levered part of the mix is a PROXY** (25 bp/yr
  headline, swept 0-100 bp/yr). It is the study's only non-tape input, and the sweep is
  reported because a wider spread makes the mix worse and therefore *flatters* the arm we
  are testing.
- **As-of 2026-06-30.** The partial current month is dropped so the sample never creeps.
  The window starts at EDV's inception plus the 252-day beta warmup (2009-02-02).
