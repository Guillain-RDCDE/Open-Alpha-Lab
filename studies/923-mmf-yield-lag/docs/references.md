# References & literature map — Study 923 (The Cash Lag)

## The claim under test

- **The cash-lag thesis.** Cash vehicles do not all reprice at the same speed. A money
  fund or bill ETF holds a *ladder*: its book yield is roughly the average of the rates at
  which its holdings were bought, so it inherits a change in the bill rate only as the
  ladder rolls. A fund with a 45-day weighted average maturity therefore lags the 13-week
  bill quote by weeks, while a floating-rate note fund whose coupon resets weekly does
  not. The folk conclusion, repeated on every treasury-management desk and in every
  personal-finance thread: *when rates are rising, sit in the fast repricer; when they are
  falling, sit in the slow one, which keeps a stale-high yield and books a duration gain
  on top*. This study tests both halves — the lag, and the rotation.
- **The steelman.** The lag is not folklore; it is bond arithmetic, and it is large enough
  to see. If the rotation worked it would be the rare edge with no market risk attached:
  you are in cash either way. That is exactly why it is worth measuring honestly rather
  than asserting.

## Why the lag exists — the mechanism

- **Macaulay (1938)** and the modern textbook treatment in **Fabozzi, *Bond Markets,
  Analysis and Strategies*** — duration as the first-order sensitivity of price to yield.
  A portfolio of *n*-day bills has duration ≈ *n*/2 days, which is what makes SHV's
  realised 0.177 years and USFR's zero the *predicted* values rather than surprises.
- **Floating-rate note pricing.** A Treasury FRN's coupon resets against the 13-week bill
  auction each week, so its interest-rate duration collapses to the reset interval while
  its spread duration remains. See the US Treasury's FRN specification and **Duffie &
  Singleton (2003), *Credit Risk*** on the general reset mechanic. Our A1 measurement puts
  USFR's realised rate duration at −0.001 years (*t* = −0.04): a textbook zero.
- **Money-fund yield stickiness.** **Kacperczyk & Schnabl (2013), *How Safe Are Money
  Market Funds?*, Quarterly Journal of Economics** and the SEC's Rule 2a-7 disclosure
  regime document the WAM constraint that produces the lag mechanically. **Di Maggio,
  Kermani & Palmer (2020)** and the deposit-beta literature (**Drechsler, Savov &
  Schnabl (2017), *The Deposits Channel of Monetary Policy*, QJE**) study the far larger
  and far stickier version of the same effect in bank deposits — a useful contrast, since
  deposit betas are *chosen* while a bill fund's is *arithmetic*.

## Why the rotation fails anyway

- **Short-rate changes are close to a martingale.** The core reason the trade dies:
  knowing the last 21 days of the bill rate tells you little about the next 21.
  **Fama (1984), *The Information in the Term Structure*, Journal of Financial Economics**
  and **Duffee (2002), *Term Premia and Interest Rate Forecasts in Affine Models*, Journal
  of Finance** — beating a random walk at the short end is famously hard. Our lookback
  grid finds a maximum gross |*t*| of 0.53 across four windows.
- **The prize is smaller than the friction.** The entire dispersion between the four
  vehicles is ~16 bp/yr; one round trip at 2 bps a leg costs 4 bps. This is the same
  arithmetic that kills the ladder-versus-ETF trade in **Study 921**, and it is the reason
  the reversed rule and a turnover-matched random placebo lose almost exactly as much as
  the real rule — the loss is friction, not direction.
- **The trailing-window artefact.** Any "realised yield" built from a trailing *w*-day
  return carries a mechanical *w*-day averaging lag of its own. Our proxy-window sweep
  (5/10/21/42 days) separates the ruler from the vehicle and reports the smaller,
  honest figure (13-19 days at *w* = 5) rather than the flattering one.
- **Multiple looks.** **Harvey, Liu & Zhu (2016), *… and the Cross-Section of Expected
  Returns*, Review of Financial Studies** — with four pre-registered signal windows, a
  nominal *t* ≈ 2 on the best one would be unremarkable. We report the whole grid.

## Related desk studies (dedup)

- **[Study 921 — Bill-Ladder-vs-ETF](../../921-bill-ladder-vs-etf/)**: a *simulated*
  held-to-maturity 91-day bill ladder priced off ^IRX, versus the cash ETF — and it found
  the ladder's edge to be exactly the ETF's expense ratio. Study 923 never simulates a
  ladder: it measures the *repricing speed of the real, listed vehicles* against each
  other, and asks whether rotating between them pays. The two meet at one point — 921's
  "simply owning the cheaper fund erases it" is the same arithmetic as our SGOV-over-BIL
  finding — reached from opposite directions.
- **[Study 892 — Corporate-Bond-Ladder](../../892-corporate-bond-ladder/)**: ladder versus
  aggregate fund once duration is matched, in *credit*. Same ladder-folklore family, a
  different asset and a different question (hold-to-par accounting, not repricing speed).
- **[Study 922 — Floating-Rate Front End](../../922-frn-vs-fixed-front-end/)**: the
  closest neighbour by *instrument* — USFR and TFLO against BIL and SHY — but it **holds,
  never trades**, and asks an unconditional question (does the floating end out-pay the
  fixed end?) with a regime split for context. Study 923 asks whether you can *rotate*
  between those same vehicles on the rate's direction, and prices the turnover that
  requires. Their static USFR-over-BIL pickup (~15 bp/yr, no *t* above 1.1) is the same
  quantity as our B1 static arm (+16.7 bp/yr, *t* = +0.71), and the two agree.
- **[Study 925 — Front-End Trend](../../925-short-rate-momentum-switch/)**: the same
  *signal* — the sign of a trailing change in ^IRX, one execution lag — but pointed at a
  different pair: TLT (long duration) versus BIL. That study risks ~17 years of duration on
  the call; ours risks 0.18 of a year. Both find the signal empty, from opposite ends of
  the curve, which is a useful convergence rather than a duplication.
- **[Study 826 — Treasury-Duration-BAB](../../826-treasury-duration-bab/)**: betting
  against beta *along* the Treasury maturity curve, levered. Study 923 is unlevered,
  long-only, and confined to the 0-1 year bucket where the whole dispersion is basis
  points rather than percentage points.
- **[Study 924 — Cut-Cycle-Duration-Extension](../../924-cut-cycle-duration-extension/)**:
  the *event-study* cousin — extend duration when the Fed starts cutting. That study
  conditions on a hand-labelled list of four cut cycles; ours conditions on a continuous,
  mechanical signal (the sign of a trailing rate change) over 3,118 daily observations,
  and rotates only within cash.
- **[Study 380 — Curve-Roll-Down](../../380-curve-roll-down/)** and
  **[Study 132 — Yield-Curve-Steepener](../../132-yield-curve-steepener/)**: timing *long*
  duration off the curve's shape. Both take real interest-rate risk; this study takes
  ~18 basis points of it, on purpose.
- **[Study 581 — Term-Premium](../../581-term-premium/)**: forecasting long-bond returns
  from a term-premium estimate. The forecasting target and the instrument are both at the
  opposite end of the curve from ours.

## Method lineage

- **HAC / Newey-West.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica —
  [`strategy.newey_west_t`](../cash_lag/strategy.py) and
  [`strategy.hac_ols`](../cash_lag/strategy.py), used for both the distributed-lag
  regressions and every excess-return *t*.
- **Distributed-lag / pass-through regressions.** Almon (1965), *The Distributed Lag
  Between Capital Appropriations and Expenditures*, Econometrica — the design of
  [`strategy.lag_profile`](../cash_lag/strategy.py) (an unrestricted lag polynomial on a
  7-day grid, with the coefficient sum read as eventual pass-through).
- **Circular block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap*, JASA
  — [`strategy.block_bootstrap_mean_ci`](../cash_lag/strategy.py) and
  [`quantlab.stats`](../../../quantlab/stats.py).
- **Reproducibility stamp.** [`quantlab.repro`](../../../quantlab/repro.py) — the as-of
  slice and content fingerprint carried by every headline table.

## Data sources

- **BIL** (SPDR 1-3 Month T-Bill), **SGOV** (iShares 0-3 Month Treasury), **USFR**
  (WisdomTree Floating Rate Treasury), **SHV** (iShares Short Treasury) — daily
  **total-return** closes via `yfinance` (`auto_adjust=True`). Total return is not
  optional here: a cash ETF's entire return is its distribution, so a price-only series
  would be nearly flat and meaningless.
- **^IRX** — the 13-week Treasury bill **discount yield quote**, in percent. A quote, not
  an investable total return; it appears only on the right-hand side of the regressions
  and inside the direction signal, never as a return.
- **Non-tape inputs, all labelled.** The nominal WAMs (issuer fund pages) are used only to
  state the a-priori ordering. The 2 bps one-way cost is an assumption and is swept from 0
  to 10 bps. The expense-ratio commentary attached to the SGOV-over-BIL result is issuer
  disclosure, not a tape measurement, and nothing in the study's arithmetic depends on it.
- **As-of 2026-06-30.** The partial current month is dropped so the sample never creeps.
  The headline window starts at USFR's 2014-02-04 inception; SGOV, launched in 2020, is
  reported on its own window rather than truncating everything else to six years.
