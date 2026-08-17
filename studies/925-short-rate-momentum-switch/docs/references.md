# References & literature map — Study 925 (Front-End Trend)

## The claim under test

- **The front end trends, so let it pick your duration.** Policy rates do not wander:
  central banks move in *cycles*, hiking or cutting for quarters at a time, and the
  13-week bill yield inherits that persistence almost by construction. The folk rule that
  falls out of this is disarmingly simple — if the 3-month bill yield has fallen over the
  last three months, own long duration (TLT); if it has risen, sit in bills (BIL). Rates
  falling is the tailwind for a long bond; rates rising is the headwind; the front end
  tells you which regime you are in before the long end has finished pricing it.
- **The steelman.** The rule needs neither a forecast nor a term-structure model. It only
  needs *autocorrelation in rate changes*, which the Fed's own reaction function supplies:
  once a cutting cycle starts it rarely reverses within a quarter. And the instrument is
  the cheapest, most liquid duration in the world. That is a genuinely testable mechanical
  claim, and this study tests it on the real, costed, excess-of-cash tape.

## Why the rule *can* work — the mechanism

- **Time-series momentum.** Moskowitz, Ooi & Pedersen (2012), *Time Series Momentum*,
  Journal of Financial Economics 104(2) — an asset's own past 12-month return predicts its
  next month across 58 instruments, bond futures included. Hurst, Ooi & Pedersen (2017),
  *A Century of Evidence on Trend-Following Investing*, Journal of Portfolio Management.
  The `^IRX` change rule is the single-instrument, single-signal version of that idea,
  applied to the yield rather than the price.
- **Monetary-policy inertia.** Rudebusch (2002), *Term Structure Evidence on Interest Rate
  Smoothing and Monetary Policy Inertia*, Journal of Monetary Economics — policy rates
  adjust partially and persistently, which is exactly the autocorrelation a trend rule
  needs. Coibion & Gorodnichenko (2012), *Why Are Target Interest Rate Changes So
  Persistent?*, AEJ: Macroeconomics, quantifies how much of that persistence is real
  smoothing versus persistent shocks.
- **The front end leads the long end.** Estrella & Mishkin (1998), *Predicting U.S.
  Recessions: Financial Variables as Leading Indicators*, Review of Economics and
  Statistics, and Cochrane & Piazzesi (2005), *Bond Risk Premia*, American Economic Review
  — short-rate dynamics carry information about the whole curve, so a front-end signal is
  not obviously the wrong place to look for a duration timer.
- **Bond-market trend followers exist and are studied.** Ilmanen (2011), *Expected
  Returns*, ch. 9 (duration timing) surveys the practitioner rules; Baltas & Kosowski
  (2013), *Momentum Strategies in Futures Markets and Trend-Following Funds*, examine how
  much of managed-futures performance is trend on fixed income.

## Why it can fail

- **Rate *changes* are close to a martingale even when rate *levels* are persistent.** A
  persistent level is not a predictable increment. The efficient-markets read — Fama
  (1984), *The Information in the Term Structure*, JFE — is that the forward curve already
  embeds the expected path of the front end, so a publicly visible three-month change
  carries no incremental information about the next three months. Our synthetic control
  isolates precisely this distinction: the knob is AR(1) persistence *in the increments*,
  with the level's volatility held fixed.
- **Zakamulin (2014), *The Real-Life Performance of Market Timing with Moving Average and
  Time-Series Momentum Rules*, Journal of Asset Management** — once the cash leg is
  credited honestly and time-in-market accounted for, moving-average and momentum timing
  advantages shrink to statistical invisibility. Our excess-of-cash construction is the
  same correction, and it produces the same collapse.
- **The random-control turnover trap.** A frequency-matched random control switches about
  `2p(1−p)N` times — here ~2,400 against the rule's 321 — so at any positive cost it is
  handicapped by churn it never chose. Reporting "we beat a random control" *net only* is
  therefore not evidence of timing skill. Study 912 uses the same control on gold; this
  study adds the seed sweep and the gross-versus-net split that shows how easily that
  comparison misleads (see `strategy.random_control_sweep`).
- **The repair: a matched constant-weight blend.** The clean exposure control holds the
  rule's own in-market fraction as a *fixed* weight (here 47.8% TLT / 52.2% bills,
  rebalanced daily) — same average duration, **no regime flips**, so no friction handicap
  and no seed to draw (`strategy.constant_weight_control`, `strategy.matched_blend_race`).
  It is the practitioner's "just hold the average" benchmark and the natural fixed-weight
  analogue of the Zakamulin correction below; the rule must beat it before any timing claim
  can be made. On the planted synthetic world it separates cleanly (adv +1.55, *t* +4.45),
  so the control has power — it is not simply insensitive.
- **Two-event samples.** The rule's whole *timing* record is 2022 (out of duration for all
  251 sessions) and late 2023 (37 sessions long while TLT rose +14.7%); its other large
  winning years are long-duration beta, not timing. `docs/results.md` prints **all twenty**
  calendar years rather than a chosen subset, because a hand-picked year table can be made
  to argue either side. Harvey, Liu & Zhu (2016),
  *…and the Cross-Section of Expected Returns*, Review of Financial Studies, on the
  multiple-testing bar such episode-driven records must clear.

## Related desk studies (dedup)

- **[Study 829 — Global-Sovereign-Bond-Momentum](../../829-global-sovereign-bond-momentum/)**:
  12-1 *price* momentum across a cross-section of foreign government-bond ETFs. Study 925
  is single-market, and the signal is the **short-rate yield change**, not a bond price
  trend — the front end reading, not the instrument's own trend.
- **[Study 132 — Yield-Curve-Steepener](../../132-yield-curve-steepener/)** and
  **[Study 380 — Curve-Roll-Down](../../380-curve-roll-down/)**: both time duration on the
  curve's **level/slope** (10y−3m, roll+carry). Those are *level* signals. Study 925 uses
  the **change** in the front end — a trend, not a state.
- **[Study 864 — Yield-Curve-Twist](../../864-yield-curve-twist/)**: the curve's third
  factor (curvature/butterfly) as a *belly* signal. Different factor, different instrument.
- **[Study 924 — Cut-Cycle-Duration-Extension](../../924-cut-cycle-duration-extension/)**:
  buys duration on a **hardcoded list of first-cut FOMC dates** — an event study with five
  observations and a non-tape event list. Study 925 is the continuous, always-on version
  of the same intuition with **no assumed dates at all**: the tape decides every day
  whether rates are falling.
- **[Study 826 — Treasury-Duration-BAB](../../826-treasury-duration-bab/)**: a *static*
  cross-maturity beta bet, no timing.
- **[Study 912 — Gold-Trend-Managed](../../912-gold-trend-managed/)**: the same binary
  in/out-of-cash architecture and random control, applied to gold's own 200-day trend.
  Same machinery, entirely different signal and asset.
- **[Study 518 — Time-Series-Momentum](../../518-time-series-momentum/)** and
  **[Study 427 — Rate-of-Change](../../427-rate-of-change/)**: the generic trend
  primitives. Study 925 is not a re-run of them: the input is a *yield level from `^IRX`*,
  a non-tradable series that never enters any return calculation.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica —
  [`strategy.newey_west_t`](../front_end_trend/strategy.py) and
  [`quantlab.analytics`](../../../quantlab/analytics.py). Every *t* the verdict rests on is
  HAC, including the day-level bucket spread: because the buckets are selected by a signal
  that persists for months, the iid Welch statistic is not the right one and is printed for
  reference only. The HAC spread *t* is the Newey-West *t* of `e·(s − p)/(p(1−p))`, whose
  mean equals the difference in bucket means — the dummy-regression contrast with HAC
  errors (`strategy.conditional_day_test`, key `spread_hac_t`).
- **Return-difference (Sharpe comparison) test.** Jobson & Korkie (1981), *Performance
  Hypothesis Testing with the Sharpe and Treynor Measures*, Journal of Finance, with
  Memmel (2003)'s correction — [`strategy.sharpe_diff_tstat`](../front_end_trend/strategy.py).
- **Circular / stationary block bootstrap.** Politis & Romano (1994), *The Stationary
  Bootstrap*, JASA — [`strategy.bootstrap_sharpe_ci`](../front_end_trend/strategy.py) and
  [`strategy.bootstrap_diff_ci`](../front_end_trend/strategy.py) (paired blocks, so the
  two arms share resampled dates).
- **Reproducibility stamp.** [`quantlab.repro`](../../../quantlab/repro.py) — as-of slicing
  plus a content fingerprint on the exact input frame.

## Data sources & the price-only / total-return line

- **SHY (1-3y), IEF (7-10y), TLT (20y+), BIL (1-3 month bills)** — daily **total-return**
  closes via `yfinance` (`auto_adjust=True`), cached under `studies/_cache`. Coupons are
  most of a Treasury ETF's return, so a price-only series would misstate every arm.
- **`^IRX`** — the CBOE 13-week Treasury bill **discount yield, in percent**. This is a
  **price-only yield level**, not a tradable instrument and not a return series. It enters
  the study **only** through `strategy.rate_change` to form the signal; every return in
  every table comes from an ETF total-return close.
- **Survivorship.** None of the five legs is a survivorship-prone universe: they are four
  named, still-listed ETFs and one index yield, all held for the whole window. There is no
  cross-section to select from and no delisted constituent to omit — the usual
  survivorship channel is absent here by construction.
- **Non-tape inputs.** There are none beyond the cost assumption. The 2 bps one-way switch
  cost is an **ASSUMPTION** (a fair estimate for penny-wide, multi-billion-dollar TLT and
  BIL) and is swept 0 → 25 bps in `docs/results.md`. There is no short leg anywhere, so no
  borrow rate is assumed or paid. The 63-day lookback and the 2016 era split are the two
  discretionary choices, and both are swept.
- **As-of 2026-06-30.** The partial current month is dropped so the sample never creeps.
  The window starts at BIL's 2007 inception — the first date on which the cash leg of this
  race actually existed as a tradable fund.
