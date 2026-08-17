# References & literature map — Study 949 (Riding the TIPS Curve)

## The claim under test

- **The roll-down pitch.** Sell-side and advisory notes on inflation-linked bonds argue
  that the *real* yield curve, like the nominal one, is normally upward-sloping, so a
  linker held for a year ages down the curve and is repriced at a lower real yield — a
  capital gain on top of the running real coupon. The retail form of the pitch is
  "extend from short TIPS into intermediate or long TIPS and you pick up the roll".
  If true, the longer maturity buckets should out-earn cash by more than the short one,
  and the part of a linker's return not explained by duration should be reliably positive.
- **The steelman.** Roll-down is not a forecast: it is arithmetic on a static curve, the
  same arithmetic Study 380 tested on nominal Treasuries. It fails only if the curve
  *moves* — and over a long enough sample the moves are supposed to wash out, leaving the
  slope you were paid for.
- **The rival hypothesis this study cannot rule out.** A long-linker / short-nominal
  residual is mechanically a **long-breakeven** position: its expected return is roll-down
  in real yields *plus* (realised inflation − the breakeven priced at entry). Fund total
  returns cannot separate the two. This is stated in the README, in `strategy.py` and in
  the results, and it is why the era cut around 2021-2023 carries the argument.

## Roll-down, carry, and the term premium

- **Litterman & Scheinkman (1991), _Common Factors Affecting Bond Returns_, Journal of
  Fixed Income** — level, slope and curvature. The level factor dominates bond-fund
  covariance, which is precisely why a single duration-matched hedge (this study's
  `hedged_sleeve`) removes 30-70% of a linker's variance and why what is left is small.
- **Koijen, Moskowitz, Pedersen & Vrugt (2018), _Carry_, Journal of Financial
  Economics** — carry as the return under an unchanged price/curve, across asset classes,
  including fixed income. The desk's cross-sectional version of this is
  [Study 868 — Global Curve-Slope Carry](../../868-global-curve-slope-carry/) (None /
  Mirage); Study 949 is the *within-real-curve*, time-series form.
- **Ilmanen (2011), _Expected Returns_, Wiley, ch. 9** — the bond term premium is small,
  time-varying and often negative; "riding the curve" is the term premium wearing a
  carry costume. Our ladder race, in which real duration earned less than the T-bill leg
  over 13.5 years, is a clean instance.
- **Cochrane & Piazzesi (2005), _Bond Risk Premia_, American Economic Review** — expected
  excess bond returns are predictable and strongly time-varying, so any single-sample
  estimate of "the carry you get for extending" is a statement about one regime. Our era
  cut makes that concrete.

## Inflation-linked bonds specifically

- **Campbell, Shiller & Viceira (2009), _Understanding Inflation-Indexed Bond Markets_,
  Brookings Papers on Economic Activity** — the anatomy of TIPS returns: real yield,
  inflation accrual, and a liquidity/illiquidity component that was very large in
  2008-2009. Their decomposition is the reason a TIPS-minus-nominal residual must not be
  read as pure roll-down.
- **Fleckenstein, Longstaff & Lustig (2014), _The TIPS-Treasury Bond Puzzle_, Journal of
  Finance** — TIPS were persistently cheap relative to nominal Treasuries plus inflation
  swaps, sometimes by more than 200 bps. A relative-value premium of this kind would show
  up in exactly the residual this study measures — another reason a positive residual is
  not evidence of *roll-down*.
- **Pflueger & Viceira (2016), _Return Predictability in the Treasury Market: Real Rates,
  Inflation, and Liquidity_ (in _Handbook of Fixed-Income Securities_)** — breakeven and
  TIPS excess returns are predictable by liquidity and inflation-risk proxies, with the
  predictability concentrated in stress regimes. Consistent with our finding that the
  residual lives entirely in and after 2021.
- **D'Amico, Kim & Wei (2018), _Tips from TIPS: The Informational Content of Treasury
  Inflation-Protected Security Prices_, Journal of Financial and Quantitative Analysis** —
  the liquidity premium embedded in TIPS yields, and its decay after 2004. Relevant to
  our inception-gated window, which begins in 2012 and so excludes the crisis-era
  dislocation entirely.

## Why the era cut is the decisive test here

- The 2021-2023 US inflation shock was, by construction, a period in which **realised**
  inflation exceeded the breakeven priced at entry. A long-breakeven position had to pay
  then, whatever the real curve's slope was doing. Our VTIP−SHY sleeve made +2.36%/yr net
  in 2021-2023 against **+0.18%/yr gross / −0.20%/yr net** across the **7.2 preceding
  years** (2013-10-21 → 2020-12-31, 1,813 trading days — the 252-day beta warmup eats the
  first year, so it is not "eight"), and the whole full-sample result evaporates when 2021
  alone is removed. Roll-down carry does not switch on and off with an inflation surprise;
  a breakeven leg does.
- This is also the *only* place in the study where a HAC *t* reaches 2: the VTIP−SHY
  **gross** carry inside that shock window is +2.81%/yr at *t* = **+2.04**. It is reported
  rather than suppressed, and it argues *for* the rival hypothesis, not against it — a
  roll-down carry that exists only inside the one CPI surprise in the sample is a
  breakeven leg by another name. One hit in 56 tests, against a Bonferroni bar of ≈3.2.

## Related desk studies (dedup)

- **[Study 380 — Curve-Roll-Down](../../380-curve-roll-down/)**: the same roll-down claim
  on the **nominal** Treasury curve, built from constant-maturity yields (`^IRX`/`^FVX`/
  `^TNX`/`^TYX`) with a promised-vs-realized decomposition. Study 949 is the **real-yield**
  counterpart and uses an entirely different instrument set and method: live linker ETFs,
  an excess-of-cash ladder race, and a *duration-hedged residual* against nominal funds.
  380 asks "does riding the nominal curve beat cash?"; 949 asks "is any of the linker's
  return left once you take the duration away?".
- **[Study 381 — TIPS-Breakeven](../../381-tips-breakeven/)**: uses `log(TIP/IEF)` as a
  **predictive signal** for forward returns of TIPS, equities and gold. Study 949 uses the
  same instrument family but does no forecasting at all — it is a *return decomposition*,
  not a timing rule, and its residual is the thing 381 treats as a regressor.
- **[Study 868 — Global Curve-Slope Carry](../../868-global-curve-slope-carry/)**: a
  **cross-sectional** carry sort across countries' nominal curves. 949 is single-country,
  single-curve, and about the real curve.
- **[Study 886 — Agency-MBS-Carry](../../886-agency-mbs-carry/)** and
  **[Study 906 — EM Local Bonds FX-Hedged](../../906-em-local-hedged/)**: the same
  *hedge-out-the-obvious-factor-and-see-what-is-left* method applied to prepayment risk
  and to FX. 949 applies it to the real/nominal split.
- **[Study 921 — Bill Ladder vs ETF](../../921-bill-ladder-vs-etf/)**: the cash leg's own
  study; it is why this desk uses BIL's actual total return rather than a flat rate proxy
  when it races anything excess-of-cash.
- **[Study 152 — Inflation-Hedge](../../152-inflation-hedge/)** and
  **[Study 69 — Safe-Haven](../../69-safe-haven/)**: whether *other* assets hedge
  inflation. 949 takes the instrument that hedges it by contract and asks what it costs.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), _A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix_, Econometrica —
  [`strategy.newey_west_t`](../tips_roll/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py). Bond-fund daily
  returns carry stale-mark autocorrelation, so the naive *t* is not admissible here.
- **Circular block bootstrap.** Politis & Romano (1994), _The Stationary Bootstrap_,
  JASA — [`strategy.bootstrap_mean_ci`](../tips_roll/strategy.py) and
  [`strategy.bootstrap_sharpe_ci`](../tips_roll/strategy.py); 21-day blocks preserve the
  month-scale persistence of rate moves.
- **Reproducibility stamp.** [`quantlab.repro`](../../../quantlab/repro.py) — the as-of
  slice and content fingerprint printed above every table in
  [`docs/results.md`](results.md).

## Data sources

- **VTIP, SCHP, TIP, LTPZ** (inflation-linked buckets), **SHY, IEI, IEF, TLT** (nominal
  Treasury duration matches), **BIL** (the 1-3 month T-bill cash leg) — daily
  **total-return** closes via `yfinance` (`auto_adjust=True`). Total return is
  non-negotiable for this study: a linker's inflation accrual reaches the holder as a
  distribution, so price-only closes would understate every linker leg and bias the
  residual *downwards* — the opposite of the direction that would flatter our conclusion.
- **As-of 2026-06-30**, the last complete month; the partial current month is dropped so
  the sample cannot creep between reruns. The common window 2012-10-16 → 2026-06-30 is
  gated by VTIP's inception, which confines the study to the post-GFC real-rate regime —
  named on the Signal axis as the sample's principal limitation.
- **The published fund durations** quoted in the tables come from the sponsors' fact
  sheets and are **labels, not inputs**: every hedge ratio in the study is estimated from
  returns, and the pairing choice is swept.
