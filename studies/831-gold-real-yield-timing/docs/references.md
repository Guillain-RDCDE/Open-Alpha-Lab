# References & literature map — Study 831 (Gold Real-Yield Timing)

## The claim, at full strength

- **Erb & Harvey (2013)**, *"The Golden Dilemma."* *Financial Analysts Journal* 69(4). The canonical
  modern teardown of gold's "fundamentals": gold has no cash flow, so its valuation rests on real
  rates, inflation hedging and safe-haven demand. Erb & Harvey stress that the widely-quoted
  gold-vs-real-yield relationship is empirically loose and *contemporaneous*, not a reliable forecast
  — exactly the distinction this study makes.
- **Baur & McDermott (2010)**, *"Is gold a safe haven? International evidence."* *Journal of Banking &
  Finance* 34(8). Establishes gold's negative co-movement with risk/real-rate shocks as a *contempor-
  aneous* safe-haven property, not a timing signal.
- **Barsky & Summers (1988)**, *"Gibson's Paradox and the Gold Standard."* *Journal of Political
  Economy* 96(3). The classic theoretical link between the real interest rate and the relative price
  of gold — the intellectual root of "gold ∝ −real yield."
- **Gorton & Rouwenhorst (2006)**, *"Facts and Fantasies about Commodity Futures."* *Financial
  Analysts Journal* 62(2). Frames commodities (gold included) as macro-hedges whose returns relate to
  real-rate and inflation states; the backdrop for a real-yield conditioning rule.
- **Bernanke (2013 speeches) / FOMC-era "taper tantrum" evidence.** The 2013 real-yield spike and gold
  crash is the single most-cited episode behind "rising real yields sink gold"; it is a *level-shock*
  co-movement, again contemporaneous.

## The real-yield estimate we build

- The official 10-year TIPS constant-maturity real yield (FRED **DFII10**) is not reachable from a
  no-key retail (Yahoo) stack, so this study builds two proxies. **Primary:** TIP total return as an
  inverse real-yield gauge — the iShares TIPS ETF price rises exactly when real yields fall, so
  ``ryfall = log(TIP_t) − log(TIP_{t−63})`` is a sign-flipped stand-in for the real-yield *change*
  (duration/offset constants drop out under ranking). **Secondary (cross-check):** the identity
  ``real ≈ nominal − breakeven``, with the 10y breakeven proxied by the relative total return of TIPS
  vs nominal Treasuries (``bei ∝ log TIP − log IEF``), giving a level proxy ``ry = TNX − 100·(log TIP −
  log IEF)``. Both are simplifications, **named on the SIGNAL axis**.

## Neighbours on this bench (the dedup map)

- **[Study 640 — Gold-Overnight](../../640-gold-overnight/)** — the *intraday-vs-overnight* return
  decomposition of GLD (a session/microstructure effect). Study 831 is a *macro* conditioning signal
  (the real-yield trend) on gold's total daily return — different axis entirely.
- **[Study 649 — Gold-Seasonality](../../649-gold-seasonality/)** — a *calendar* signal on the same GLD
  instrument (month-of-year effects). Orthogonal to the fundamental real-yield signal here.
- **[Study 381 — TIPS-Breakeven](../../381-tips-breakeven/)** — trades the *breakeven-inflation* spread
  (TIP vs nominal Treasuries) as the signal-and-instrument. Study 831 *uses* the same TIP/IEF pair only
  to **build a real-yield gauge for timing a third asset (gold)**; the traded thing is GLD, not the
  breakeven.
- **[Study 580 — Gold-Lease-Rate](../../580-gold-lease-rate/)** — conditions gold on the *gold-lease /
  carry* market (a supply-side funding signal). Study 831 conditions on the *real interest rate* — the
  demand-side opportunity-cost signal. Complementary, not the same.

## Shared method

- **Newey & West (1987)** — the HAC (heteroskedasticity- and autocorrelation-consistent) *t*-stat used
  on the Q5−Q1 forward-return spread and on the contemporaneous inverse-link beta; essential when
  forward returns overlap.
- **Block / circular-shift permutation testing** (Politis & Romano 1994; Good 2005) — the placebo null:
  rotate the forward-return series in blocks against the real-yield signal and read the spread's tail
  probability, preserving overlap-induced autocorrelation.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (HAC *t* ≥ 2 on the
  real tape for `REAL`; a significant *wrong-sign* result reads `NONE`), one execution lag (signal at
  close *t*, hold *t+1*), costs one-way × NAV per switch, and seed-robust synthetic controls (≥ 20
  seeds).
