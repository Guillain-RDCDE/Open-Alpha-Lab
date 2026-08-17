# References & literature map — Study 918 (Creation Halt)

## The claim under test

- **The creation-halt folklore.** An exchange-traded product is pinned to the value of
  what it holds by one mechanism only: authorised participants create new shares when the
  price is rich and redeem them when it is cheap. Suspend the creation side and that
  arbitrage is switched off *in one direction* — supply is frozen while demand is not, so
  the price should float up to a premium, hold it while the suspension lasts, and collapse
  when issuance resumes. Suspend the *redemption* side instead (the pre-2024 Grayscale
  case) and the identical logic runs in reverse: the price can only drift to a discount.
  The trading claim built on top of this is that a publicly announced suspension is a buy
  signal for the capped fund against an uncapped equivalent.
- **The steelman.** This is not a forecast of the underlying. It is a claim about a
  *mechanical* break in a no-arbitrage relationship, with a dated announcement and a dated
  resumption — the cleanest possible event study, if the events exist and a clean ruler
  exists to measure against.

## The events themselves (primary, publicly verifiable)

- **UNG, July 2009.** The United States Natural Gas Fund stopped issuing creation baskets
  after exhausting its registered shares, while the CFTC was consulting on speculative
  position limits in energy futures; the fund traded at a double-digit premium to its net
  assets for most of the quarter before creations resumed in late September. See the
  CFTC's 2009 *Energy Position Limits and Hedge Exemptions* hearings and the fund's
  registration filings. The resumption date used here is our public reading and is
  labelled an ASSUMPTION and swept.
- **USO, April 2020.** The United States Oil Fund announced it had used all registered
  shares and suspended creations the day after WTI's 2020-04-20 settlement at −$37.63, in
  the middle of a forced restructuring of its roll schedule. See the fund's 8-K/424(b)
  filings of April 2020 and Study **661-uso-roll-decay** on this desk for the roll side.
- **VXX and OIL, March 2022.** Barclays suspended further sales and issuances of the iPath
  Series B S&P 500 VIX Short-Term Futures ETN (VXX) and the iPath Pure Beta Crude Oil ETN
  (OIL) on 2022-03-14, having issued beyond its registered amount, and reopened issuance
  in August 2022. Both dates are FIRM. VXX traded at a large, widely reported premium to
  its indicative value throughout.
- **GBTC, 2015–2024.** The Grayscale Bitcoin Trust had no redemption programme for its
  entire pre-ETF life (a consequence of the 2016 SEC action over unregistered
  redemptions), which is why its discount could persist; creations and redemptions both
  switched on at the 2024-01-11 ETF conversion.
- **BITO, late 2021.** ProShares' futures-based bitcoin ETF ran into CME position limits
  within weeks of launch and moved into later-dated contracts. This was a **capacity
  constraint, not a formal suspension**; it is carried here flagged SOFT with both dates
  ASSUMPTIONs, and it is jackknifed.
- **Not testable, and named for honesty.** The TVIX creation suspension of February 2012 —
  the most cited case of all — and the original 2009–2018 VXX ETN cannot be tested: both
  instruments were redeemed and delisted and no continuous free tape survives. Their
  absence biases this study's event list toward the survivors.

## Why the mechanism should work — the method literature

- **Limits of arbitrage.** Shleifer & Vishny (1997), *The Limits of Arbitrage*, Journal of
  Finance — mispricings persist exactly where the arbitrage capital or the arbitrage
  *mechanism* is constrained. A creation halt is the cleanest possible instance: the
  constraint is announced, dated and total.
- **Short-sale constraints and overpricing.** Miller (1977), *Risk, Uncertainty and
  Divergence of Opinion*, Journal of Finance; Lamont & Thaler (2003), *Can the Market Add
  and Subtract?*, Journal of Political Economy — when the supply of a security cannot
  expand and it cannot be cheaply shorted, the optimists set the price. Precisely the
  regime a capped ETP enters.
- **ETF arbitrage and premiums.** Petajisto (2017), *Inefficiencies in the Pricing of
  Exchange-Traded Funds*, Financial Analysts Journal; Ben-David, Franzoni & Moussawi
  (2017), *Do ETFs Increase Volatility?*, Journal of Finance — ETF prices deviate from NAV
  when the creation/redemption channel is impaired, and the deviation mean-reverts when it
  is restored.
- **Closed-end funds as the limiting case.** Lee, Shleifer & Thaler (1991), *Investor
  Sentiment and the Closed-End Fund Puzzle*, Journal of Finance — a fund with no
  creation/redemption channel at all can sit at a persistent discount or premium for
  years. A creation halt turns an ETP into a temporary closed-end fund; the GBTC event is
  that case for nearly nine years.

## Method lineage

- **Event study with a firm-specific placebo null.** Brown & Warner (1985), *Using Daily
  Stock Returns*, Journal of Financial Economics — the standardisation of an event-window
  cumulative return by the same asset's own non-event dispersion, used here in its
  non-parametric form (the empirical percentile among all other K-day windows).
- **Overlapping windows are not independent draws.** Hansen & Hodrick (1980),
  *Forward Exchange Rates as Optimal Predictors*, Journal of Political Economy; Richardson
  & Stock (1989), *Drawing Inferences from Statistics Based on Multiyear Asset Returns*,
  Journal of Financial Economics — a rolling k-day sum overlaps its neighbours by
  `(k−1)/k`, so a pool of `n` such windows carries roughly `n/k` independent observations.
  `event_car` therefore reports `pct_indep` on a **non-overlapping** pool alongside the
  overlapping one: VXX's 20-day announcement percentile is 0.990 of 99 independent
  windows, not 0.999 of 1,978.
- **Data snooping across the design's looks.** White (2000), *A Reality Check for Data
  Snooping*, Econometrica; Harvey, Liu & Zhu (2016), *… and the Cross-Section of Expected
  Returns*, Review of Financial Studies — five events × three horizons × two legs = 30
  placebo percentiles are inspected before a winner is quoted, so `family_size` and
  `bonferroni_percentile` print the family-wise bar (0.99833) next to the result. The
  study's single best look does not clear it; the joint announce-and-fade pattern does.
- **HAC / Newey-West *t*.** Newey & West (1987), *A Simple, Positive Semi-Definite …
  Covariance Matrix*, Econometrica — [`strategy.newey_west_t`](../creation_halt/strategy.py)
  and [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Circular block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap*, JASA —
  [`strategy.block_bootstrap_mean_ci`](../creation_halt/strategy.py). The per-event
  resampling used for the pooled statistic (``event_resample_ci``) treats the *event*, not
  the day, as the sampling unit — with six events the honest interval is very wide.
- **Reproducibility stamp.** [`quantlab.repro`](../../../quantlab/repro.py) — the as-of and
  the input fingerprint printed above every headline table.

## Related desk studies (dedup)

- **[618 — GBTC Premium Cycle](../../618-gbtc-premium-cycle/)**: reconstructs GBTC's
  premium/discount to modelled bitcoin-per-share across its whole life. Study 918 does not
  re-derive that path; it treats the no-redemption era as *one signed event* in a
  cross-instrument list and asks whether the halt mechanic generalises beyond it.
- **[378 — ETF NAV Premium](../../378-etf-nav-premium/)**: does a below-NAV discount close,
  on ETFs whose creation channel is working normally? The opposite regime — 918 studies
  the days the channel is switched *off*.
- **[367 — CEF Discount](../../367-closed-end-fund-discount/)**: funds with no
  creation/redemption channel by design; 918's events are the temporary version.
- **[661 — USO Roll Decay](../../661-uso-roll-decay/)** and
  **[375 — VXX Roll Decay](../../375-vxx-roll-decay/)** and
  **[619 — BITO Roll Drag](../../619-bito-roll-drag/)**: the *roll* cost these same
  instruments pay when nothing unusual is happening. 918 measures the orthogonal thing —
  price versus an uncapped twin around dated issuance suspensions — and the roll is the
  confound that makes the curve-mismatched rulers useless.
- **[917 — NAV Staleness](../../917-nav-staleness-timezone/)**: a mechanical price-versus-
  fair-value gap driven by clocks, not by a suspended primary market.

## Data sources

- **Capped instruments:** UNG, USO, VXX, OIL, GBTC, BITO. **Uncapped rulers:** VIXY (same
  short-dated VIX-futures index as VXX), BTC-USD (the asset GBTC holds), DBO / USL / BNO
  (WTI and Brent futures ETFs that were never suspended), NG=F (front-month natural gas).
  **Cash reference:** BIL. Daily closes via `yfinance` with `auto_adjust=True` for every
  ETF/ETN — **total return**; `NG=F` is a continuous futures print and is **price-only**.
- **Non-tape inputs, all labelled.** The event list itself, the direction sign, the
  resumption dates flagged APPROX, the **published expense ratios** in `data.FEE_ANNUAL`,
  the 3%/yr baseline borrow and the 10 bps one-way cost are **ASSUMPTIONS**, not
  measurements. Each is swept: dates in `resume_date_sweep`, borrow and commission in
  `trade_sweep`, event membership in `jackknife`, and the resumption date is removed
  entirely by the blind fixed-horizon exit (`trade_event(hold="blind")`).
- **Three contaminants the audit made explicit.** (1) A fund's total return is net of its
  fee, so the fund-minus-ruler spread carries `(fee_ruler − fee_fund)/252` per day before
  any halt effect exists; against an unfeed ruler this does not cancel, and GBTC's 2.00%/yr
  against spot bitcoin is +0.79 bps/day — 18% of its measured drift (`fee_drag_bps`).
  (2) The `hold="halt"` trade exits on the resumption date, which is a **hindsight exit**;
  the blind variant is the rule a trader could have followed, and it takes VXX's net from
  +17.80% to +0.57%. (3) `exp(Σx)` is the return of a continuously dollar-neutral
  position, so the trade pays a **daily rebalancing charge** of `cost_bps × Σ|xₜ|` on top
  of the entry/exit crossings — 5.73% rather than 0.40% over GBTC's 2,183 sessions.
- **GBTC's "halt" date is the start of the free tape, not an announcement.** The
  redemption programme was already suspended before the shares were publicly quoted, so
  the pair has no pre-halt control window at all: its 597 "outside" sessions are entirely
  post-conversion. The 2024-01-11 resumption date is firm; the 2015-05-11 start is not a
  filing date and is labelled as such.
- **As-of 2026-06-30.** The partial current month is dropped so the sample never creeps.
