# References & literature map — Study 958 (Spot ETF Basis)

## The claim under test

- **The compression thesis.** Before January 2024 the only US-listed, 40-Act route to
  bitcoin exposure ran through CME futures (BITO, from October 2021) or a closed-end
  trust (GBTC). The standard explanation for the persistently rich CME front basis was
  *segmented demand*: leveraged and access-constrained longs bid the futures curve into
  contango because the cash market was closed to them, while the arbitrageurs who would
  normally sell that basis faced balance-sheet, custody and financing frictions. The
  natural prediction is that once a physically-backed spot wrapper exists — one that
  creates and redeems in size at NAV — the constrained demand migrates to spot and the
  futures basis **compresses**. This study asks the tape whether it did, dating the event
  at the first spot-ETF session, **2024-01-11**.
- **The steelman for "no compression".** The basis is a *financing* rate, not an access
  premium. As long as levered longs pay up for exposure and the balance sheet that sells
  the basis is scarce and expensive, the carry survives whatever wrappers exist — and may
  even widen when a bull market brings fresh levered demand faster than fresh arbitrage
  capital. On this reading the spot ETFs change *who holds the spot leg*, not the price of
  the carry.

## Why a futures wrapper bleeds at all — the mechanism

- **Cost of carry.** Keynes (1930), *A Treatise on Money*, and Kaldor (1939), *Speculation
  and Economic Stability* — the futures-spot relation as storage cost, financing and
  convenience yield. For a non-perishable, costlessly stored digital asset the carry
  collapses to financing plus a risk premium, which is exactly what the implied basis in
  this study measures.
- **Roll yield in a contango curve.** Erb & Harvey (2006), *The Strategic and Tactical
  Value of Commodity Futures*, Financial Analysts Journal; Gorton & Rouwenhorst (2006),
  *Facts and Fantasies about Commodity Futures* — a fully collateralised long futures
  position in persistent contango earns spot return minus the basis plus collateral yield.
  Our decomposition `basis = cash − fee − drag` is that identity, inverted.
- **The crypto cash-and-carry specifically.** Makarov & Schoar (2020), *Trading and
  Arbitrage in Cryptocurrency Markets*, Journal of Financial Economics — large, persistent
  crypto arbitrage spreads sustained by capital controls, custody frictions and
  counterparty risk rather than by mispricing per se. Hazlett & Luther (2020) and the CME
  basis literature document the same premium in the regulated futures venue.

## Why the launch might (or might not) compress it

- **Limits to arbitrage.** Shleifer & Vishny (1997), *The Limits of Arbitrage*, Journal of
  Finance — a spread survives when the capital that would close it is scarce, financed
  and risk-averse. The spot ETFs added a *creation/redemption* channel for holding
  bitcoin; they did not add balance sheet willing to be short CME futures against it.
- **Segmented-demand pricing.** Gromb & Vayanos (2010), *Limits of Arbitrage*, Annual
  Review of Financial Economics — clientele segmentation as the source of persistent
  wedges between economically identical claims. The pre-2024 story was pure segmentation;
  our matched-window null is evidence that segmentation was not the binding constraint.
- **Event dating and placebo discipline.** Bertrand, Duflo & Mullainathan (2004), *How
  Much Should We Trust Differences-in-Differences Estimates?*, Quarterly Journal of
  Economics — serially correlated outcomes make a single before/after break look far more
  significant than it is. Our placebo sweep over 44 arbitrary split dates is the direct
  application: the launch date's break statistic ranks seventh, so the "significant" full
  sample break is a property of a wandering series, not of the event.

## Measurement — tracking difference and the timestamp problem

- **Tracking difference vs tracking error.** Elton, Gruber, Comer & Li (2002), *Spiders:
  Where Are the Bugs?*, Journal of Business, and Petajisto (2017), *Inefficiencies in the
  Pricing of Exchange-Traded Funds*, Financial Analysts Journal — the standard framework
  for measuring what a wrapper costs its holder relative to the thing it wraps.
- **Non-synchronous prices.** Scholes & Williams (1977), *Estimating Betas from
  Nonsynchronous Data*, Journal of Financial Economics; Lo & MacKinlay (1990) — comparing
  series stamped at different hours injects a large, mean-reverting error. Bitcoin quoted
  at 00:00 UTC against ETFs marked at 16:00 New York is a severe case, which is why this
  study estimates the drag as an all-observation **trend slope** rather than the
  two-endpoint mean of daily differences, and calibrates that estimator on a *known* cost
  (the spot wrappers' own expense ratios).
- **HAC inference.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica —
  [`strategy.hac_ols`](../etf_basis/strategy.py), reported across bandwidths 20-250
  because the residual basis wanders at multi-month scale.
- **Why a trend *t* is not enough, and the blunt cross-check.** Granger & Newbold (1974),
  *Spurious Regressions in Econometrics*, Journal of Econometrics, and Phillips (1986) —
  regressing a persistent level on time produces flattering *t* statistics that a HAC
  covariance mitigates but does not cure (on this tape the IBIT-referenced residual has an
  AR(1) of 0.98 and a Dickey-Fuller *t* of only −3.0, i.e. borderline). Every headline is
  therefore repeated on [`strategy.monthly_drag`](../etf_basis/strategy.py): the mean of
  **non-overlapping month-end gaps** with an ordinary *t* and no HAC assumption at all. It
  is much less efficient by construction — which is the point. The same overlap logic
  governs [`strategy.cycle_regression`](../etf_basis/strategy.py), where the 126-day
  windows leave only ~8 independent observations and the regression is reported as
  descriptive, never certified.
- **Circular block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap*, JASA —
  [`strategy.bootstrap_drag_ci`](../etf_basis/strategy.py) and
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).

## Related desk studies (dedup)

- **[Study 619 — BITO Roll Drag](../../619-bito-roll-drag/)**: measures the *level* of the
  futures wrapper's toll and where in the month it is paid (answer: everywhere, not in the
  roll week), using the CME front basis from `BTC=F` and a monthly-aggregation ruler.
  Study 958 does not re-litigate the level — it asks the **event question 619 leaves
  open**: did the January-2024 spot ETFs *change* it? That needs a different estimator (a
  broken-trend slope with a placebo date sweep rather than a mean of monthly gaps), a
  different ruler calibration (the spot wrappers' own fees, read to the basis point), and
  a different tape (no `BTC=F`; IBIT *and* FBTC as independent spot references).
- **[Study 618 — GBTC Premium Cycle](../../618-gbtc-premium-cycle/)**: the *other*
  pre-2024 wrapper wedge — a closed-end trust's premium and discount to NAV, which the
  same 2024-01-11 event genuinely did extinguish (mean |premium| 0.6% afterwards). Read
  together, the pair is instructive: the launch closed the **trust discount** completely
  and left the **futures basis** untouched, because only the first was an access wedge.
- **[Study 100 — Melting Ice](../../100-melting-ice/)** and
  **[Study 375 — VXX Roll Decay](../../375-vxx-roll-decay/)**: the same contango-decay
  mechanic in commodity and VIX ETPs — mechanism siblings, different instruments, and
  neither has a dated structural event to test.

## Data sources

- **BITO** (ProShares Bitcoin Strategy ETF, CME front-month futures, fully collateralised,
  0.95% fee), **IBIT** (iShares Bitcoin Trust, 0.25%), **FBTC** (Fidelity Wise Origin
  Bitcoin Fund, 0.25%), **BTC-USD** (the coin) and **BIL** (1-3M T-bill, the cash and
  collateral proxy) — daily **total-return** closes via `yfinance` (`auto_adjust=True`).
  Total return is not optional here: BITO's monthly distributions have run at times above
  50% annualised, and a price-only tape would report most of that returned capital as
  "drag".
- **As-of 2026-06-30**; the partial current month is dropped so the sample never creeps.
  Common windows are set by inception: BITO from 2021-10-20, IBIT and FBTC from
  2024-01-11. Expense ratios and the borrow rate are prospectus/desk **assumptions**, not
  tape measurements, and are swept wherever they matter.
