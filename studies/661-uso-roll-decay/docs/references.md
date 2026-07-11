# References & literature map — Study 661 (USO-Roll-Decay)

## The claim under test

- **The folklore.** "USO tracks oil" — retail shorthand for *"buy USO, own the oil price."*
  USO (United States Oil Fund LP) does not hold physical crude; since inception it has held
  (predominantly front-month, later a laddered basket across contract months) NYMEX WTI
  futures. When the term structure sits in **contango** — the normal state of a market with
  positive storage/financing cost — each monthly roll sells the cheaper *expiring* contract and
  buys the pricier *next-month* one, a structural "sell low, buy high" that a spot holder never
  pays. The complaint is decades old among options/futures desks and shows up every time crude
  rallies "but my USO position barely moved."
- **The academic anchor.** Erb & Harvey (2006, *The Strategic and Tactical Value of Commodity
  Futures*, FAJ) is the foundational paper on roll yield as a first-order driver of commodity
  index/ETF returns — a positive-carry ("backwardated") curve pays the long roll a premium, a
  negative-carry ("contangoed") curve charges it a toll. Alquist & Gervais (2013, *The Role of
  Financial Speculation in Driving the Price of Crude Oil*, Energy Journal) and Mou (2010,
  *Limits to Arbitrage and Commodity Index Investment: Front-Running the Goldman Roll*) document
  the mechanics and market impact of large futures-ETF rolls specifically. Simon (2013, *Trading
  the Oil-Volatility Correlation*, JOI) is one of several practitioner notes quantifying USO's
  specific tracking error against spot/near-month WTI.
- **The dramatic confirmation.** On 2020-04-20 the May-2020 WTI futures contract settled at
  **-$37.63/barrel** — the first negative settlement in the contract's history, driven by a
  storage-capacity crunch at the Cushing, OK delivery point (EIA *This Week in Petroleum*,
  April-May 2020 issues). USO's structure and size were widely reported as *contributing* to
  the pressure on that specific contract in the days before expiry (SEC/CFTC commentary and
  contemporaneous financial press, April 2020), and the fund filed prospectus supplements that
  same month spreading its holdings further across contract months and executed a 1-for-8
  reverse stock split (2020-04-29) to keep its per-share price economically viable.

## What we measure, and the honesty rails

- **CL=F, not physical spot.** No free, continuous cash-WTI series exists; CL=F (Yahoo's
  continuously-rolled NYMEX front-month print) *is* the number every ticker and news headline
  calls "the price of oil." Benchmarking USO against it is not a compromise — it is the exact
  comparison retail investors make, and it is what the folklore's "tracks oil" actually claims.
- **A single documented data quirk.** The negative WTI settlement (2020-04-20) makes `np.log`
  of that price (and of the following day, whose *previous* close is negative) undefined. Both
  days drop out of the continuous gap series automatically via `NaN`/`dropna` — no manual row
  deletion, which would otherwise silently splice non-adjacent trading days together. That
  window is instead a **named, separate case study** using simple (not log) returns, which stay
  well-defined across a sign flip.
- **One-sample inference, not Welch.** Every day carries a gap (there is no "control group" the
  way FOMC-day vs non-FOMC-day works) — the paired daily differential is tested with a naive
  one-sample *t* and a **Newey-West (1987)** HAC *t* at three lags (5/21/63 sessions), plus a
  **circular block-bootstrap** (block ≈ one roll cycle) for a distribution-free CI/placebo. The
  regime split (contango-stress windows vs the rest) *is* a genuine two-group comparison and
  uses Welch (1947), matching house convention.
- **Regime windows are hardcoded, not fitted.** The 2008-09 storage glut and the 2020 COVID
  collapse are named, dated, and cited *before* the split is run — this is a documented
  historical regime label, the same convention as the FOMC-calendar and press-conference-era
  splits used elsewhere on the desk, not a threshold tuned on the outcome.

## Why the tradable echo is graded separately

- The "long spot / short USO" pairs book is a constant-notional, **monthly**-rebalanced book —
  costs are charged one-way × NAV per leg on rebalance days only (not daily, which would
  overstate a marginal-turnover book's real transaction costs), and the short leg pays a named,
  conservative 0.75%/yr borrow fee (USO is large and liquid; this is not a crisis-borrow
  assumption).
- **The decisive number is the ex-crisis split**, not the full-sample Sharpe: stripping the two
  hardcoded stress windows collapses the net return from +7.17%/yr to **+0.27%/yr** — a
  textbook case of a backtest whose entire edge rides on two rare, unforecastable historical
  events (a storage crisis on each side of the sample), the same critique the desk applies to
  any strategy whose Sharpe evaporates once its lucky windows are removed.
- No survivorship correction is needed on either leg — USO and CL=F are each the single live
  instrument/continuous print across the sample, not a basket conditioned on survivors.

## Data sources

- **USO adjusted close** and **CL=F front-month close** — yfinance (no key), cached under
  `_cache/` (`urd_uso.csv`, `urd_clf.csv`), 2006-04-10 → 2026-06-30.
- **Contango-stress windows**, hardcoded in [`data.py`](../uso_roll_decay/data.py): EIA *This
  Week in Petroleum* storage commentary (Cushing utilization, Dec 2008–Jun 2009 and Mar–Jun
  2020) and CME/NYMEX settlement history for 2020-04-20.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [100-melting-ice](../100-melting-ice/) — the **general commodity-contango decay** story
  across a broad futures-index universe. This study: **one specific, huge retail ETF (USO)
  against the exact number its holders think it tracks**, with the regime concentration and the
  April-2020 case study as its own axis.
- [226-crude-seasonality](../226-crude-seasonality/) — a **calendar/seasonal** pattern in crude
  returns. Not about the futures-vs-spot structural wedge at all.
- [375-vxx-roll-decay](../375-vxx-roll-decay/) — the same contango-roll mechanism, but in
  **VIX futures** (fear, not a physical commodity) via VIXY — a short-carry-vs-crash-tail study
  with the opposite trade direction (short the decaying vehicle) from this study's long-spot
  framing.
- [619-bito-roll-drag](../619-bito-roll-drag/) — the same family of "futures-ETF-vs-spot" toll,
  in **bitcoin** (BITO vs spot BTC/IBIT), on a **monthly** CME roll rather than USO's rolling
  front-month/laddered-basket structure, with its own matched-close spot-ETF ruler.
- Any leveraged-fund decay study (e.g. variance drag from daily rebalancing) is a **different
  mechanism** — this study's drag comes entirely from the futures curve's shape, not from
  compounding a fixed daily leverage ratio.

None of the siblings test **USO specifically against the CL=F front-month print retail
investors actually see quoted** — that comparison, and the finding that most of the damage
lives in two hardcoded crisis windows rather than a smooth daily bleed, is this study's own
axis.
