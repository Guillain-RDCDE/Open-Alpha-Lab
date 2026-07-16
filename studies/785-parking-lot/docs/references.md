# References & literature map — Study 785 (Parking-Lot)

## The claim under test

- **The folklore.** "Count the cars in Walmart's parking lots from orbit and you can nowcast the
  quarter before the earnings print — long the busy quarters, short the empty ones." A signature
  pitch of the **alternative-data / satellite-imagery** industry since the mid-2010s
  (Orbital Insight, RS Metrics, and the broader geospatial-alpha wave), with big-box retailers
  like **Walmart (WMT)** and Target as the poster children (huge lots, easy to image).
- **THE SIGNAL HERE IS A LABELLED PROXY, NOT REAL SATELLITE DATA.** Genuine orbital car-count
  panels are paywalled, license-restricted, and not backfilled/redistributable, so this study
  does **not** fabricate one and pass it off as a feed. The parking signal in
  [`data.py`](../parking_lot/data.py) is a hardcoded, deterministic, **stylised** quarterly
  foot-traffic index used **ordinally only** (busier vs emptier lots year-over-year). Read the
  loud banner there: a positive result would still need the real, paid panel to bank.
- **Why it's a clean forward-return test.** The satellite "verdict" for a quarter is available
  *before* the earnings print, and WMT's reporting cadence (mid/late Feb, mid-May, mid-Aug,
  mid-Nov) is known years ahead — so a "buy busy / sell slow at the print, hold K sessions"
  rule is calendar-known and zero-look-ahead by construction.
- **The efficient-markets prior.** If a satellite panel really nowcast the print, the *sell-side
  and the panel's own paying subscribers* would trade it into the price by the release — so any
  residual **post-print drift** a proxy could capture should be arbitraged away. The desk's prior
  is a clean null. See Fama (1970, *Efficient Capital Markets*, JF).

## What the literature actually says

- **Satellite imagery as an equity signal** — Katona, Painter, Patatoukas & Zeng (2018/2023,
  "On the Capital Market Consequences of Alternative Data", working paper / *Management Science*):
  parking-lot satellite counts *did* predict retailer earnings surprises AND abnormal returns —
  but the tradable edge accrued **early, to the data's subscribers**, and *diminished* as the
  data diffused; the price impact concentrates in the **pre-announcement** window, not a
  post-print drift a latecomer can harvest. This is the most directly relevant paper and it cuts
  *against* the naive "beat the print with a proxy" story for a public, delayed signal.
- **Post-earnings-announcement drift (PEAD)** — Ball & Brown (1968, *JAR*); Bernard & Thomas
  (1989, *JAR*; 1990, *JAE*). The canonical "prices drift *after* a scheduled information event."
  Our forward windows (K = 5, 21 sessions post-print) are exactly where a residual drift would
  live if the parking signal added anything beyond the print.
- **Alternative data & alpha decay** — Kolanovic & Krishnamachari (2017, J.P. Morgan, *Big Data
  and AI Strategies*); Denev & Amen (2020, *The Book of Alternative Data*). Both stress that
  alt-data edges are real early and **decay fast** once commoditised — a capacity/timeliness
  story, not a free lunch for a stylised public proxy.
- **Attention & retail catalysts** — Da, Engelberg & Gao (2011, *JF*) on search-based attention;
  Barber & Odean (2008, *RFS*). Motivate *why* a high-profile nowcast narrative attracts crowding
  around retailer prints, but attention is not, by itself, a tradable edge.

## Data & method

- **Real tape:** `WMT` and `SPY` daily adjusted (total-return) closes via
  [yfinance](https://github.com/ranaroussi/yfinance), one combined panel. We measure the
  *abnormal* return `WMT − SPY`, not the raw move.
- **Signal:** a **LABELLED PROXY** quarterly foot-traffic index (2009→2025), used ordinally — the
  sign of the same-quarter year-over-year change (busy vs slow). NOT a live feed.
- **Statistics:** one-sample *t* of the direction-signed long/short forward return (busy minus
  slow); Welch two-sample *t* of the busy-vs-slow spread; Spearman(yoy, forward); Wilson hit-rate
  interval; a 40-seed × 250-draw sign-shuffle placebo; a leave-one-out jackknife; a costed net leg.
- **Synthetic positive control:** a seeded paired (asset, benchmark) world with a *planted*
  two-sided "busy→forward-up / slow→forward-down" link — the detector must recover a planted link
  monotonically and stay quiet on the null. See [`strategy.py`](../parking_lot/strategy.py).

*Fama, E. (1970). **JF**. · Katona, Z., Painter, M., Patatoukas, P. & Zeng, J. (2018/2023).
**Management Science** (wp). · Ball, R. & Brown, P. (1968). **JAR**. · Bernard, V. & Thomas, J.
(1989, 1990). **JAR / JAE**. · Da, Z., Engelberg, J. & Gao, P. (2011). **JF**. · Barber, B. &
Odean, T. (2008). **RFS**. · Kolanovic, M. & Krishnamachari, R. (2017). **J.P. Morgan**. · Denev,
A. & Amen, S. (2020). **The Book of Alternative Data**, Wiley.*
