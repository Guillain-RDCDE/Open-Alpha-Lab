# References & literature map — Study 640 (Gold-Overnight, "Gold Trades in Its Sleep")

## The claim under test

- **The folklore chart.** Adrian Douglas, *"Gold market is not 'fixed', it's rigged"*
  (GATA, 2010, <https://www.gata.org/node/8560>): buying gold at the London PM fix and
  selling at the next AM fix from 2001 compounded spectacularly, while the reverse
  (AM → PM, the London trading day) lost money — the origin of the "gold only goes up while
  you sleep" meme, read there as proof of fix-hour suppression.
- **The academic version of the fix mechanism.** Andrew Caminschi & Richard Heaney,
  *Fixing a Leaky Fixture: Intraday Effects of the London PM Gold Price Fixing* (2014,
  **Journal of Futures Markets** 34(11), 1003–1039, <https://doi.org/10.1002/fut.21636>):
  significant price moves and information leakage in the minutes around the PM fix calls —
  the paper that put the London fix under the microscope before the 2014 Barclays FCA fine
  (£26m, May 2014) and the benchmark reforms.
- **The fix reform (our externally-dated break).** The twice-daily phone-based London Gold
  Fix (est. 1919, five bullion banks) was replaced by the electronic, auditable **LBMA Gold
  Price** auction administered by ICE Benchmark Administration on **2015-03-20**
  (<https://www.lbma.org.uk/prices-and-data/precious-metal-prices>). If the overnight/day
  asymmetry were fix-rigging leakage, it should shrink after this date. (It didn't — it grew.)
- **Gold-fix microstructure literature.** Brian Lucey, Fergal O'Connor et al. on gold fixes
  and gold market efficiency (e.g. O'Connor, Lucey, Batten & Baur, *The Financial Economics
  of Gold — a survey*, **IRFA** 2015, <https://doi.org/10.1016/j.irfa.2015.07.005>); Aggarwal,
  Lucey & O'Connor on the fixes' price-discovery role. These frame the fixes as *the* candidate
  mechanism the claim leans on.
- **The manipulation reading of overnight drift in general.** Bruce Knuteson's series
  ("Wake Up", "They Still Haven't Told You", arXiv:1912.01708 etc.) includes GLD among the
  instruments whose gains accrue overnight — the desk's full audit of that inference is the
  sibling study below.

## The named sibling — how this study is distinct

- **[01-overnight-anomaly](../01-overnight-anomaly/)** (Real × Mirage) is the **US
  equities** version: SPY's overnight mean at NW *t* ≈ 5, breadth across 497 stocks, the
  calendar-time illusion, capacity, and the Knuteson manipulation posterior. **This study is
  the gold version**: a different asset class whose *mechanism claim* is specific — the
  London fixes — with its own natural experiment (the 2015-03-20 LBMA reform) that equities
  don't have, plus a second-sponsor confirmation (IAU) and SPY recycled as the *placebo*
  rather than the subject. The headline question is not "do prices rise overnight?" (01
  answered that) but "is gold's night/day split bigger than the market-wide clock effect,
  did the fix reform dent it, and was it ever harvestable at daily frequency?".

## Method notes

- **Session split.** overnight = prior close → open; intraday = open → close; the legs
  compound exactly to close-to-close. Adjusted (total-return) opens and closes, both legs
  from the same adjusted series — the house rule that adjustment mode moves return between
  night and day, stated in [`data.py`](../gold_overnight/data.py).
- **HAC/Newey-West t** (Newey & West 1987, *Econometrica*) on daily means and on the paired
  daily difference — daily session returns are mildly autocorrelated and volatility-clustered.
  **Welch t** (Welch 1947) for the group splits and for the change-in-gap across the reform.
- **Sign-flip permutation placebo** (Fisher randomization logic; Efron & Tibshirani 1993):
  under the null that night/day labels are exchangeable within a session, each day's
  difference is symmetric around zero — 20,000 seeded sign-flips give the exact-style *p*.
- **Sub-period contrast discipline.** The split date is externally dated (the LBMA reform),
  not snooped, and the decay claim carries its own Welch *t* on the *difference between
  sub-period gaps* — per METHODOLOGY's "no conditional claim without uncertainty".
- **Harvest test.** Buy MOC / sell next MOO is the unconditional clock rule (the MOC/MOO
  convention is the documented execution lag — nothing is conditioned on same-bar
  information); costs are 2 one-way trades/day × cost × NAV; long-only so no borrow.
  Frazzini, Israel & Moskowitz (2018, *Trading Costs*) motivates the gross-vs-net discipline.

## Data sources used here

- **yfinance** daily adjusted open + close for **GLD** (SPDR Gold Shares, listed
  2004-11-18), **IAU** (iShares Gold Trust, from 2005-01-31) and **SPY**, 2004-11-18 →
  2026-06-30, cached under [`_cache/go_ohlc.csv`](../_cache/go_ohlc.csv). All headline
  numbers pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py) (fingerprint `648ab68f37f2`).

## Related desk studies

- [01-overnight-anomaly](../01-overnight-anomaly/) — the equities original (Real ×
  Mirage), cited throughout as the placebo benchmark.
- [113-gold-silver-ratio](../113-gold-silver-ratio/), [305-gold-oil-ratio](../305-gold-oil-ratio/),
  [580-gold-lease-rate](../580-gold-lease-rate/) — the desk's other gold-market claims
  (cross-asset ratios and the lease-rate carry), none of which touch the intraday clock.
- [116-power-hour](../116-power-hour/) — the desk's other time-of-day slice (equity
  last-hour folklore), a cousin of the session-split machinery used here.
