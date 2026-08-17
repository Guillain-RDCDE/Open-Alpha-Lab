# References & literature map — Study 961 (Which Gold)

## The claim under test

- **The wrapper-choice folklore.** Five US physically-backed gold trusts hold the same
  allocated London bullion, publish a bar list, and strike at the same 16:00 New York close.
  They differ in essentially one contractual respect — the sponsor fee, from GLDM's 10 bp to
  GLD's 40 bp. The claim: that difference is not a marketing number but a *realised* one, so
  the ranking of realised tracking differences should be the exact reverse of the ranking of
  published fees, and holding the wrong wrapper for a decade should cost you the compounded
  gap.
- **The steelman against it.** A grantor trust's tracking difference is not only its fee. It
  also carries the premium/discount at which its shares trade to NAV, the timing of its
  in-kind creations, and (for the thinner names) the execution cost of getting in and out.
  If those swamp a 30 bp fee spread, the fee sheet is a poor guide to what you actually
  keep — and the cheapest wrapper is not the right answer for a large ticket.

## Why the fee should show up — the mechanism

- **Sharpe (1966), *Mutual Fund Performance*, Journal of Business**, and the long line after
  it: costs are the one component of a fund's return that is known in advance and
  subtracted with certainty. A physically-backed bullion trust is the purest case — it holds
  one asset, does not trade, does not lend, and sells metal to pay the sponsor.
- **Carhart (1997), *On Persistence in Mutual Fund Performance*, Journal of Finance** — the
  expense ratio is among the very few fund characteristics that predict future net returns,
  and it predicts them negatively, roughly one for one. Our cross-sectional pass-through
  slope of −0.89 to −0.94 is that result on a five-point, single-asset cross-section where
  nothing else can be doing the work.
- **French (2008), *Presidential Address: The Cost of Active Investing*, Journal of
  Finance** — the aggregate cost of the wrapper is the reliable part of the investment
  outcome. On bullion it is essentially the *whole* of the wrapper decision.
- **Elton, Gruber & Busse (2004), *Are Investors Rational? Choices Among Index Funds*,
  Journal of Finance** — investors demonstrably fail to buy the cheapest of a set of
  near-identical index funds. GLD, at four times GLDM's fee and ten times its
  assets-per-basis-point of visibility, is the bullion instance of exactly that puzzle.

## Why it can fail to show up on a price tape

- **Premium/discount noise.** Petajisto (2017), *Inefficiencies in the Pricing of
  Exchange-Traded Funds*, Financial Analysts Journal — ETF closing prices deviate from NAV,
  and the deviations are transient. On this cohort the wrapper-minus-wrapper residual has a
  first-order autocorrelation of −0.46 daily (and −0.42 monthly): a mean-reverting premium,
  which is why the daily estimator is nearly powerless (naive *t* = +0.78 on a 26 bp/yr
  effect) and the non-overlapping monthly one is not — and why the HAC correction points the
  wrong way here.
- **The detection floor.** A 30 bp/yr fee gap is 0.12 bp/day against ~6 bp/day of pairwise
  noise. Ben-David, Franzoni & Moussawi (2017), *Exchange-Traded Funds*, Annual Review of
  Financial Economics, on the microstructure that generates that noise. The consequence is
  arithmetic and reported explicitly: the coarse fee ranking resolves and the single-digit
  gaps do not.
- **Liquidity is not fee.** Amihud (2002), *Illiquidity and Stock Returns*, Journal of
  Financial Markets — the cost of moving size is a separate, larger number than the holding
  cost for anyone trading in size. The counterweight in this study is the ADV table, where
  the fee-cheapest tier is 200× thinner than the fee-priciest wrapper.

## Related desk studies (dedup)

- **[Study 920 — Total Cost of Ownership](../../920-total-cost-of-ownership/)**: the same
  cheap-versus-dear tracking-difference measurement on **equity index** wrappers
  (IVV/VOO/SPY, QQQM/QQQ), where the gaps are 6–7 bp, the expensive leg is a unit investment
  trust whose cash drag cannot be separated from its fee, and the whole result rests on five
  shared years. Study 961 is the commodity-trust case: **five** wrappers rather than pairs, a
  **30 bp** spread rather than 6, eight years, a grantor-trust structure with **no cash drag
  and no securities lending** to confound the fee, and a *ranking* test (Spearman across the
  whole cohort) rather than a pairwise break-even.
- **[Study 913 — Tracking-Difference Persistence](../../913-tracking-difference-persistence/)**:
  asks whether *last year's* tracking winner stays the winner — a persistence question, and
  its answer there is no. Study 961 asks the level question (does the fee ranking predict the
  outcome ranking at all) and reports the persistence rule only as a counterweight: chasing
  the winner loses to owning the cheapest.
- **[Study 959 — Crypto Fee War](../../959-crypto-etf-fee-war/)**: the same measurement stack
  on the ten spot-bitcoin ETFs, where a 24/7 asset against a 16:00 strike puts a 135 bp/day
  clock stub into every fund-versus-spot comparison and the fee spread is 131 bp on 29
  months. Gold has no clock problem, a spread four times narrower, and eight years of tape —
  a slower, cleaner instance of the same question.
- **[Study 915 — K-1 vs 1099](../../915-k1-vs-1099-structure/)** and
  **[Study 908 — Optimized-Roll Commodities](../../908-optimized-roll-commodities/)**: other
  commodity-wrapper choices, but both turn on the *futures* structure (tax form, roll
  method). The five trusts here hold physical metal and roll nothing.
- **[Study 912 — Gold + Trend](../../912-gold-trend-managed/)**, **[640](../../640-gold-overnight/)**,
  **[649](../../649-gold-seasonality/)**, **[831](../../831-gold-real-yield-timing/)**: gold
  *timing* studies. Study 961 takes no view on the metal at all — every number in it is a
  difference between two wrappers, in which the gold price cancels exactly.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica —
  [`strategy.newey_west_t`](../which_gold/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py). Note the unusual
  direction here: the residual is negatively autocorrelated **daily (−0.46) and monthly
  (−0.42)**, so the HAC variance is *smaller* than the iid one at both frequencies and the
  HAC *t* keeps growing with the bandwidth (+3.18 naive → +9.43 at lag 12 on the headline
  pair). Every estimator therefore prints both *t*s and the study quotes the naive one —
  the one case on this desk where Newey-West is the optimistic choice, not the cautious one.
- **Rank correlation and its permutation null.** Spearman (1904), *The Proof and Measurement
  of Association between Two Things*, American Journal of Psychology; Fisher (1935), *The
  Design of Experiments*, for the randomisation test —
  [`strategy.rank_test`](../which_gold/strategy.py), enumerated exactly over all 120
  permutations of five funds.
- **Moving-block bootstrap.** Künsch (1989), *The Jackknife and the Bootstrap for General
  Stationary Observations*, Annals of Statistics; Politis & Romano (1994), *The Stationary
  Bootstrap*, JASA — [`strategy.block_bootstrap_ci`](../which_gold/strategy.py) and
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).

## Data sources

- **GLD, IAU, GLDM, SGOL, BAR** — daily adjusted closes via `yfinance`
  (`auto_adjust=True`), 2004 → 2026-06-30, common window from GLDM's 2018-06-26 inception.
  None of the five distributes, so adjusted close = price close and the total-return /
  price-only labels coincide. Daily dollar volume (close × volume) from the same source.
- **BIL** (1–3 month T-bill ETF) — genuine total return, used only as the cash leg of the
  excess-of-cash ownership race.
- **Sponsor fees** — the funds' published fee schedules at build time, plus GLDM's
  2020-10-01 cut from 18 bp. Both are **ASSUMPTIONS**: they are announcement facts, not tape
  facts, they carry hindsight, and every result that uses them is also quoted on the
  fee-stable sub-window and on rank order alone.
- **As-of 2026-06-30.** The partial current month is dropped so the monthly estimator never
  eats a stub and the sample never creeps between reruns.
