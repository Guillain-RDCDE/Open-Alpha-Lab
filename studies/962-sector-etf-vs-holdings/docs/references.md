# References & literature map — Study 962 (Do It Yourself)

## The claim under test

- **The do-it-yourself replication argument.** A cap-weighted sector fund is extremely
  top-heavy: the Technology Select Sector SPDR's three largest positions were roughly 42%
  of the fund at the study's as-of date, and its top ten around 66%. The retail conclusion
  follows easily — buy those names directly in a commission-free account, skip the
  0.08%/yr expense ratio, and you have the fund for free. It is stated most often in the
  Bogleheads and r/investing archives, and periodically in the trade press whenever
  "direct indexing" is being marketed.
- **The steelman.** With zero-commission trading and fractional shares the mechanical
  objections (odd lots, minimum tickets) really have gone away, and the fee really is a
  certain, compounding drag. If the top ten names *were* a good proxy for the fund, the
  argument would be arithmetically correct.
- **What the desk actually tests.** Whether the replication error is small enough for the
  fee saving to be the operative term. That is a question about *tracking error*, not
  about fees — and it is answered on the tape, twice: once with the fund's holdings as
  they are published *today* (the hindsight basket a reader would accidentally build) and
  once with the fund's holdings as published at the *start* of the sample.

## Direct indexing, replication and tracking error

- **Sampling vs full replication.** Frino & Gallagher (2001), *Tracking S&P 500 Index
  Funds*, Journal of Portfolio Management; Blume & Edelen (2004), *S&P 500 Indexers,
  Tracking Error, and Liquidity*, Journal of Portfolio Management. Both quantify what a
  partially-replicating fund gives up: tracking error rises sharply as the sampled basket
  shrinks, and it is two-sided. Our depth sweep is the same experiment run backwards, from
  the investor's side of the wrapper.
- **Optimal index replication.** Beasley, Meade & Chang (2003), *An Evolutionary Heuristic
  for the Index Tracking Problem*, European Journal of Operational Research — a formal
  treatment of choosing K names to track an N-name index. The relevant result for a
  retail reader is the shape of the frontier: tracking error falls slowly in K and stays
  in whole percentage points for the K a private investor would actually hold.
- **Direct indexing in practice.** Chaudhuri, Burnham & Lo (2020), *An Empirical
  Evaluation of Tax-Loss-Harvesting Alpha*, Financial Analysts Journal — the honest case
  for holding an index's constituents directly rests on **tax-loss harvesting**, not on
  the expense ratio. This study deliberately does not model tax (it would be an unlabelled
  assumption about a reader's bracket); it tests only the fee-saving argument, which is
  the one that circulates.

## Why the naive test breaks — look-ahead in the constituent list

- **Survivorship and back-fill bias.** Brown, Goetzmann, Ibbotson & Ross (1992),
  *Survivorship Bias in Performance Studies*, Review of Financial Studies; Elton, Gruber &
  Blake (1996), *Survivorship Bias and Mutual Fund Performance*, RFS. Building a basket
  from *today's* membership list and running it backwards is the textbook version of the
  bias, and this study measures its size directly — **+5.81%/yr** across three sectors,
  after separating it from the cap-versus-equal weighting change that a naive
  hindsight-minus-contemporaneous difference (+6.84%/yr) silently bundles in.
- **Index-membership drift.** Chen, Noronha & Singal (2004), *The Price Response to S&P 500
  Index Additions and Deletions*, Journal of Finance — membership turns over enough that a
  fixed list is a different portfolio from the fund within a few years. Our era cut shows
  exactly that: the tracking error of a fixed 2011 basket roughly doubles in 2019–2026.
- **Concentration is a risk, not a free lunch.** Bessembinder (2018), *Do Stocks
  Outperform Treasury Bills?*, Journal of Financial Economics — the extreme skew of
  single-name returns means a ten-name basket is a wager on a handful of outcomes, not an
  approximation to a diversified sleeve. Our worst rolling-12-month shortfalls (−12.3%
  blended, −25.7% in energy) are that skew, priced.

## Related desk studies (dedup)

- **[Study 920 — Total Cost of Ownership](../../920-total-cost-of-ownership/)**: the same
  *fee-versus-friction* arithmetic, but between two **wrappers** (SPY vs SPLG, QQQ vs
  QQQM). Study 962 keeps the fee question and removes the wrapper on one side entirely —
  the competitor is a basket of individual shares, so the term that dominates is
  replication error rather than spread.
- **[Study 177 — Megacap-Concentration](../../177-megacap-concentration/)**: does buying
  the biggest stocks beat *the market*? That is a return question about a cross-sectional
  size tilt. Study 962 asks a *tracking* question about a specific fund: does a top-N
  basket **reproduce** the sector it was carved from, cheaply enough for the fee to matter.
- **[Study 28 — Carousel](../../28-carousel/)** and
  **[Study 225 — Sector-Rotation](../../225-sector-rotation/)**: both *choose between*
  sectors over time. Study 962 never rotates — the sector is given, and the only decision
  is whether to own it through the fund or through its own top holdings.
- **[Study 890 — Sector Risk-Parity](../../890-sector-risk-parity/)** and
  **[Study 903 — Sector-Neutral Low-Vol](../../903-sector-neutral-lowvol/)**: re-weighting
  *across* the eleven sectors. Study 962 works strictly *inside* three of them.
- **[Study 870 — Industry-Leader Lead-Lag](../../870-industry-leader-lead-lag/)**: uses the
  biggest name in a sector as a *predictive signal* for the rest. Here the biggest names
  are not a signal — they are the portfolio.
- **[Study 913 — Tracking-Difference Persistence](../../913-tracking-difference-persistence/)**:
  measures tracking difference *between funds tracking the same index*. Study 962 measures
  it between a fund and a home-made substitute, where it is two orders of magnitude larger.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica —
  [`strategy.newey_west_t`](../diy_sector/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Circular block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap*, JASA
  — [`strategy.bootstrap_gap_ci`](../diy_sector/strategy.py), 21-day blocks so the
  interval respects the volatility clustering in a tracking-difference series.
- **Sharpe-difference inference.** Jobson & Korkie (1981), *Performance Hypothesis Testing
  with the Sharpe and Treynor Measures*, Journal of Finance — the excess-of-cash return
  difference is tested directly, in its Newey-West form.
- **Reproducibility stamp.** [`quantlab/repro.py`](../../../quantlab/repro.py) — as-of
  slice plus a content fingerprint of the exact input frame.

## Data sources

- **XLK, XLE, XLF** (SPDR Select Sector funds), **44 constituent names** drawn from their
  published top-10 lists (today's and January 2011's), and **BIL** (1-3 month T-bill, the
  cash leg) — daily **total-return** closes via `yfinance` (`auto_adjust=True`),
  2011-01-03 → 2026-06-30. Total return is mandatory here: the three sectors' dividend
  yields differ by several percentage points, so a price-only comparison would invent a
  gap out of nothing.
- **The holdings lists are hardcoded PROXIES.** Yahoo! Finance serves no point-in-time
  holdings history, so both lists are typed in from public fund fact sheets and treated as
  a fixed, labelled input. The January-2011 energy list is **survivor-only** — Anadarko
  (`APC`, acquired 2019) and Marathon Oil (`MRO`, acquired 2024) were genuine
  constituents but have no retrievable tape; the surviving symbol `APC` now resolves to an
  unrelated new listing and is not used, so the two slots go to the next *surviving* names
  down the sheet. More generally every name in every 2011 list is by construction one that
  still has a tape, so the control is survivor-conditional in all three sectors. That bias
  runs in the do-it-yourself basket's favour and is named on the Signal axis rather than
  papered over.
- **Ticker-continuity PROXIES.** `GOOGL` and `BRK-B` stand in for the share lines a 2011
  investor actually held — Google traded as a single `GOOG` line until the April-2014
  class split, and Yahoo!'s back-adjusted series is the closest retrievable total return.
- **No tax, and frictionless dividend reinvestment on both legs.** `auto_adjust=True`
  reinvests every distribution at the close on both sides, and no capital-gains tax is
  modelled anywhere. This favours the **do-it-yourself** basket: an ETF disposes of
  appreciated stock through in-kind redemption, whereas a private investor rebalancing ten
  names realises the gain. The single one-way `cost_bps` charge covers commission and
  spread together, and is swept from 0 to 50 bps.
- **Expense ratios (0.08%/yr).** Quoted for context only and never added anywhere: each
  fund's own total-return close already carries its fee, which is precisely what makes the
  race fair.
- **As-of 2026-06-30.** The partial current month is dropped so the sample never creeps.
