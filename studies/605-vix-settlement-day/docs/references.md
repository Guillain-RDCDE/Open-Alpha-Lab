# References — Study 605 (VIX Settlement Day)

## The claim's source

- **Griffin, J. M. & Shams, A. (2018).** *Manipulation in the VIX?* — The Review of Financial
  Studies, 31(4), 1377-1417. <https://doi.org/10.1093/rfs/hhx085>
  The origin of the claim: at the monthly VIX-derivative settlement, volume spikes in the
  deep-OTM SPX options that feed the settlement formula, and the settlement print (SOQ/VRO)
  systematically deviates from the VIX levels immediately around it — consistent with the
  auction being pushed, not with hedging.

## Context: the settlement mechanism & its aftermath

- **CBOE, VX futures contract specifications** — final-settlement rule (the Wednesday 30 days
  before the following month's S&P-500 option expiration, holiday-adjusted) and the SOQ
  opening-auction procedure. <https://www.cboe.com/tradable_products/vix/vix_futures/specifications/>
- **CBOE futures settlement calendars** (used to verify the rule-built calendar, incl. the
  holiday-shifted Tuesdays). <https://cdn.cboe.com/resources/aboutcboe/Cboe-2026FuturesSettlementCalendar.pdf>
- **CBOE product update (2024).** *Juneteenth Holiday Closure Impact on VIX Options and VX
  Futures* — the 2024-06-18 Tuesday settlement.
  <https://cdn.cboe.com/resources/product_update/2024/Reminder-Juneteenth-Holiday-Closure-Impact-on-VIX-Options-and-VX-Futures.pdf>
- **Macroption, VIX expiration calendar** — independent cross-check of the 2025-2026 dates.
  <https://www.macroption.com/vix-expiration-calendar/>
- **In re CBOE Volatility Index Manipulation Antitrust Litigation**, N.D. Ill. (2018-) — the
  consolidated class actions filed in the wake of the paper; part of the "did it fade after
  2018?" third axis.
- **Pearson, N., Yang, Z. & Zhang, Q. (2020s working papers)** on VIX settlement deviations
  and hedging-vs-manipulation interpretations — the counterpoint that unwinding hedges can
  produce similar prints.

## Method & confounder citations

- **Welch, B. L. (1947).** The generalization of "Student's" problem — the group-split *t*.
- **White, H. (1980).** A heteroskedasticity-consistent covariance matrix estimator — the
  robust *t* on the settlement × gap interaction.
- **Lucca, D. & Moench, E. (2015).** *The Pre-FOMC Announcement Drift*, JF — why FOMC
  statement Wednesdays (40 of our 270 settlements) must be stripped before attributing
  Wednesday volatility to the settlement.
- **Federal Reserve, FOMC historical calendars** — the statement-day table (shared with
  sibling studies [67-fed-drift](../../67-fed-drift/), [135-fomc-cycle](../../135-fomc-cycle/)).
  <https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm>

## Data sources

- **yfinance** `^VIX` daily OHLC (CBOE Volatility Index) and `^GSPC` daily OHLC —
  <https://finance.yahoo.com/quote/%5EVIX/> (no key; known quirk: ~4% of ^VIX opens exactly
  equal the prior close — stale-open prints, screened in robustness).
- **Settlement calendar** — built by rule in [`vix_settlement_day/data.py`](../vix_settlement_day/data.py)
  (Gregorian Easter for Good Friday + Juneteenth from 2022), asserted against 18 known CBOE dates.

## Named siblings (dedup guard)

This study is the **settlement-day microstructure event** — a calendar-dated auction
footprint in the *index*. It is deliberately distinct from the desk's VIX *pricing-structure*
studies: [111-vix-term-structure](../../111-vix-term-structure/) (the futures curve's shape
and carry) and [375-vxx-roll-decay](../../375-vxx-roll-decay/) (the ETP roll-drag arithmetic).
Neither touches the monthly settlement print; this study touches nothing else.
