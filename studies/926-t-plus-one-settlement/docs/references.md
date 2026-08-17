# References & literature map — Study 926 (T+1)

## The event under test

- **SEC Release No. 34-96930 (15 February 2023), *Shortening the Securities Transaction
  Settlement Cycle*.** The rule amending Exchange Act Rule 15c6-1(a) to move the standard
  settlement cycle for US cash equities, ETFs, corporate and municipal bonds and unit
  investment trusts from **T+2 to T+1**, with a compliance date of **Tuesday 28 May
  2024**. The Commission's own economic analysis predicted lower counterparty and margin
  exposure, offset by higher operational risk and a compressed affirmation window — it
  made no prediction about intraday price formation, which is precisely the null this
  study ends up confirming.
- **Canada and Mexico moved one business day earlier**, on 27 May 2024, and the UK and EU
  had not moved during this study's post-period (the UK Accelerated Settlement Taskforce
  and ESMA both targeted **October 2027**). That asymmetry is the study's treatment: a
  US-listed fund holding European or Asian securities settles its *shares* in one day and
  its *holdings* in two, so it must pre-fund a day of stock and a day of FX.

## Why the effect might be visible in prices

- **Settlement risk and price formation.** Duffie, Gârleanu & Pedersen (2005),
  *Over-the-Counter Markets*, Econometrica — search and settlement frictions are priced
  when they bind on the marginal holder. If a shortened cycle changes who can hold what
  overnight, the *overnight* leg is where it should show.
- **Fails to deliver and settlement frictions.** Evans, Geczy, Musto & Reed (2009),
  *Failure Is an Option: Impediments to Short Selling and Options Prices*, Review of
  Financial Studies; Fotak, Raman & Yadav (2014), *Fails-to-Deliver, Short Selling, and
  Market Quality*, Journal of Financial Economics. Both find settlement mechanics affect
  liquidity provision — the honest steelman for expecting a T+1 footprint. Our Study 558
  tests the fails channel directly and finds nothing tradable there either.
- **Overnight vs intraday decomposition.** Lou, Polk & Skouras (2019), *A Tug of War:
  Overnight Versus Intraday Expected Returns*, Journal of Financial Economics — the
  canonical statement that the night and day legs of a daily return are economically
  different objects. Knuteson (2019+, arXiv) documents the overnight leg's dominance of
  long-run index return. This study borrows their exact decomposition as its measuring
  instrument, not as its hypothesis.
- **Month-end and quarter-end flows.** Etula, Rinne, Suominen & Vaittinen (2020),
  *Dash for Cash: Monthly Market Impact of Institutional Liquidity Needs*, Review of
  Financial Studies — month-end settlement and funding needs move prices. If T+1
  compresses when those flows have to settle, the turn-of-month window is where a change
  should appear, which is why it is this study's one tradable arm.

## Why it plausibly leaves no price fingerprint

- **Plumbing changes are usually invisible in daily prices.** The costs a shortened cycle
  moves — FX swap funding for the extra day, securities-lending recall timing, ETF
  creation/redemption pre-funding, CCP margin — are borne in the *financing* layer, not in
  the tape. A price file has no column for any of them; this study is explicit that it
  tests only what a price file can see.
- **Event studies over a two-year post-window are underpowered against a macro cycle.**
  Roll (1988), *R²*, Journal of Finance, and the broader event-study literature
  (MacKinlay 1997, *Event Studies in Economics and Finance*, Journal of Economic
  Literature) both warn that long event windows drown a small treatment in common
  variation. Our placebo-date distribution quantifies exactly that: in 2022–2026, the
  *median* arbitrary date produces a bigger difference-in-difference *t* than the real one.
- **Difference-in-difference with a single treated group and one break date.** Bertrand,
  Duflo & Mullainathan (2004), *How Much Should We Trust Differences-in-Differences
  Estimates?*, Quarterly Journal of Economics — serial correlation makes naive DiD
  standard errors wildly over-confident, and placebo interventions are the recommended
  fix. That paper is the direct methodological ancestor of this study's placebo-switch
  table and its HAC-plus-block-bootstrap inference.

## Related desk studies (dedup)

- **[Study 01 — Overnight Anomaly](../../01-overnight-anomaly/)** and
  **[Study 788 — Overnight/Intraday Tug of War](../../788-overnight-intraday-tug-of-war/)**:
  both ask whether the *level* of the overnight leg is an exploitable return. Study 926
  uses the same decomposition but asks a different question — whether the *split itself*
  changed at a known institutional date. No return-predicting sort is formed here.
- **[Study 558 — Failures-To-Deliver](../../558-failures-to-deliver/)**: the other
  settlement-plumbing study on the desk, testing whether settlement *fails* forecast
  squeezes. Study 926 tests the settlement *cycle*, not fails, and is a difference-in-
  difference around a policy date rather than a signal sort.
- **[Study 605 — VIX Settlement Day](../../605-vix-settlement-day/)**: a recurring monthly
  settlement auction and its gap behaviour — a calendar effect in one instrument, not a
  one-off regime change across four.
- **[Study 89 — Turn-of-the-Month](../../89-turn-of-the-month/)**,
  **[Study 42 — Last-Call](../../42-last-call/)** and
  **[Study 604 — Month-End Rebalancing Flows](../../604-month-end-rebalancing-flows/)**:
  all measure the turn-of-month effect *as an effect*. Study 926 uses the turn-of-month
  window only as the venue for its DiD, and its tradable arm is a **long/short spread
  between a foreign and a domestic ETF**, which none of those three forms.
- **[Study 193 — Window-Dressing](../../193-window-dressing/)**: quarter-end flows in the
  cross-section of stocks; 926 is at the index-ETF level and conditioned on a settlement
  regime change.

## Method lineage

- **HAC / Newey-West standard errors.** Newey & West (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*,
  Econometrica — [`strategy.hac_ols`](../t_plus_one/strategy.py) and
  [`strategy.newey_west_t`](../t_plus_one/strategy.py), bandwidth
  `floor(4 (n/100)^(2/9))`.
- **Circular block bootstrap.** Politis & Romano (1992, 1994) — 21-day blocks preserve the
  volatility clustering that an i.i.d. bootstrap would destroy:
  [`strategy.did_bootstrap_ci`](../t_plus_one/strategy.py).
- **Placebo interventions in DiD.** Bertrand, Duflo & Mullainathan (2004), as above —
  [`strategy.placebo_switches`](../t_plus_one/strategy.py).
- **Night/day identity.** [`quantlab.decompose`](../../../quantlab/decompose.py) states the
  same identity used by [`strategy.decompose`](../t_plus_one/strategy.py).

## Data sources

- **SPY** (control), **IWM** (domestic placebo-treated), **EFA** and **EEM** (the
  settlement-mismatched legs), **BIL** (cash leg for the excess-of-cash Sharpe race) —
  daily **total-return** OHLC via `yfinance` (`auto_adjust=True`), each ticker's full
  history through 2026-06-30 (the common frame starts at BIL's 2007-05-30 inception),
  cached in the shared `studies/_cache`. `auto_adjust=True` scales the open and the close
  by the same daily factor, so the night/day identity is exact after adjustment; the
  *levels* are total-return, not price-only, and are labelled as such everywhere.
- **As-of 2026-06-30.** The partial current month is dropped so the sample never creeps.
  The event windows are symmetric by construction: 524 trading days either side of
  28 May 2024.
