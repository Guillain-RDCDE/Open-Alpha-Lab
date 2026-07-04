# References & literature map — Study 630 (Late Filers, Form NT)

## The claim under test

- **The folk rule.** "A company that cannot file its 10-K on time is telling you something —
  sell the moment the NT hits EDGAR." Under SEC Rule **12b-25**, a registrant that will miss a
  periodic-report deadline must file a **Form 12b-25** (EDGAR form types **NT 10-K** / **NT
  10-Q**, "NT" = *notification* of late filing), publicly admitting the delay and its reason.
  The filing buys 15 extra days (10-K) or 5 (10-Q) — and broadcasts that something inside the
  reporting process broke: an audit dispute, a restatement in progress, going-concern doubt,
  an internal-control failure, or plain distress.
- **The academic anchor.** Late filings are among the better-documented *negative* corporate
  events:
  - Alford, Jones & Zmijewski (1994, *JAE*) — extensions under Rule 12b-25 are associated with
    poorer performance and financial distress.
  - Griffin (2003, *Review of Accounting Studies*) — investors respond negatively to late 10-K
    filings around Form 12b-25 events.
  - Bartov, DeFond & Konchitchki (2015; earlier WP ~2010, "The Consequences of Untimely
    Quarterly and Annual Financial Reporting") — late filers, especially of 10-Ks citing
    accounting reasons, earn **negative abnormal returns that persist after** the NT filing —
    the market *underreacts* to the admission.
  - Duarte-Silva, Fu, Noe & Ramesh (2013, *JAE*, "How do investors interpret announcements of
    earnings delays?") — delay announcements carry significantly negative announcement and
    post-announcement returns.
- **The mechanism.** The NT is a certified bad-news event with a lawyer-reviewed excuse
  attached. The claim we test is not the announcement drop (that is instantaneous) but the
  **post-filing drift**: does the market *keep* marking the name down for weeks — an
  underreaction a short can actually capture?

## What we measure

- **Event = the filing, not the language.** One event per (CIK, filing date) of an exact-form
  **NT 10-K** or **NT 10-Q** (amendments excluded), from the EDGAR quarterly form-type master
  index, 2004-01 → 2026-03. This is deliberately distinct from the desk's
  [565-filing-readability](../565-filing-readability/) sibling, which scores the *text style*
  of filings that DID arrive; here the information is the **event itself** — the report that
  did *not* arrive on time.
- **CAR[+1, +60td] vs SPY.** Short at the close of the day after the filing (exactly ONE
  execution lag — NTs often land after hours), then sum daily (stock − SPY) returns over 60
  trading days. The announcement move (filing day → entry close) is reported separately and
  never counted as tradable.
- **Calendar-time portfolio is the primary test.** NT filings cluster around the 10-K and 10-Q
  deadlines, so event windows overlap heavily and the naive cross-sectional *t* over-counts
  (Fama 1998, *JFE*, "Market efficiency, long-term returns, and behavioral finance";
  Mitchell & Stafford 2000, *JB*). We hold the equal-weight portfolio of every name inside its
  post-NT window and put a **Newey-West HAC *t*** (Newey & West 1987) on the daily
  market-adjusted series.
- **Matched self-control.** Pseudo-events on the *same firms* 252 trading days before each NT
  (collision-screened), Welch *t* (Welch 1947) of event vs pseudo CARs — "or do these firms
  just always drift down?".
- **Random-dates placebo.** Same tickers, uniform random dates, averaged over **25 seeds**
  (house rule: no single-seed baselines).
- **Third axis.** Practitioners say the *second* NT in a row is the real kill signal (the first
  may be a one-off audit hiccup). We flag events with a prior NT within ~15 months
  ("repeat offenders") and Welch-test repeat vs first-offense CARs.

## Survivorship — named, and it matters more than usual here

The CIK → ticker map (`company_tickers.json`) covers **currently registered** companies only.
An NT filer that later delisted — and chronic late filers delist at high rates; the very worst
outcomes (fraud, Chapter 7) *end* in delisting — cannot enter the panel. Our panel is the
survivors, so the measured drift **understates** the true sell signal. This is stated on the
Signal axis, and it is the reason the tradability discussion also flags accessibility (the
names you would most want to short are the ones hardest to borrow — Duarte-Silva et al. and
the short-selling literature, e.g. D'Avolio 2002, *JFE*, on borrow costs in distressed
small-caps).

## Data sources used here

- **EDGAR quarterly form-type master index** —
  `https://www.sec.gov/Archives/edgar/full-index/<YYYY>/QTR<q>/master.idx` (fetched once,
  2002Q1–2026Q2; exact-form NT rows cached in `_cache/nt_filings.csv`).
- **SEC CIK→ticker registry** — `https://www.sec.gov/files/company_tickers.json`
  (cached in `_cache/cik_tickers.csv`).
- **yfinance** daily auto-adjusted (total-return) closes for the event tickers + SPY,
  cached in `_cache/prices.parquet`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Method lineage (the desk's shared engine)

- Event CARs with one documented lag: [`strategy.build_event_cars`](../late_filers/strategy.py).
- Calendar-time HAC primary test: [`strategy.calendar_time_portfolio`](../late_filers/strategy.py)
  + [`strategy.hac_t`](../late_filers/strategy.py).
- Matched self-control & multi-seed placebo: [`strategy.pseudo_events`](../late_filers/strategy.py),
  [`strategy.random_dates_placebo`](../late_filers/strategy.py).
- Deterministic synthetic control with a planted post-NT drift:
  [`data.synthetic_world`](../late_filers/data.py) — machinery proof only, never market evidence.

## Related desk studies

- [565-filing-readability](../565-filing-readability/) — scores the **text style** of filings
  that arrived; this study is the **filing event** itself (the report that didn't arrive).
- [515-earnings-announcement-premium](../515-earnings-announcement-premium/) — the mirror-image
  scheduled-disclosure event (showing up on time carries a premium; failing to show up at all
  is tested here).
