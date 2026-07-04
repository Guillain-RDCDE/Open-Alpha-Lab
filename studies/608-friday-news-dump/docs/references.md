# References & literature map — Study 608 (Friday News Dump)

## The claim under test

- **The folklore.** "Companies bury bad news on Friday after the close — nobody is
  watching, the market underreacts, and the stock keeps drifting down the following
  weeks." A newsroom and PR-desk staple ("take out the trash day", a *West Wing*
  coinage), promoted to a trading claim by the academic underreaction literature.
- **DellaVigna & Pollet (2009)**, *Investor Inattention and Friday Earnings
  Announcements*, Journal of Finance 64(2), 709-749 — the source paper: Friday
  earnings announcements show a ~15% lower immediate response and ~70% higher
  delayed response (drift) than non-Friday announcements. https://doi.org/10.1111/j.1540-6261.2009.01447.x
- **Niessner (2015)**, *Strategic Disclosure Timing and Insider Trading*, Yale SOM
  working paper — firms disproportionately file negative 8-K information after
  trading hours and before weekends; the *hiding* margin itself.
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2439040
- **doyle & Magilke (2009)**, *The Timing of Earnings Announcements: An Examination
  of the Strategic Disclosure Hypothesis*, The Accounting Review 84(1) — finds
  little support for strategic Friday timing of *earnings*; the counter-voice.
- **Michaely, Rubin & Vedrashko (2016)**, *Further evidence on the strategic timing
  of earnings news: Joint analysis of weekdays and times of day*, Journal of
  Accounting & Economics 62(1) — the Friday earnings effect weakens once time-of-day
  and firm composition are controlled; the replication-era caution.
- **deHaan, Shevlin & Thornock (2015)**, *Market (in)attention and the strategic
  scheduling and timing of earnings announcements*, JAE 60(1) — attention is lower
  after hours and on Fridays; managers exploit it.

Our contribution is the desk treatment on the *disclosure-timing* margin with
unambiguous bad news: a stratified panel of negative 8-K classes (Item 4.02
non-reliance, Item 2.06 impairments, Item 5.02 CEO exits) with EDGAR acceptance
timestamps to the second, a Welch test of the Friday-vs-weekday drift gap, a
label-permutation placebo, an honest short-the-dump cost stack, and the hiding
margin tested against an Item 2.02 earnings control.

## The event panel (the frozen input)

- **Scope rule (fixed before looking at returns).** Pure `8-K` filings (amendments
  excluded) accepted 2004-08-23 (the SEC's expanded 8-K item regime, when Items
  2.06 / 4.02 / 5.02 came into force) through 2026-06-30, harvested from the EDGAR
  full-text search API by item-title phrase and *verified against the filing's item
  codes*: `nonreliance` = Item 4.02, `impairment` = Item 2.06, `ceo_exit` =
  Item 5.02 with a resignation phrase; control `earnings` = Item 2.02.
- **Sampling cap (documented, weekday-orthogonal).** Per class × calendar quarter,
  the first 12 eligible FTS hits (8 for the control) are kept — heavy quarters
  (the 2005-2007 restatement wave) are *sampled*, not exhausted. See
  [`data.py`](../friday_news_dump/data.py).
- **Timing.** The EDGAR **acceptance timestamp** (Eastern, to the second) from each
  filing's index page — the moment the filing became publicly retrievable.
  `friday` = accepted on a Friday; `after_close` = accepted ≥ 16:00 ET;
  `friday_pm` = both.
- **Sources.** EDGAR full-text search https://efts.sec.gov/LATEST/search-index?q=...
  (coverage 2001→); filing index pages under https://www.sec.gov/Archives/edgar/data/;
  CIK→ticker via https://www.sec.gov/files/company_tickers.json; daily adjusted
  closes via yfinance (tickers + SPY).

## Method notes

- **Event time.** Day 0 = the first trading session whose *close* reflects the
  filing (accepted before 16:00 ET on a trading day = that day; otherwise the next
  session — Friday-PM filings land on Monday). One convention, applied everywhere.
- **Abnormal returns.** Daily stock return minus SPY (market-adjusted model);
  reaction = AR(day 0), drift = CAR[+1..+10]. Welch (1947) *t* on the Friday vs
  Mon-Thu drift gap is the Signal-axis statistic; a seeded 2,000-draw
  label-permutation placebo (Fisher randomization logic) sits beside it, and a
  1%-winsorized Welch *t* guards the small-cap tails.
- **Survivorship — named.** CIK→ticker uses the SEC's *current* map and yfinance
  carries only listed histories: firms that died after their bad news are absent,
  which *understates* any bad-news drift. The hiding margin (third axis) is
  computed on the FULL filing panel (no ticker mapping), so it does not carry this
  bias.
- **Execution.** ONE lag: entry at the day-0 close (the filing is public before
  that close by construction); the day-0 crash is never captured. Costs 10 bps
  one-way × 2 stock legs + 1 bp × 2 SPY hedge legs; shorts pay borrow (5%/yr
  generic hard-to-borrow, 2%/yr shown as robustness) over the 10-session hold.

## Data sources used here

- EDGAR FTS + filing indexes + `company_tickers.json` (SEC, User-Agent declared),
  cached under `_cache/events_raw.csv`, `_cache/acceptance.csv`,
  `_cache/cik_tickers.csv`.
- yfinance daily adjusted closes, cached `_cache/prices.parquet`.
  Headline numbers pinned in [`docs/results.md`](results.md), reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup frame)

- [565-filing-readability](../565-filing-readability/),
  [566-earnings-call-tone](../566-earnings-call-tone/),
  [567-uncertainty-word-count](../567-uncertainty-word-count/) — the *text-content*
  siblings: WHAT the filing says (readability, tone, hedging words). This study is
  orthogonal: WHEN the filing is released (the timing of disclosure), with the news
  class fixed to unambiguously-bad items.
- [90-weekend](../90-weekend/) — the calendar weekend effect (all stocks, all
  Fridays). Here the Friday flag is conditioned on a *specific disclosure event*.
- [602-macro-announcement-premium](../602-macro-announcement-premium/) — scheduled
  macro announcements; this study is firm-level, unscheduled, and strategically
  timed by the discloser.
