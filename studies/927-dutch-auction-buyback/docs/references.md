# References & literature map — Study 927 (Dutch Auction)

## The claim under test

- **The self-tender-marks-the-bottom thesis.** A *modified Dutch auction* issuer self-tender
  is the most emphatic repurchase a board can run: it posts a price range, invites holders
  to name a price inside it, and buys a large block at a single clearing price inside a
  fixed twenty-business-day window. Because the offer usually opens at a **premium** to the
  market and is funded with real cash, the folklore reads it as insiders declaring the stock
  cheap — so the stock is supposed to keep out-performing after the dust settles.
- **The steelman.** Three mechanisms could produce post-offer out-performance: a *signalling*
  effect (management's private information), a *free-cash-flow* effect (Jensen — cash paid
  out cannot be wasted), and a mechanical *EPS/float* effect from retiring a large block.
  All three are testable against the tape, and the desk tests them net of costs, one day
  late, against SPY.

## Where the claim comes from

- **Masulis (1980)**, *Stock Repurchase by Tender Offer: Signaling of Personal Tax Effects*,
  Journal of Finance — the founding event study; large positive announcement returns on
  repurchase tender offers.
- **Vermaelen (1981)**, *Common Stock Repurchases and Market Signalling*, Journal of
  Financial Economics — the signalling interpretation, with premium size as the signal.
- **Comment & Jarrell (1991)**, *The Relative Signalling Power of Dutch-Auction and
  Fixed-Price Self-Tender Offers and Open-Market Share Repurchases*, Journal of Finance —
  the paper this study is named after. Dutch-auction self-tenders carry a **smaller**
  announcement return than fixed-price tenders (roughly 8% vs 11% in their 1984–89 sample)
  precisely because the auction lets the market reveal the supply curve. Our +4.72% on
  2010–2025 sits below both, consistent with a well-documented decay of announcement
  premia as the mechanism became routine.
- **Ikenberry, Lakonishok & Vermaelen (1995)**, *Market Underreaction to Open Market Share
  Repurchases*, JFE — the source of the "and then it drifts for years" belief. Note the
  object there is **open-market** programmes, not tenders; Study 368 on this desk tests the
  authorisation version and also fails to certify the drift.
- **Peyer & Vermaelen (2005)**, *The Many Facets of Privately Negotiated Stock Repurchases*,
  JFE, and **Peyer & Vermaelen (2009)**, *The Nature and Persistence of Buyback Anomalies*,
  Review of Financial Studies — long-run abnormal returns after repurchase tenders, strongest
  among small, beaten-down value names. Our liquidity split runs in the same direction: the
  day-0 pop shrinks from +4.72% to +3.43% on names above $10m/day, i.e. the effect is largest
  exactly where you cannot trade it.

## Why the tradable legs can be empty even when the event is real

- **Fama (1998)**, *Market Efficiency, Long-Term Returns, and Behavioral Finance*, JFE — the
  standard warning that long-horizon abnormal returns are fragile to the benchmark and to
  the statistical method, and that the anomalies survive by accident of test construction.
- **Barber & Lyon (1997)**, *Detecting Long-Run Abnormal Stock Returns*, JFE, and **Kothari &
  Warner (1997)**, *Measuring Long-Horizon Security Price Performance*, JFE — buy-and-hold
  abnormal returns are skewed and **cross-sectionally correlated**, so the naive one-sample
  *t* over-rejects at long horizons. Our synthetic null reproduces exactly this: on a pure
  null panel with one shared market factor the 6-month cross-event *t* clears |2| on 7/20
  seeds, which is why the calendar-time portfolio and the placebo carry the verdict here.
- **Mitchell & Stafford (2000)**, *Managerial Decisions and Long-Term Stock Price
  Performance*, Journal of Business — the **calendar-time portfolio** as the fix for
  overlapping-event clustering; the method used in `strategy.calendar_time_portfolio`.
- **Lakonishok & Vermaelen (1990)**, *Anomalous Price Behavior Around Repurchase Tender
  Offers*, Journal of Finance — documents that the exploitable piece of the tender is a
  short-horizon phenomenon around the offer itself, not a durable post-expiry drift.

## Related desk studies (dedup)

- **[Study 368 — Buyback-Drift](../../368-buyback-drift/)**: open-market repurchase
  **authorisations** (a board approving a $X programme with no obligation to execute), 32
  hand-listed mega-caps, Weak/Fragile. Study 927 tests the *opposite end* of the buyback
  spectrum — a legally binding, cash-funded, fixed-window **auction** at a posted premium —
  on a 145-event, filing-derived sample rather than a remembered one, and separates the
  announcement pop from the tradable window rather than only measuring the drift.
- **[Study 564 — Short-Report-Event](../../564-short-report-event/)**: the same event-study
  shape (hardcoded basket, excess-of-SPY drift at 1/3/6m, placebo null) with the opposite
  sign and the short-side squeeze pathology. Study 927 shares the machinery, not the claim.
- **[Study 390 — Activist-13D](../../390-activist-13d/)**: an *external* blockholder filing
  a 13D. Here the buyer is the **issuer itself**, which is what the signalling story turns on.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite …
  Covariance Matrix*, Econometrica — [`strategy.newey_west_t`](../dutch_auction/strategy.py)
  and [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Circular block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap*, JASA —
  [`strategy.bootstrap_mean_ci`](../dutch_auction/strategy.py).
- **Jackknife.** Quenouille (1956) / Tukey (1958) — the leave-one-out *t* range in
  [`strategy.jackknife_t`](../dutch_auction/strategy.py).
- **Reproducibility stamp.** [`quantlab.repro`](../../../quantlab/repro.py) — the as-of slice
  and content fingerprint printed above every table.

## Data sources

- **Event set.** SEC EDGAR **full-text search** (`efts.sec.gov`), one query:
  `q="modified Dutch auction"`, `forms=SC TO-I`, run year by year 2010→2025. Filings are
  clustered per registrant (>150-day gap = new offer) and the earliest SC TO-I filing date
  of each cluster is the event date. Every row in `data.EVENTS` carries its **SEC accession
  number**, so the whole table is checkable filing by filing. CIK→ticker via SEC's
  `company_tickers.json`. Rule 14e-1 (twenty-business-day minimum) is the source of the
  expiry proxy.
- **Prices.** Daily **total-return** closes via `yfinance` (`auto_adjust=True`) for 128
  issuer tapes plus **SPY** (the abnormal-return benchmark) and **BIL** (the cash leg),
  2008→2026-06-30, in the shared `studies/_cache`. Total return matters here: several event
  issuers paid special dividends inside the measurement windows, and a price-only tape would
  read those as post-event under-performance.
- **Dollar volume.** A second, **unadjusted** pull (raw close × raw volume) cached as
  `dvol_<TK>_1d.parquet`, used only for the $10m/day liquidity screen. yfinance does not
  split-adjust volume, so this is an order-of-magnitude filter, not a microstructure measure.
- **As-of 2026-06-30.** The partial current month is dropped, and no event after
  2025-11-21 can appear because every event needs 147 sessions of post-tape.
