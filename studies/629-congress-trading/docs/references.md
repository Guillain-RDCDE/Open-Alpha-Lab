# References & literature map — Study 629 (Congress Trading)

## The claim under test

- **The seminal paper.** Alan J. Ziobrowski, Ping Cheng, James W. Boyd & Brigitte J. Ziobrowski,
  *Abnormal Returns from the Common Stock Investments of the U.S. Senate* (2004, **Journal of
  Financial and Quantitative Analysis** 39(4), 661–676). Headline: senators' stock **purchases**
  beat the market by roughly **85 bps/month (~10–12%/yr)** over 1993–1998 — "senators trade like
  corporate insiders." The House sequel: Ziobrowski, Boyd, Cheng & Ziobrowski, *Abnormal Returns
  from the Common Stock Investments of Members of the U.S. House of Representatives* (2011,
  **Business and Politics** 13(1)).
- **The crucial fine print.** Ziobrowski measures from the **transaction date** — a date nobody
  outside the senator's office can trade on. The only *replicable* claim is the **disclosure-date**
  version, which is what this study tests: buy at the first close after the PTR hits the public
  record (median **19 days** after the trade on our tape).

## The modern counter-evidence (what our tape agrees with)

- **Eggers & Hainmueller**, *Capitol Losses: The Mediocre Performance of Congressional Stock
  Portfolios* (2013, **Journal of Politics** 75(2)): on 2004–2008 holdings, members of Congress
  **underperform** by ~2–3%/yr — the exact opposite of the meme.
- **Belmont, Sacerdote, Sehgal & Van Hoek**, *Do Senators and House Members Beat the Stock
  Market? Evidence from the STOCK Act* (2022, **Journal of Public Economics** 207): post-2012
  disclosed trades show **no excess returns** at the transaction *or* the disclosure date.
- **The STOCK Act (2012)** — the Stop Trading on Congressional Knowledge Act — is what makes
  this study possible at all: it forces PTR disclosure within 30–45 days on
  efdsearch.senate.gov, the source the Senate Stock Watcher scraper reads.
- **The 2020 scandal (the meme's fuel).** The DOJ/SEC probe of senators' COVID-era trading —
  Richard Burr, Kelly Loeffler, James Inhofe, Dianne Feinstein (reported by NYT/ProPublica,
  March–May 2020) — plus the separate DOJ review of David Perdue's Cardlytics trades (NYT,
  2020-06-16). Burr and Feinstein have no ticker-resolvable stock *purchases* on this tape, so
  our "famous" third-axis subset is **Perdue, Loeffler, Inhofe**.

## Method citations

- **Calendar-time portfolios for overlapping event windows.** Eugene F. Fama, *Market
  efficiency, long-term returns, and behavioral finance* (1998, **JFE** 49) — the
  buy-and-hold-abnormal-return (BHAR) pooled t is unreliable under overlap; the calendar-time
  portfolio is the fix. Brad Barber & Terrance Odean, *Trading is Hazardous to Your Wealth*
  (2000, **JF**) for the calendar-time alpha convention we mirror.
- **HAC standard errors.** Whitney K. Newey & Kenneth D. West, *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (1987,
  **Econometrica** 55(3)). Daily excess returns of a rolling portfolio are serially dependent by
  construction; the verdict t uses NW with 10 lags.
- **Group splits.** B. L. Welch, *The generalization of "Student's" problem when several
  different population variances are involved* (1947, **Biometrika**) — party / size / famous
  splits. The famous-vs-rest pooled Welch t is itself the third axis's cautionary tale: Welch
  assumes independent observations, which massively-overlapping event windows are not.
- **Random baselines averaged over ≥ 20 seeds** (house rule): the random-dates placebo
  (same tickers, random disclosure dates) isolates *timing* information from the *stock list*
  and is averaged over 20 seeds.

## Data sources used here

- **Senate Stock Watcher** — <https://senatestockwatcher.com>, data repo
  <https://github.com/timothycarambat/senate-stock-watcher-data> (scraped from the Senate's own
  <https://efdsearch.senate.gov>). We parse every per-day `transaction_report_for_*.json` (the
  filename carries the **disclosure date** — the aggregate file omits it), keep stock purchases
  with a resolvable ticker: **2,776 events, 2015-01→2021-03, 26 senators**. The scraper stopped
  updating in **March 2021** — a hard coverage bound, stated on the front card. Cached as
  `_cache/senate_purchases.csv`.
- **yfinance** daily auto-adjusted (total-return) closes for the 438 resolvable tickers + SPY,
  2013-06 → 2022-12, cached as `_cache/px_panel.parquet`. Tickers that no longer resolve
  (delisted/acquired) drop ~20% of events — **survivorship**, named on the Signal axis. One
  recycled ticker (ITC) is dropped by an integrity guard documented in
  [`data.py`](../congress_trading/data.py).
- Senator → party map hardcoded from the US Senate biographical directory
  (bioguide.congress.gov) in [`data.py`](../congress_trading/data.py).

## Related desk studies (dedup map)

- [263-insider-buying](../263-insider-buying/) — the structural sibling with a different actor:
  **corporate insiders** (Form 4 filers) trading their *own* companies. This study is
  **politicians** (Senate PTR filers) trading *any* listed stock — different filer, different
  form, different information story (committee knowledge vs firm knowledge).
- [313-geopolitical-shock](../313-geopolitical-shock/) — politics-adjacent event machinery, but
  on macro shock dates, not personal trading disclosures.
- [515-earnings-announcement-premium](../515-earnings-announcement-premium/) — same
  calendar-time discipline applied to a scheduled-event premium.
