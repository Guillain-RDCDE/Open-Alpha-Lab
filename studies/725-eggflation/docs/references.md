# References & literature map — Study 725 ("Eggflation")

## The claim under test

- **The pitch.** A recurring finance-media and retail-trader idea that the **avian-flu
  egg-price spike is a tradable event**: because **Cal-Maine Foods (`CALM`)**, the largest
  US shell-egg producer, is a near-pure bet on the egg price, you can buy CALM when eggs
  spike and ride the shortage. The testable version: (H₁) the retail egg-price change
  **forecasts** CALM's next-month return; (H₂) that forecast beats a placebo; (H₃) an
  egg-momentum timer beats buy-and-hold CALM *and* SPY net of costs.
- **The economics behind the steelman.** Cal-Maine's revenue is approximately
  (dozens sold) × (egg price), so a price spike is almost pure operating leverage — its
  FY2023 net income rose roughly **8×** on the 2022–23 spike. See Cal-Maine Foods 10-K /
  investor filings (SEC EDGAR, CIK 0000016160): https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000016160 .

## The egg-price series (the "real tape" we proxy)

- **BLS — Average Price Data, Eggs, Grade A, Large, per dozen, U.S. city average**, series
  `APU0000708111` (CPI Average Price program). The retail number that becomes the
  "eggflation" headline. Public but **published with a ~2-week lag** and not conveniently
  single-call API-available — hence our hardcoded, cited, *approximate* monthly
  reconstruction. https://data.bls.gov/timeseries/APU0000708111
- **USDA AMS — Egg Markets Overview** (weekly wholesale/shell-egg price report), the
  faster upstream series the equity market actually tracks:
  https://www.ams.usda.gov/market-news/egg-market-news-reports
- **USDA APHIS — Highly Pathogenic Avian Influenza (HPAI) confirmed detections** (the
  flock-cull driver behind each spike): https://www.aphis.usda.gov/livestock-poultry-disease/avian/avian-influenza/hpai-detections

### Press / data anchors used to pin the level & spike shape (cited, approximate)

- 2015 HPAI outbreak — the worst US animal-health event to that point (~50M birds);
  retail eggs peaked near **$2.97/dozen** in autumn 2015. USDA ERS retrospectives; BLS.
- Jan-2023 **record** retail price ≈ **$4.82/dozen** on the 2022–23 HPAI wave (CPI Average
  Price). Widely reported (e.g. USDA ERS *Egg Prices* commentary; BLS release).
- Feb/Mar-2025 **all-time high** ≈ **$5.90 / $6.23/dozen** on the 2024–25 HPAI wave (BLS
  Average Price Data). Widely reported.

> **Transparency.** `eggflation.data.load_egg_index` is a **small, hardcoded, approximate**
> monthly series ($/dozen) whose *path* matches the public anchors above (quiet base,
> three HPAI spikes). It is a **labelled proxy for the real BLS/USDA series, never the
> series itself**, and the study's verdict reflects that limitation.

## The tradable equity

- **Cal-Maine Foods (`CALM`, NASDAQ).** Largest US shell-egg producer; revenue tracks
  the egg price closely, making it the market's closest "listed egg price." Month-end Adj
  Close via yfinance. Continuous listed history over the sample (no survivorship screen).
- **`SPY`** — SPDR S&P 500 ETF, the benchmark the "you'd have done better" comparison
  invokes.

## Why "trade the public egg print" is the wrong default — the finance

- **Efficient markets / prices lead public statistics.** Fama (1970, 1991), *Efficient
  Capital Markets*: a liquid equity aggregates forward-looking information (wholesale
  futures, flock data, company guidance) and prices a shock *before* a lagged government
  statistic reports it. Our reverse regression (`CALM → egg print`, *t* ≈ 2) is a direct
  instance: the stock leads the print.
- **Spurious regression / two trending series.** Granger & Newbold (1974), *Spurious
  Regressions in Econometrics*; Phillips (1986). Two integrated, upward-drifting series
  (egg *level*, CALM *level*) correlate strongly by construction — the +0.76 here — with
  no implication for month-to-month prediction. The reason we test *changes*, not levels.
- **Data-snooping across lags.** White (2000), *A Reality Check for Data Snooping*;
  Sullivan, Timmermann & White (1999). A single "significant" lag (our *t*+2 = 2.02) is
  not evidence once you searched several lags — the circular-shift placebo (*p* = 0.11)
  is the correction.
- **HAC inference.** Newey & West (1987). Egg/flu shocks persist for months, so naive
  standard errors overstate significance; every *t* here is Newey-West (Bartlett, 6-lag).
- **"Trade the headline" alt-data.** The broad prior that public, co-moving alt-data
  rarely *leads* a liquid tape — same family as cardboard-box, box-office and
  sports-sentiment signals on this desk (below).

## Method lineage (the desk's shared engine)

- **HAC regression / lead-lag.** Newey-West OLS with a circular-shift placebo
  ([`strategy.predictive`](../eggflation/strategy.py),
  [`strategy.circular_shift_placebo`](../eggflation/strategy.py),
  [`strategy.reverse_lead`](../eggflation/strategy.py)). `REAL` needs HAC |t| ≥ 2 on the
  *predictive* slope **and** a placebo *p* < 0.05 — neither is met.
- **Timer + cost realism (beat 6).** One-month execution lag; excess-of-cash Sharpe; 10 bp
  one-way per switch ([`strategy.egg_momentum_timer`](../eggflation/strategy.py),
  [`strategy.apply_costs`](../eggflation/strategy.py)).
- **Deterministic synthetic control.** A fixed-seed planted-lead world
  ([`data.synthetic_world`](../eggflation/data.py)) proving the engine recovers a real
  lead — runs with no network.
- **Reproducibility.** As-of slice + content fingerprint (`quantlab.repro`), pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 358 — Watch-Index](../../358-watch-index/)**: the same *labelled-proxy* pattern
  (a cited, approximate price series + the only tradable leg) applied to luxury watches.
- **[Study 550 — Box-Office-Momentum](../../550-box-office-momentum/)** and the
  "trade the headline" alt-data family: public data that co-moves but doesn't *lead*.
- **[Study 307 — Coffee-Seasonality](../../307-coffee-seasonality/)**: a vivid, true
  commodity story (Brazilian frost) that makes a terrible *tradable* calendar.
