# References & literature map — Study 723 ("Guacamole-Bowl")

## The claim under test

- **The folklore.** A recurring food-and-finance claim that the **Super-Bowl guacamole binge** (early
  February) drives a **January–February seasonal** in the avocado / produce trade — a date-certain
  demand spike you can position ahead of. The testable version: (H₁) the Jan–Feb window out-returns the
  rest of the year with *t* ≥ 2; (H₂) it sits in the extreme tail of a month-pair placebo, not the
  crowd; (H₃) a long-window timer beats buy-and-hold net of costs.
- **The binge, in the reporting.** The volume is genuinely enormous and well-documented:
  - **Hass Avocado Board / industry press** — the U.S. consumes on the order of **100+ million pounds**
    of avocados around the Super Bowl, historically the single largest guacamole-consumption weekend of
    the year; Cinco de Mayo (early May) is the second peak. https://hassavocadoboard.com/ ·
    volume/consumption data portal: https://hassavocadoboard.com/category-data/
  - **Avocados From Mexico** — the marketing body that runs the recurring Super-Bowl ad campaigns,
    explicitly built around the game-day guacamole surge. https://avocadosfrommexico.com/

## The avocado price series (the "real tape" we proxy, shape only)

- **USDA AMS Market News — Specialty Crops / Terminal Markets.** Public wholesale/terminal avocado
  price and shipment reports (Hass, by origin and size). The seasonality is the load-bearing public
  fact: winter carries heavy **Mexican Hass** supply (soft prices despite the Super-Bowl demand), and
  the annual price peak is the **late-summer supply gap** between the Mexican and Californian crops —
  *not* February. https://www.ams.usda.gov/market-news/fruits-vegetables · portal:
  https://mymarketnews.ams.usda.gov/
- **Hass Avocado Board — volume & price seasonality.** Weekly U.S. volume and average price by region;
  the basis for the "big Super-Bowl volume, small price move" picture. https://hassavocadoboard.com/
- **The Kaggle "Avocado Prices" dataset** (Hass Avocado Board weekly, 2015–2018, widely reused in
  teaching) — a convenient public snapshot of the same weekly price/volume seasonality.
  https://www.kaggle.com/datasets/neuromusic/avocado-prices

> **Transparency.** Our `guacamole_bowl.data.load_avocado_seasonal` is a **small, hardcoded,
> approximate** 12-month seasonal *index* (base 100 ≈ annual mean) whose *shape* — winter-soft,
> late-summer-peak — matches the public USDA/HAB reporting above. It is a **labelled proxy used only to
> falsify the price-surge premise**, never a live feed and never a Signal stamp.

## The tradable expression (what a public investor can actually buy)

- **Calavo Growers (`CVGW`, NASDAQ).** The pure-play packaged-avocado / fresh-produce company — the
  *intended* tradable proxy for "the avocado trade." **Currently unavailable on the Yahoo feed** (its
  daily history returns a single bar — a documented feed/symbology outage at the time of the run), which
  is *itself* part of the tradability story: the on-thesis instrument is a thin micro-cap that isn't
  even reliably quotable. Fresh Del Monte (`FDP`) is similarly truncated on the feed.
- **PepsiCo (`PEP`, NASDAQ) — the tradable leg used here.** PepsiCo's **Frito-Lay** arm is the
  Super-Bowl chip-and-dip complex — **Tostitos**, Fritos, Lay's and the branded dips that guacamole
  literally rides on — the largest liquid, long-history equity most directly geared to Super-Bowl
  snacking. A *labelled proxy*: a mega-cap snack conglomerate is not a basket of avocados, but it is the
  strongest **available** expression of the game-day-snacking trade.
- **`SPY`** — SPDR S&P 500 ETF, the benchmark and the cash-alternative the timer is raced against.

## Why "a visible calendar demand spike" is the wrong default — the finance

- **Anticipated demand is arbitraged into the supply chain, not the price.** A date-certain, nationally
  advertised consumption event gives growers, importers and retailers months of warning; they plant,
  pre-book and stock ahead, so the *marginal traded price* absorbs the spike. This is the produce-market
  analogue of the efficient-markets point that **publicly known information is already in the price**
  (Fama, 1970, *Efficient Capital Markets*).
- **Calendar anomalies and data-snooping.** Sullivan, Timmermann & White (2001, *Dangers of Data-Mining:
  The Case of Calendar Effects in Stock Returns*, *J. Econometrics*) — most published calendar seasonals
  do not survive a search-corrected test; the "best" window out of many is significant by construction.
  This motivates our **66-month-pair placebo** and the Bonferroni bar on per-month *t*-stats. See also
  Lo & MacKinlay (1990, *Data-Snooping Biases in Tests of Financial Asset Pricing Models*) and White
  (2000, *A Reality Check for Data Snooping*).
- **Seasonality that is real but untradable.** Even where a physical-commodity seasonal exists, it lives
  in inventory and futures curves (the convenience-yield / storage literature: Working 1949; Fama &
  French 1987, *Commodity Futures Prices*), and a naive long-the-calendar equity rule captures none of
  it — as the coffee frost study on this desk also found.

## Method lineage (the desk's shared engine)

- **Per-month HAC (Newey-West) *t*-stats.** `strategy.month_stats` — Bartlett-kernel long-run variance
  of the monthly-mean, with a Bonferroni bar for 12 tests. Newey & West (1987).
- **Window spread + placebo.** `strategy.window_spread_tstat` (Welch two-sample) and
  `strategy.placebo_pairs` (all C(12,2)=66 pairs, rank + *z*) — the search-corrected honesty check.
- **Block-bootstrap CI.** `strategy.spread_bootstrap_ci` — circular 12-month-block resampling to respect
  the annual seasonal structure (Politis & Romano's stationary/circular bootstrap family).
- **Timer + cost realism (beat 6).** `strategy.seasonal_timer` (calendar-known, no execution lag; cash
  earns the T-bill) and `strategy.apply_costs` (one-way × NAV); the race is on excess-of-cash Sharpe,
  like-for-like. `strategy.newey_west_alpha_t` for the proxy's alpha vs SPY.
- **Deterministic synthetic control.** `data.synthetic_world` plants a tunable Jan–Feb premium (and a
  null); the engine must recover the plant (*t* ≫ 2, placebo rank 66/66) and find nothing under the null
  — a machinery proof, never market evidence (METHODOLOGY → the inference bar).

## Data sources used here

- **yfinance** (Yahoo Finance) daily closes resampled to month-end for `PEP`, `SPY`, `^IRX`, cached
  under `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py) (fingerprint `14fdb930823d`, as-of 2026-06-01).
- **Hardcoded wholesale-Hass seasonal index** as above (public USDA/HAB reporting; approximate; a proxy).

## Related desk studies

- **[Study 307 — Coffee-Seasonality](../../307-coffee-seasonality/)**: the same shape in a soft
  commodity — a vivid, *true* weather/harvest story that makes a *terrible* calendar trade.
- **[Study 358 — Watch-Index](../../358-watch-index/)**: the labelled-proxy pattern (a cited approximate
  price series + the only tradable equities), used honestly.
