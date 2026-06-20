# References & literature map — Study 326 (Stablecoin-Depeg)

## The claim under test

- **"A stablecoin breaking the buck is a crypto-wide sell signal."** A recurring
  trading-desk and crypto-Twitter heuristic: when a major stablecoin loses its peg, it
  signals systemic stress (liquidity drying up, collateral doubts, contagion), so the whole
  market is about to fall — sell BTC/ETH on the depeg. The two canonical reference events
  are the **TerraUSD (UST) / Luna** algorithmic death spiral of May 2022 and the **USDC**
  break to ~$0.88 over the SVB-failure weekend of March 2023. We test it as an event study:
  the cumulative abnormal return of a BTC+ETH basket around each depeg date, against a
  synthetic control of random non-event windows.

## The two events, documented

- **TerraUSD / Luna collapse (May 2022).** Liu, Makarov & Schoar (2023), *Anatomy of a
  Run: The Terra Luna Crash* (NBER Working Paper 31160) — the algorithmic-stablecoin run
  that wiped out ~$40bn and is widely credited with triggering the broader 2022 crypto
  deleveraging (3AC, Celsius, Voyager). An **endogenous, crypto-native** failure.
- **USDC depeg over the SVB weekend (March 2023).** Circle disclosed that $3.3bn of USDC
  reserves were held at the failed Silicon Valley Bank; USDC traded to ~$0.88 on 2023-03-11
  before the FDIC backstop restored the peg. See Circle's 2023-03-11 statement and FDIC /
  Federal Reserve press releases on the SVB resolution (2023-03-12/13). An **exogenous,
  TradFi-originated** shock — and, notably, one that coincided with a crypto *rally* as
  capital fled the banking system. The opposite sign to UST is the heart of this study.

## Event-study methodology (the engine behind Beat 4)

- **Event-study design.** MacKinlay (1997), *Event Studies in Economics and Finance*
  (Journal of Economic Literature) — the canonical reference for abnormal-return windows
  and cumulative abnormal returns (CAR). Brown & Warner (1985), *Using Daily Stock Returns:
  The Case of Event Studies* (Journal of Financial Economics) — daily-return event-study
  practice, the constant-mean-return benchmark, and the small-sample / non-normality
  cautions that bite hardest at tiny n. Crypto has no clean multi-factor model, so we use
  the assumption-light constant-mean benchmark and a **synthetic-control** null rather than
  a market-model residual.
- **The small-sample wall.** With only two undisputed market-wide depegs, no
  autocorrelation-robust *t* is even definable, and bootstrap intervals are uninformative.
  This is the textbook event-study failure mode of a rare-event study; we make it explicit
  rather than papering over it with a parametric p-value on n = 2.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West *t*-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.hac_tstat`](../stablecoin_depeg/strategy.py). Returns NaN below n = 3 by
  design: the real two-event sample cannot certify anything.
- **Circular block bootstrap.** Politis & Romano (1992), *A Circular Block-Resampling
  Procedure for Stationary Data* — the CI on the mean CAR
  ([`strategy.block_bootstrap_ci`](../stablecoin_depeg/strategy.py)).
- **Synthetic / placebo control.** The randomly-placed non-event windows
  ([`strategy.null_car_distribution`](../stablecoin_depeg/strategy.py)) are a placebo /
  permutation test in the spirit of Abadie, Diamond & Hainmueller (2010), *Synthetic Control
  Methods for Comparative Case Studies* (JASA) — does the treated window look unusual
  against untreated ones drawn from the same series?

## Data sources used here

- **Yahoo! Finance daily closes** (via `yfinance`), `auto_adjust=True`, BTC-USD from 2017
  and ETH-USD from late-2017; equal-weight daily-return basket. Spot crypto has no dividend,
  so price-only == total-return. All headline numbers carry an as-of date (2026-05-31, the
  in-progress month dropped) and content fingerprints (see [`docs/results.md`](results.md)).
  The offline reproducible core and test-suite run on the deterministic
  [`data.synthetic_tape`](../stablecoin_depeg/data.py) generator, never the network.

## Related desk studies

- **[Study 295 — Stablecoin-Supply](../../295-stablecoin-supply/)**: the *other* stablecoin
  thesis — does aggregate stablecoin **supply growth** ("dry powder") forecast BTC? That is
  a continuous predictive-regression / timing study on a monthly series; **this** study is a
  discrete **event study** around two depeg dates. Distinct input (peg break vs supply
  level), distinct method (CAR/synthetic-control vs predictive slope), distinct verdict.
- **[Study 291 — Doge-Tweets](../../291-doge-tweets/)**: another crypto event study around
  dated public shocks — same family of "does a salient crypto headline move the market"
  questions.
