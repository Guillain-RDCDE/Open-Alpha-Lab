# References & literature map — Study 634 (US-Leads-the-World)

## The claim under test

- **The folklore.** *"When America sneezes, the world catches a cold"* — today's US close
  predicts tomorrow's Tokyo, Frankfurt, London and Sydney sessions, because their trading
  days **start after New York's ends**. The mechanism *is* the time zone: by the Nikkei's
  opening bell (~19:00 ET), the full US session is public information that Tokyo has not
  yet been able to price in its own trading.
- **The seminal empirical papers.**
  - Yasushi Hamao, Ronald W. Masulis & Victor Ng, *Correlations in Price Changes and
    Volatility across International Stock Markets* (1990, **Review of Financial Studies**
    3(2), 281–307) — the classic New York → Tokyo → London spillover study; US returns and
    volatility transmit to the next foreign session.
  - Kent G. Becker, Joseph E. Finnerty & Manoj Gupta, *The Intertemporal Relation Between
    the U.S. and Japanese Stock Markets* (1990, **Journal of Finance** 45(4), 1297–1306) —
    today's S&P return explains a large share of the next Nikkei session.
  - Cheol S. Eun & Sangdal Shim, *International Transmission of Stock Market Movements*
    (1989, **JFQA** 24(2), 241–256) — VAR evidence that US innovations lead all other
    national markets; no market leads the US.
  - Wen-Ling Lin, Robert F. Engle & Takatoshi Ito, *Do Bulls and Bears Move Across
    Borders?* (1994, **RFS** 7(3), 507–538) — daytime US returns move Tokyo's **overnight
    open**, i.e. the spillover lands in the opening gap, exactly the decomposition we run.
  - David E. Rapach, Jack K. Strauss & Guofu Zhou, *International Stock Return
    Predictability: What Is the Role of the United States?* (2013, **Journal of Finance**
    68(4), 1633–1662) — the modern treatment: lagged US returns predict non-US returns far
    more than the reverse; they attribute it to gradual diffusion of US information. The
    documented cross-**timezone** effect this study replicates at the daily horizon.
- **Why it "never quite arbitrages away".** The predictable part is impounded at the next
  session's **opening auction** — a print you cannot trade ahead of. Close-to-close
  predictability can therefore persist indefinitely without leaving free money on the
  table (it is information flow, not an inefficiency). Post-publication decay logic —
  McLean & Pontiff, *Does Academic Research Destroy Stock Return Predictability?* (2016,
  **JF** 71(1)) — applies only to the *tradable* residual, which is what we find dead.

## What we measure, and the honesty choices

- **Alignment = the one execution lag.** Each US day *t* (SPY close-to-close, known at
  16:00 ET) is paired with the **first foreign session strictly after** date *t*
  (Tokyo/Sydney open 2–3 h after the US close; Frankfurt/London the next morning). The
  predictor is fully public before the target session begins — exactly one lag, by
  construction, documented in [`data.build_pairs`](../us_leads_the_world/data.py).
- **Gap vs open→close decomposition.** Next-session return = overnight gap (previous
  foreign close → open; prints before the first tradable tick) + open→close (the only leg
  an open-entry trader owns). Lin-Engle-Ito's construction, run per market.
- **Stale Yahoo opens, named.** Yahoo's daily index opens are stale (open = previous
  close) in long stretches — ^FTSE ~88%, ^AXJO ~41% of days. All open-based legs run on
  **live-open days only**, with the stale share quoted next to every number. The ASX
  "open" is additionally a **staggered 10-minute auction** (ASX Group, market phases
  documentation) — index opens embed not-yet-opened constituents at stale prices, so
  measured open→close drift there overstates what an order could capture.
- **Price-only indices, labeled.** ^N225/^GDAXI*/^FTSE/^AXJO are price levels (yfinance
  carries no dividend adjustment for them); SPY is total-return adjusted. On daily
  return-on-return slopes the dividend drift is negligible; labeled everywhere.
  (*^GDAXI is a performance index — dividends are reinvested in the level itself.)
- **HAC inference throughout.** Newey & West, *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (1987,
  **Econometrica** 55(3)) — Bartlett-kernel long-run variances on both slope and
  strategy-mean t's; the standard `floor(4*(n/100)^(2/9))` lag rule.
- **Shuffle placebo, seed-averaged.** The random baseline (US predictor permuted) is
  averaged over **50 seeds** per house rule (single-seed baselines are banned).

## The wrapper trap (the tradability catch)

- **EWJ / EWG / EWA / EWU** (iShares MSCI country ETFs) trade **during US hours**: their
  US-close price already embeds the US day contemporaneously via futures/fair-value
  pricing (see Engle & Sarkar, *Premiums-Discounts and Exchange-Traded Funds*, 2006,
  Journal of Derivatives, on international ETF pricing vs stale NAVs). Buying the wrapper
  at the next US open buys the spillover **after** the foreign market has already opened,
  gapped and closed — the time zone that creates the predictability is what the accessible
  vehicle takes away.

## Method lineage (the desk's shared engine)

- **OLS slope + Newey-West HAC t.** [`strategy.ols_hac`](../us_leads_the_world/strategy.py),
  mirroring `quantlab/analytics.mean_tstat_hac`'s kernel.
- **Gap/intraday decomposition + live-open guard.** [`data._per_ticker`](../us_leads_the_world/data.py)
  and [`strategy.market_summary`](../us_leads_the_world/strategy.py).
- **Feasible open-entry trade vs the phantom close-to-close backtest.**
  [`strategy.feasible_trade`](../us_leads_the_world/strategy.py) /
  [`strategy.phantom_trade`](../us_leads_the_world/strategy.py).
- **Deterministic synthetic control.** [`data.synthetic_world`](../us_leads_the_world/data.py)
  plants a known overnight-spillover beta; with beta = 0 the pipeline must find nothing.

## Data sources used here

- **yfinance** daily Open+Close bars: SPY (total-return adjusted) + ^N225, ^GDAXI, ^FTSE,
  ^AXJO (price-only index levels), 1997-01-02 → 2026-06-30, cached under
  `_cache/uslw_bars.csv`. All headline numbers pinned in [`docs/results.md`](results.md)
  and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies (and why this one is distinct)

- [379-etf-lead-lag](../379-etf-lead-lag) — **intra-US, same-timezone** daily lead-lag
  (SPY leader → smaller US members), verified **None**: within one timezone the "lead" is
  contemporaneous co-movement and the next-day link is absent. This study is the
  **cross-timezone** cousin — the sessions genuinely do not overlap, so a *predictive*
  next-session link can (and does) exist. Same-timezone folklore: None; cross-timezone
  information flow: Real — and both are untradable, for opposite reasons.
- [633-btc-vol-targeting](../633-btc-vol-targeting) and the overnight/close-to-close
  decomposition studies on the bench share the gap-vs-intraday accounting used here.
