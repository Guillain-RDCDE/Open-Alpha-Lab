# References & literature map — Study 318 (Election-Volatility)

## The claim under test

- **Elections spike volatility and pay an "uncertainty premium."** The folk and
  practitioner version: US presidential elections inject political uncertainty into
  markets, so volatility rises into the vote and the option market charges a premium for
  it — which a vol seller can harvest. Brokerage and options-desk notes routinely flag the
  "election VIX bump" and the post-election "vol crush"; the implicit trade is to **sell
  volatility into the event** and collect the crush. We test all three links: does
  *realized* vol actually spike, does *implied* vol over-price it, and does an
  election-timed short-vol carry beat the volatility-risk premium you'd earn anyway.

## Election uncertainty and asset prices (the academic case)

- **Kelly, Pástor & Veronesi (2016), *The Price of Political Uncertainty: Theory and
  Evidence from the Option Market* (Journal of Finance).** The cleanest statement of the
  claim: options whose lives span major political events (elections, summits) carry higher
  implied variance and a variance-risk premium, and the premium is larger when uncertainty
  is greater. This is the steelman — it predicts an *implied*-vol effect, which is exactly
  what we find tends to be present (mean VRP +5.2) yet cannot certify on 9 US elections.
- **Pástor & Veronesi (2012), *Uncertainty about Government Policy and Stock Prices*
  (Journal of Finance); (2013), *Political Uncertainty and Risk Premia* (JFE).** The
  theory that policy/political uncertainty commands a risk premium and raises volatility,
  especially around resolution events like elections.
- **Białkowski, Gottschalk & Wisniewski (2008), *Stock Market Volatility around National
  Elections* (Journal of Banking & Finance).** A 27-country panel finding abnormal return
  *volatility* around election days — the multi-country evidence the single-country US
  realized-vol test here is too thin to reproduce (our 25-election US ratio is +5%, t≈0.5).
- **Boutchkova, Doshi, Durnev & Molchanov (2012), *Precarious Politics and Return
  Volatility* (Review of Financial Studies).** Politically sensitive industries show
  higher volatility around elections — a cross-sectional refinement of the index-level
  claim we test.

## The variance-risk premium (why the carry "works" at all)

- **Carr & Wu (2009), *Variance Risk Premiums* (Review of Financial Studies).** The
  benchmark: implied variance systematically exceeds subsequent realized variance, so
  selling variance is paid on average — the **always-on** premium our race must net out
  before crediting any *election* edge. Our finding that random-date carry earns +3.80 of
  the +6.15 election carry is this premium showing up.
- **Bollerslev, Tauchen & Zhou (2009), *Expected Stock Returns and Variance Risk Premia*
  (Review of Financial Studies).** The VRP varies through time and predicts returns; an
  "election premium" must be shown to be *more* than time-varying VRP that happens to be
  elevated around November.
- **Bakshi & Kapadia (2003), *Delta-Hedged Gains and the Negative Market Volatility Risk
  Premium* (Review of Financial Studies).** The short-vol payoff and its tail risk — the
  −23-vol-point 2008 election trade is the canonical "steamroller" this literature warns of.

## Method lineage (the desk's shared engine)

- **Event-study CAR / abnormal-window machinery.** Brown & Warner (1985), *Using Daily
  Stock Returns: The Case of Event Studies* (JFE) — the constant-baseline event-window
  design adapted here to a *volatility* (not return) statistic, with a random-date placebo
  distribution as the falsification test.
- **Realized vs implied volatility measurement.** Annualised rolling std of log returns
  for realized; the ^VIX level as the model-free implied. The VRP is implied − subsequent
  realized in vol points (a units-stable proxy for the short-variance-swap payoff).
- **Block / event bootstrap and cross-event t.** Politis & Romano (1994) stationary
  bootstrap lineage; events are far apart in time, so the resampling unit is the *event*
  once each is summarised to one number. See [`strategy.bootstrap_ci`](../election_volatility/strategy.py).
- **Excess-vs-baseline racing.** The desk rule that a premium racing against an always-on
  carry is judged on **excess of that carry**, not the carry itself — the difference
  between this study's Weak/Mirage verdict and a naive "t = 4.5, ship it."

## Data sources used here

- **^GSPC** (S&P 500 price index, 1928–2026) for realized volatility and **^VIX** (CBOE
  Volatility Index, 1990–2026) for implied volatility, via the shared `_cache` parquets
  (originally Yahoo! Finance via `yfinance`). Both are **price/level** series (no
  dividends) — labelled price-only, not total-return. All headline numbers are pinned with
  an as-of date and content fingerprint (see [`docs/results.md`](results.md)). The offline
  reproducible core and the test-suite run on the deterministic
  [`data.synthetic_daily`](../election_volatility/data.py) generator, never the network.

## Related desk studies

- **[Study 81 — Four-Year-Itch](../../81-four-year-itch/)** and
  **[Study 248 — Presidential-Honeymoon](../../248-presidential-honeymoon/)**: the
  *return-cycle* election studies. This study is deliberately the **volatility/event**
  angle — realized-vol spike, implied-vol over-pricing, and a short-vol carry — and shares
  none of their return-cycle machinery.
- **[Study 313 — Geopolitical-Shock](../../313-geopolitical-shock/)**: the sibling
  event-study design (curated event table, CAR windows, placebo distribution, one
  execution lag), here pointed at *scheduled, calendar-known* events (elections) rather
  than *surprise* shocks — and at volatility rather than the return drift.
