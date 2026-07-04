# References & literature map — Study 636 (Exchange-Listing-Pop)

## The claim under test

- **The folklore.** The **"Coinbase effect"**: when Coinbase announces it will list a coin, the
  coin pops double digits on the news — then gives it all back over the following weeks. The
  phrase was coined by crypto media around 2018–2021; the canonical quantification is
  **Messari Research** (Jack Purdy / Ryan Watkins, 2020–2021, *"The Coinbase Effect"*), which
  measured average ~**+90% (5-day peak) / +30–40% announcement pops** on early listings, and a
  string of press analyses of the same name (CoinDesk, The Block, Forbes — e.g. the DOGE
  Coinbase-Pro pop of June 2021). Coinbase itself acknowledged the pattern in its 2022
  listing-policy posts.
- **The mechanism.** A listing on the largest regulated US exchange is (a) an **accessibility
  shock** — a new pool of US retail/institutional buyers can suddenly hold the asset — and
  (b) a **certification signal** (Coinbase's legal/security review). Both fire **at the
  announcement**, not at the first trade; the price should jump when the news drops and any
  give-back should follow after the listing itself.
- **The insider chapter.** *DOJ, U.S. v. Ishan Wahi et al.* (S.D.N.Y., July 2022) — the first
  crypto insider-trading case, built precisely on front-running Coinbase listing announcements.
  Coinbase's response, **"Increasing transparency for new asset listings"** (coinbase.com blog,
  2022-04-28), created the public **listing roadmap** that front-loads the news by weeks — the
  era split we test.

## The equity cousin (dedup guard)

- **[249-index-inclusion](../249-index-inclusion/)** is the desk's equity sibling: the S&P 500
  **index-inclusion pop** (Shleifer 1986, Harris & Gurel 1986) — the same
  demand-shock-on-a-membership-announcement story, found there to be effectively dead ex-TSLA.
  This study is the **crypto** version with a different day-0 (an *exchange venue* listing, not
  an index membership) and a different accessibility mechanism. **[635-coinbase-premium](../635-coinbase-premium/)**
  is the Coinbase **price premium vs Binance** on BTC (institutional-flow footprint), and
  **[294-coinbase-rank](../294-coinbase-rank/)** is the Coinbase **App-Store rank** (retail
  attention) — neither touches listing events.

## Academic anchors

- Shleifer, A. (1986), *Do Demand Curves for Stocks Slope Down?*, **JF** — the original
  demand-shock-on-inclusion result the crypto folklore transplants.
- Harris, L. & Gurel, E. (1986), *Price and Volume Effects Associated with Changes in the S&P
  500*, **JF** — the reversal (give-back) half of the story.
- Foerster, S. & Karolyi, G.A. (1999), *The Effects of Market Segmentation and Investor
  Recognition on Asset Prices*, **JF** — the equity **cross-listing** premium (the accessibility
  shock in stocks); for crypto specifically: L. Ante (2019, BRL WP), *Market Reaction to Exchange
  Listings of Cryptocurrencies*, and Benedetti & Kostovetsky (2021, JCF), *Digital Tulips? Returns
  to Investors in Initial Coin Offerings* — document large listing-day abnormal returns on new
  venue listings.
- McLean, D. & Pontiff, J. (2016), *Does academic research destroy stock return
  predictability?*, **JF** — post-publicity decay; the roadmap era is the natural experiment
  here.

## Data sources used here

- **Event table (hardcoded in [`data.py`](../exchange_listing_pop/data.py)).** Day 0 per coin =
  the **first daily candle of its USD product on Coinbase Exchange**, from Coinbase's own public
  API (`GET api.exchange.coinbase.com/products/{id}/candles?granularity=86400`, fetched once
  2026-07-03) — the venue's own record of when trading started, no hand-curated blog dates.
  Filters: Coinbase-USD ∩ Binance-USDT, listing ≥ 2018-01-01, ≥ 60 days of Binance pre-history,
  listing ≥ 35 days old, and a **venue price-agreement check** (CB first close / Binance same-day
  close ∈ [0.6, 1.6]) that kills cross-venue ticker collisions. Stablecoins/wrapped assets
  excluded.
- **Price tape.** **Binance USDT spot daily klines** (`GET api.binance.com/api/v3/klines`,
  public, keyless), full history per event coin + BTCUSDT as the market column, cached once
  under `_cache/elp_prices.parquet`. Binance serves klines for pairs it later delisted, which
  softens survivorship.
- **Announcement subset (third axis).** Coinbase blog posts, per-row URLs in
  [`data.ANNOUNCEMENTS`](../exchange_listing_pop/data.py): MATIC (2021-03-09 → 03-11), ADA
  (2021-03-16 → 03-18), SOL (2021-05-20 → 06-17, delayed), DOGE (2021-06-01 → 06-03); the
  roadmap policy post (2022-04-28).

## Method lineage (the desk's shared engine)

- **BTC-adjusted CARs.** [`strategy.abnormal_returns`](../exchange_listing_pop/strategy.py) —
  coin log return minus BTC log return (market model with β = 1, the standard crypto
  event-study adjustment); windowed CARs in [`strategy.event_car`](../exchange_listing_pop/strategy.py).
- **Cluster-by-day inference.** Coinbase lists several assets per blog post; same-day events are
  one news draw. Primary *t* is a one-sample *t* on **per-cluster** CARs
  ([`strategy.cluster_by_day`](../exchange_listing_pop/strategy.py)); Welch (1947) for the
  two-sample legs.
- **Random-date placebo.** [`strategy.placebo_pvalue`](../exchange_listing_pop/strategy.py) —
  same coins, random pseudo-listing dates ≥ 90 days from the true event; empirical *p* of the
  mean CAR + Welch *t* vs the pooled placebo windows; seed-robustness over 20 seeds (desk rule).
- **One execution lag.** The listing goes live during day 0; the follower's first fill is the
  **day-0 UTC close** ([`strategy.follower_trade`](../exchange_listing_pop/strategy.py)); costs
  one-way × NAV per leg, shorts pay borrow (10%/yr alt, 3%/yr BTC).
- **Deterministic synthetic control.** [`data.synthetic_world`](../exchange_listing_pop/data.py)
  plants a knowable pop (−2..0) and fade (+1..+20); the null must stay quiet. *(Machinery proof
  only — never cited in support of a stamp.)*
