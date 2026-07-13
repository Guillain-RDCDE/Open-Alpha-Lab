# References & literature map — Study 716 ("Short the diamond")

## The claim under test

- **The pitch.** A widely-circulated bear thesis in luxury and finance media: **lab-grown
  diamonds are destroying the natural-diamond business** — chemically identical, ~90%
  cheaper, and their price has collapsed — so natural prices are in structural decline and
  you can profit from it: **short the diamond complex, short the miners, or buy a
  beaten-down miner for the rebound.** The testable version: (H₁) the natural-diamond price
  *index* is falling vs SPY (the diagnosis); (H₂) a listed proxy carries a harvestable
  alpha; (H₃) the short survives borrow and the physical stone survives the resale spread.
- **The collapse, in the reporting.** Natural polished-diamond prices peaked in **early
  2022** and fell materially through 2023–2024; the Rapaport 1ct RAPI dropped roughly
  **−18% (2023)** and **−11% (2024)**; De Beers cut rough-diamond prices repeatedly across
  2023–2024. Lab-grown *wholesale* prices collapsed roughly **80–90%** as capacity scaled,
  and lab-grown took a large share of the US engagement-ring market.

## The natural-diamond price indices (the "real tape" we proxy)

- **Rapaport — RapNet Price Index (RAPI).** The reference wholesale polished-diamond price
  index (the "Rapaport price list"), quoted by the trade for 0.30/0.50/1.00/3.00 ct.
  https://rapaport.com/ · https://rapaport.com/rapnet-price-index/ . **Not freely
  API-available** (paywalled trade data) — hence our hardcoded, cited, *approximate* annual
  reconstruction.
- **IDEX Online — Polished Diamond Price Index.** A widely-cited polished price barometer.
  https://www.idexonline.com/ (Diamond Index).
- **Paul Zimnisky — Global Rough & Polished Diamond Price Indexes.** The most-cited
  independent analyst series and commentary on the natural-vs-lab-grown split.
  https://www.paulzimnisky.com/ · the "Global Natural Diamond Price Index" and lab-grown
  price tracking.
- **Bain & Company × AWDC — annual Global Diamond Industry Report.** The standard
  institutional read on rough/polished supply, demand, and the lab-grown share.
  https://www.bain.com/insights/global-diamond-industry-report/

### Press anchors used to pin the level/shape (cited, approximate)

- Reuters, on De Beers repeatedly cutting rough-diamond prices through 2023–2024 as demand
  weakened and lab-grown pressured the low end (multiple 2023–2024 dispatches).
- Bloomberg / trade press on the ~80–90% collapse in lab-grown *wholesale* prices and the
  narrowing natural-vs-lab-grown price relationship (2022–2024).
- The New York Times / The Economist coverage of lab-grown taking a large share of the US
  engagement market and the resulting pressure on natural-diamond retail pricing (2023–2024).

> **Transparency.** Our `diamond_price_index.data.load_price_index` is a **small, hardcoded,
> approximate** annual series (base 100 @ 2018) whose *path* matches the public anchors
> above (2019 midstream softness, 2020–21 recovery, early-2022 peak, 2022–25 decline). It is
> a **labelled proxy for the real index, never the real index**, and the study's verdict
> reflects that limitation — including that a smooth reconstructed series *inflates* any
> *t*-statistic built on it.

## The tradable equity proxies (what a public investor can actually trade)

- **Signet Jewelers (`SIG`, NYSE).** The largest US diamond *jeweler* (Kay, Zales, Jared) —
  the demand side. Note that jewelers happily retail lab-grown as well, insulating margins.
  A *labelled proxy*: a retailer's equity, not a polished stone's price.
- **Lucara Diamond (`LUC.TO`, TSX).** A pure-play natural-diamond *miner* built around the
  Karowe mine in Botswana (famous for exceptionally large stones) — the supply side and the
  cleanest listed "short the diamond" expression. A *labelled proxy*: one mine's
  idiosyncratic equity, not diamond beta. (De Beers itself is unlisted, held within **Anglo
  American** `AAL.L` / `NGLOY`, a diversified miner — noted as a diluted alternative.)
- **`SPY`** — SPDR S&P 500 ETF, the benchmark the claim invokes.

## Why "an obvious collapse" still isn't a trade — the finance

- **A correct diagnosis is not an edge.** The efficient-markets baseline (Fama, 1970) and
  the limits-of-arbitrage literature (Shleifer & Vishny, 1997, *The Limits of Arbitrage*)
  explain why a widely-known, structurally-declining price can already be in the securities
  and yet be uncapturable through the available instruments.
- **Shorting costs, and convexity works against you.** A short position pays **borrow**,
  which is punishing for hard-to-borrow small caps (D'Avolio, 2002, *The Market for Borrowing
  Stock*), and a short's payoff is **concave** in the underlying — large up-moves inflict
  outsized losses and impose a volatility drag that can turn a directionally-correct short
  negative. Illiquidity compounds it (Amihud & Mendelson, 1986, *Asset Pricing and the
  Bid-Ask Spread*).
- **Collectibles / physical goods underperform net of carry.** Dimson & Spaenjers and the
  emotional-assets literature (with Mei & Moses): physical passion assets earn lower
  risk-adjusted returns than equities once **storage, insurance and transaction costs** are
  charged. A diamond's **retail→resale** round-trip (~50–70% of retail) is the extreme case —
  the classic "a diamond is not an investment" fact.
- **Single-name idiosyncrasy ≠ theme exposure.** A pure-play miner's equity is dominated by
  mine-specific, balance-sheet and financing risk; reading its path as "the diamond trade"
  is a selection error, and the sign of the bias is not obvious a priori.

## Method lineage (the desk's shared engine)

- **Risk/return primitives.** CAGR, annualised vol, Sharpe, max-drawdown
  ([`strategy.summarize`](../diamond_price_index/strategy.py)).
- **Robust inference.** A small-sample annual-excess *t* for the index vs SPY
  ([`strategy.annual_excess_t`](../diamond_price_index/strategy.py)) and a **Newey-West (HAC)**
  *t* of the monthly proxy alpha vs SPY
  ([`strategy.newey_west_alpha_t`](../diamond_price_index/strategy.py)). `REAL` would require
  a HAC *t* ≥ 2 **in the trade's favour** — neither proxy clears it.
- **Cost realism (beat 6).** A borrow-charged short book
  ([`strategy.short_book_from_returns`](../diamond_price_index/strategy.py)) and the
  retail→resale haircut charged once on NAV
  ([`strategy.net_of_resale_cagr`](../diamond_price_index/strategy.py)).
- **Deterministic synthetic control.** A fixed-seed collapse generator
  ([`data.synthetic_collapse`](../diamond_price_index/data.py)) proving the engine recovers a
  planted *down*-trend — runs with no network.

## Data sources used here

- **yfinance** (Yahoo Finance) month-end Adj Close for `SIG`, `LUC.TO`, `SPY`, cached under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and
  reproduced by [`examples/verify.py`](../examples/verify.py).
- **Hardcoded natural-diamond price-index series** as above (public reporting; approximate;
  a proxy).

## Related desk studies

- **[Study 358 — Watches are an asset class?](../../358-watch-index/)** — the same shape in
  luxury watches: a cited approximate resale index + listed proxies, undone by carry.
- **[Study 301 — Triple-RSI](../../301-triple-rsi/)** — the survivorship/selection signature
  where a thin, selected outcome is narrated as a system.
