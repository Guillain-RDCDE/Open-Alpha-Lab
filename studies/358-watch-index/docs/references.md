# References & literature map — Study 358 ("Watches are an asset class")

## The claim under test

- **The pitch.** A recurring luxury-lifestyle and finance-media claim that **luxury
  watches are an investable asset class** — that a steel Rolex Daytona, a Patek Philippe
  Nautilus 5711 or an Audemars Piguet Royal Oak is a "store of value" that **beats the
  S&P**, with the 2020–2022 secondary-market melt-up offered as proof. The testable
  version: (H₁) the secondary-market resale *index* out-returns SPY; (H₂) you can actually
  *buy* that return; (H₃) it survives the cost of transacting and holding the metal.
- **The mania, in the reporting.** At the 2022 peak the average price of a luxury watch
  sold secondhand reached ≈ **$45,108**, with buyers paying up to ~5× retail for hyped
  references; a Patek Nautilus 5711 traded near **$103,357** (≈3× its $34,890 list) and a
  Rolex Daytona 116500 near **$29,000** (≈2× retail). The bid was concentrated in three
  family-owned brands — **Rolex, Patek Philippe, Audemars Piguet**.

## The secondary-market indices (the "real tape" we proxy)

- **WatchCharts — Overall Market Index.** A transaction-value-weighted index of ~300
  watches from the top-10 luxury brands. https://watchcharts.com/watches/price_index ·
  methodology: https://watchcharts.com/watches/index_methodology . **Not freely
  API-available** — hence our hardcoded, cited, *approximate* annual reconstruction.
- **Morgan Stanley × WatchCharts — quarterly Watch Market Review.** The widely-cited
  institutional read on the secondary market. Public reporting of its full-year figures
  anchors our series: secondary prices fell **−10.7%** (2023) and **−6.1%** (2024), then
  rose **+4.9%** (2025) — a peak-to-trough round-trip from the **March-2022** top.
- **Subdial — the Subdial50 / Market Overview.** An independent UK index of the 50
  most-traded references. https://subdial.com/market . Same shape: a 2022 blow-off and a
  multi-year decline.
- **Chrono24 — ChronoPulse.** Marketplace-level price-trend dashboard.
  https://www.chrono24.com/chronopulse.htm .

### Press anchors used to pin the level/shape (cited, approximate)

- CNBC, *"Secondhand luxury watch prices slump to near two-year low after a pandemic run"*
  (2023-08-03): average secondhand price **−31% since March 2022**; peak average ≈ $45,108.
  https://www.cnbc.com/2023/08/03/secondhand-luxury-watch-prices-slump.html
- WatchCharts market updates (2025–2026) for the stabilisation/turn:
  https://watchcharts.com/articles/p/9033/december-and-full-year-2025-watch-market-update ·
  https://watchcharts.com/articles/p/9189/march-2026-watch-market-update
- Robb Report, *"Why the Secondary Watch Market Is Finally Turning a Corner"*:
  https://robbreport.com/style/watch-collector/secondary-watch-market-changes-1237546673/

> **Transparency.** Our `watch_index.data.load_resale_index` is a **small, hardcoded,
> approximate** annual series (base 100 @ 2018) whose *path* matches the public anchors
> above (2019–21 melt-up, March-2022 peak, 2022–24 round-trip, 2025 stabilisation). It is
> a **labelled proxy for the real index, never the real index**, and the study's verdict
> reflects that limitation.

## The tradable equity proxies (what a public investor can actually buy)

- **Watches of Switzerland Group (`WOSG.L`, LSE).** The largest UK/US authorised dealer
  of Rolex / Patek / AP — the listed retailer most directly geared to watch demand. IPO
  May 2019. A *labelled proxy*: a retailer's equity, not a watch's resale price.
- **Compagnie Financière Richemont (`CFR.SW`, SIX).** The Swiss luxury group behind
  Cartier, Vacheron Constantin, IWC, Jaeger-LeCoultre, Piaget — exposure to the *primary*
  watch market via a diversified conglomerate. A *labelled proxy*.
- **`SPY`** — SPDR S&P 500 ETF, the benchmark the claim invokes.

## Why "an asset class that beats stocks" is the wrong default — the finance

- **Collectibles as investments underperform equities net of carry.** Dimson & Spaenjers
  (2011, *Ex Post: The Investment Performance of Collectible Stamps*; and the broader
  emotional-assets literature with Mei & Moses): collectibles earn lower risk-adjusted
  returns than equities once **storage, insurance and transaction costs** are charged, and
  carry large idiosyncratic risk. Watches are the same shape — high carry, wide spreads.
- **Illiquidity & transaction costs.** Amihud & Mendelson (1986), *Asset Pricing and the
  Bid-Ask Spread*. Dealer margins and grey-market discounts make the round-trip spread on a
  physical watch an order of magnitude wider than an ETF's — the spread, not the headline
  appreciation, decides the net.
- **Bubbles and round-trips.** Shiller, *Irrational Exuberance*; Kindleberger & Aliber,
  *Manias, Panics, and Crashes*. A 2020–22 melt-up driven by stimulus, low rates and
  social-media flipping, followed by a multi-year mean-reversion, is the textbook
  speculative round-trip — *not* a permanent re-rating into an "asset class."
- **Survivorship in the success stories.** The viral "I doubled my money" is a holder who
  *bought before March 2022 and sold near the top*; the median post-peak buyer ate the
  −36% round-trip. Selecting on winners manufactures the asset-class narrative.

## Method lineage (the desk's shared engine)

- **Risk/return primitives.** CAGR, annualised vol, Sharpe, max-drawdown
  ([`strategy.summarize`](../watch_index/strategy.py)).
- **Robust inference.** A small-sample annual-excess *t* for the index vs SPY
  ([`strategy.annual_excess_t`](../watch_index/strategy.py)) and a **Newey-West (HAC)**
  *t* of the monthly proxy alpha vs SPY ([`strategy.newey_west_alpha_t`](../watch_index/strategy.py)).
  `REAL` would require a HAC *t* ≥ 2 **in the proxy's favour** — neither clears it.
- **Cost realism (beat 6).** The dealer-spread + carry haircut charged once on NAV
  ([`strategy.net_of_carry_cagr`](../watch_index/strategy.py)).
- **Deterministic synthetic control.** A fixed-seed bubble-and-round-trip generator
  ([`data.synthetic_bubble`](../watch_index/data.py)) proving the engine recovers a planted
  signal — runs with no network.

## Data sources used here

- **yfinance** (Yahoo Finance) month-end Adj Close for `WOSG.L`, `CFR.SW`, `SPY`, cached
  under `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and
  reproduced by [`examples/verify.py`](../examples/verify.py).
- **Hardcoded resale-index series** as above (public reporting; approximate; a proxy).

## Related desk studies

- **[Study 144 — Permanent-Portfolio](../../144-permanent-portfolio/)** and the
  inflation-hedge / collectibles family: "real assets as a store of value" tested honestly.
- **[Study 301 — Triple-RSI](../../301-triple-rsi/)**: the survivorship/selection signature
  — a few winners narrated as a system.
