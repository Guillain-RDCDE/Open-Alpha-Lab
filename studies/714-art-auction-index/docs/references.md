# References & literature map — Study 714 ("Contemporary art is an asset class")

## The claim under test

- **The pitch.** A recurring wealth-management, gallery and finance-media claim that
  **contemporary / blue-chip art is an investable asset class** — that a Basquiat, a Warhol,
  a Richter or a hot young painter is a "store of value" that **beats the S&P**, is
  uncorrelated and inflation-proof, with the **Artprice** *Contemporary Art Market* report
  and the **Sotheby's Mei Moses** index offered as proof. The testable version: (H₁) the
  secondary-market auction *index* out-returns SPY; (H₂) you can actually *buy* that return;
  (H₃) it survives the cost of transacting and holding the physical work.
- **The mania, in the reporting.** The market melted up twice — 2003–2007 on cheap credit
  and a global collector class, and again in 2021–2022 post-COVID. The **Macklowe
  Collection** made **≈ $922M** across two Sotheby's sales (Nov-2021 / May-2022); a Basquiat
  (*Untitled*, 1982) sold for **$110.5M** in 2017. Then global auction turnover fell
  **~27%** in H1-2024 (Art Basel & UBS *Art Market Report* 2025 / Artprice).

## The art price indices (the "real tape" we proxy)

- **Artprice — Global Index & Contemporary Art Market report.** The most-cited commercial
  art price index (hedonic + repeat-sales on millions of auction results).
  https://www.artprice.com · the annual *The Contemporary Art Market Report*. **Not freely
  API-available** — hence our hardcoded, cited, *approximate* annual reconstruction.
- **Sotheby's Mei Moses.** The repeat-sales art index built by Jianping Mei and Michael
  Moses (NYU Stern); **acquired by Sotheby's in 2016** and taken private (no public feed
  since). https://www.sothebys.com/en/the-sothebys-mei-moses-indices . The academic basis is
  Mei & Moses (2002), below.
- **Art Basel & UBS — *The Art Market* report** (Clare McAndrew / Arts Economics), the
  annual institutional read on turnover, sell-through and dispersion.
  https://www.artbasel.com/about/initiatives/the-art-market .

### Press / report anchors used to pin the level/shape (cited, approximate)

- Sotheby's, *Macklowe Collection* — **$922M** total across the two 2021–22 sales
  (the largest single-collection total on record). https://www.sothebys.com/en/the-macklowe-collection
- Art Basel & UBS *Art Market Report 2025* (McAndrew): global auction sales down ~2024;
  the high end softened. https://www.artbasel.com/stories/art-market-report-2025
- Artprice/Artmarket annual *Contemporary Art Market* reports (2008 crash, 2021–22 boom,
  2023–24 correction). https://www.artprice.com/artprice-reports

> **Transparency.** Our `art_auction_index.data.load_art_index` is a **small, hardcoded,
> approximate** annual series (base 100 @ 2000) whose *path* matches the public anchors above
> (2000s melt-up, 2008–09 ~−44% crash, 2014 peak, 2021–22 records, 2023–24 correction). It is
> a **labelled proxy for the real index, never the real index**, and the study's verdict
> reflects that limitation.

## The tradable equity proxies (what a public investor can actually buy)

- **There is no listed auction house left.** **Sotheby's (NYSE: `BID`)** was **taken private
  by Patrick Drahi's BidFair in June 2019** ($3.7bn); **Christie's** is private (François
  Pinault / Groupe Artémis, since 1998); **Phillips** is private (Mercury Group). The absence
  is itself a finding: you cannot buy the auction-house business on an exchange.
- **MCH Group (`MCHN.SW`, SIX).** The Swiss group that **organises Art Basel** — the flagship
  contemporary-art fair (Basel / Miami Beach / Paris / Hong Kong). The single most directly
  art-market-linked listed equity. A *labelled proxy*: a fair organiser's equity, not a
  painting's hammer price.
- **Kering (`KER.PA`, Euronext).** The luxury group whose controlling shareholder **François
  Pinault (Groupe Artémis) owns Christie's** — the closest listed vehicle tied to a major
  auction house. A *labelled luxury proxy*, not art itself.
- **`SPY`** — SPDR S&P 500 ETF, the benchmark the claim invokes.

## Why "an asset class that beats stocks" is the wrong default — the finance

- **Art underperforms equities net of carry.** Mei & Moses (2002), *Art as an Investment and
  the Underperformance of Masterpieces* (*American Economic Review* 92(5)): art returned
  roughly like equities pre-cost but masterpieces *under*-performed, with high idiosyncratic
  risk. Renneboog & Spaenjers (2013), *Buying Beauty: On Prices and Returns in the Art Market*
  (*Management Science*): a real art return of ~3.97%/yr (1957–2007), **below equities**, with
  large volatility and selection effects.
- **Emotional / collectible assets.** Dimson & Spaenjers, on collectibles (wine, stamps, art):
  lower risk-adjusted returns than equities once **storage, insurance and transaction costs**
  are charged. Art is the same shape — high carry, very wide spreads.
- **The buyer's premium is enormous.** Sotheby's / Christie's charge a **~25–27% buyer's
  premium** on the first price tranche *plus* a seller's commission (~10%) — a round-trip
  haircut an order of magnitude wider than an ETF's. The premium, not the headline
  appreciation, decides the net (Amihud & Mendelson, 1986, *Asset Pricing and the Bid-Ask
  Spread*).
- **Bubbles and round-trips.** Shiller, *Irrational Exuberance*; Kindleberger & Aliber,
  *Manias, Panics, and Crashes*. Two credit/liquidity-driven melt-ups (2003–07, 2021–22) each
  followed by a multi-year mean-reversion is the textbook speculative round-trip — *not* a
  permanent re-rating into an "asset class."
- **Survivorship in the success stories.** The viral "I bought a Basquiat for $20k" is a
  holder who *bought a specific name early and sold it at a record*; the median post-peak
  buyer ate the correction. Selecting on winners manufactures the asset-class narrative.

## Method lineage (the desk's shared engine)

- **Risk/return primitives.** CAGR, annualised vol, Sharpe, max-drawdown
  ([`strategy.summarize`](../art_auction_index/strategy.py)).
- **Robust inference.** A small-sample annual-excess *t* for the index vs SPY
  ([`strategy.annual_excess_t`](../art_auction_index/strategy.py)) and a **Newey-West (HAC)**
  *t* of the monthly proxy alpha vs SPY
  ([`strategy.newey_west_alpha_t`](../art_auction_index/strategy.py)). `REAL` would require a
  HAC *t* ≥ 2 **in art's favour** — neither proxy clears it.
- **Cost realism (beat 6).** The buyer's-premium + seller's-commission + carry haircut
  charged once on NAV ([`strategy.net_of_premium_cagr`](../art_auction_index/strategy.py)).
- **Deterministic synthetic control.** A fixed-seed bubble-and-round-trip generator
  ([`data.synthetic_bubble`](../art_auction_index/data.py)) proving the engine recovers a
  planted signal — runs with no network.

## Data sources used here

- **yfinance** (Yahoo Finance) month-end Adj Close for `MCHN.SW`, `KER.PA`, `SPY`, cached
  under `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and
  reproduced by [`examples/verify.py`](../examples/verify.py).
- **Hardcoded art-index series** as above (public reporting; approximate; a proxy).

## Related desk studies

- **[Study 358 — Watches](../../358-watch-index/)**: the exact same "collectible is an asset
  class" shape — a real boom, a round-trip, and a mirage net of the dealer spread.
- **[Study 144 — Permanent-Portfolio](../../144-permanent-portfolio/)** and the
  inflation-hedge / real-assets family: "a store of value that beats stocks" tested honestly.
