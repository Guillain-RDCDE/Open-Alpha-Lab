# References & literature map — Study 711 ("A Birkin beats the S&P and gold")

## The claim under test

- **The pitch.** A recurring luxury-investment claim that an **Hermès Birkin is the
  best-performing asset on earth** — that it out-returns the S&P 500 **and** gold, with
  almost no volatility and "never a down year." The testable version: (H₁) the secondary-
  market resale *index* out-returns SPY *and* GLD; (H₂) you can actually *buy* that return;
  (H₃) it survives the cost of transacting and holding the bag.
- **The number everyone quotes — Baghunter (2016).** The viral source is a Baghunter
  ("The Hermès Birkin: The Handbag That's Better Than Gold") study reporting that Birkins
  returned an average **≈14.2%/yr over 1980–2015**, beating the S&P (quoted at ≈8.7%) and
  gold, with **no negative years**. https://baghunter.com/pages/handbags-vs-stock-market-vs-gold
  and https://baghunter.com/blogs/insights/the-hermes-birkin-the-handbag-thats-better-than-gold .
  This is the headline our study steelmans and tests — a **1980–2015, survivorship-laden,
  primary-plus-select-resale** figure, not a modern tradable series.
- **The institutional echo.** The Baghunter figure was recycled widely (Forbes, Business
  Insider, CNBC) and the "handbags as an asset class" theme appears in luxury-investment
  research such as **Credit Suisse's collectibles/luxury notes** and, as a live index, the
  **Knight Frank Luxury Investment Index (KFLII)** handbag component.

## The secondary-market handbag indices (the "real tape" we proxy)

- **Knight Frank Luxury Investment Index (KFLII) — handbags.** The most-cited institutional
  handbag index (compiled with Art Market Research). Public reporting: handbags were among
  the **strongest** luxury collectibles into 2021–22 but the **weakest performer of 2023–24**
  as the resale market cooled. https://www.knightfrank.com/wealthreport — **not freely
  API-available**, hence our hardcoded, cited, *approximate* annual reconstruction.
- **Art Market Research (AMR) — handbag indices.** The underlying data vendor for KFLII's
  handbag series. https://www.artmarketresearch.com/ .
- **Rebag "Clair" report & The RealReal resale reports.** Marketplace-level resale-value
  data showing Hermès (and the Birkin/Kelly specifically) retaining value best among
  handbags, with the broad category softening post-2022.
  https://www.rebag.com/clair/ · https://www.therealreal.com/luxury-resale-report .

### Press / auction anchors used to pin the level/shape (cited, approximate)

- Hermès primary Birkin price increases run ~**5–10%/yr**, dragging resale with them
  (routine luxury-press coverage of annual Hermès price hikes).
- Christie's / Sotheby's handbag sales: record **Himalaya Birkin** lots have exceeded
  **$300k–$500k**, but these are a thin, selected tail (survivorship), not the median bag.
- Luxury-resale cooling in 2023–24 (Business of Fashion, Reuters coverage of the post-
  pandemic luxury slowdown and Kering/Gucci weakness).

> **Transparency.** Our `birkin_index.data.load_resale_index` is a **small, hardcoded,
> approximate** annual series (base 100 @ 2015) whose *path* matches the public anchors
> above (steady retail-driven appreciation, a 2020–22 melt-up, a 2023–24 cooling, a mild
> 2025). It is a **labelled proxy for the real index, never the real index**, and the
> study's verdict reflects that limitation.

## The tradable equity proxies (what a public investor can actually buy)

- **Hermès International (`RMS.PA`, Euronext Paris).** The maker of the Birkin itself — the
  listed name most directly geared to the bag. A *labelled proxy*: the maison's equity
  (watches, leather, beauty, silk, buybacks) is not a Birkin's resale price.
- **LVMH (`MC.PA`, Euronext Paris).** Louis Vuitton / Dior / Tiffany luxury major — the
  broad luxury-demand proxy. A *labelled proxy*.
- **Kering (`KER.PA`, Euronext Paris).** Gucci / Saint Laurent / Bottega — included to show
  luxury single-stock risk cuts both ways (Gucci's slump drove a deep drawdown). A *labelled
  proxy*.
- **`SPY`** — SPDR S&P 500 ETF, and **`GLD`** — SPDR Gold Shares: the two benchmarks the
  claim explicitly names.

## Why "an asset class that beats stocks and gold" is the wrong default — the finance

- **Collectibles as investments underperform equities net of carry.** Dimson & Spaenjers
  (2011, *Ex Post: The Investment Performance of Collectible Stamps*; and the broader
  emotional-assets literature with Mei & Moses): collectibles earn lower risk-adjusted
  returns than equities once **storage, insurance and transaction costs** are charged, and
  carry large idiosyncratic risk. Handbags are the same shape — high consignment spreads.
- **Illiquidity & transaction costs.** Amihud & Mendelson (1986), *Asset Pricing and the
  Bid-Ask Spread*. Consignment houses and specialist resellers take **~25–33%**, making the
  round-trip spread on a physical bag an order of magnitude wider than an ETF's — the spread,
  not the headline appreciation, decides the net.
- **Survivorship in the success stories.** The viral "my Birkin tripled" is a rare auction-
  grade reference (Himalaya, diamond hardware) sold near a record, and an index that quietly
  drops bags that stopped trading. Selecting on winners manufactures the asset-class
  narrative; the bias points **for** the claim, so it must be corrected, not quoted.
- **Store of value ≠ return machine.** A near-zero-drawdown, ~5%/yr, inflation-tracking
  asset is a fine *store of value* — but the pitch swaps that for *out-returning equities and
  gold*, which is a different, and false, statement over the modern window.

## Method lineage (the desk's shared engine)

- **Risk/return primitives.** CAGR, annualised vol, Sharpe, max-drawdown
  ([`strategy.summarize`](../birkin_index/strategy.py)).
- **Robust inference.** A small-sample annual-excess *t* for the index vs SPY *and* GLD
  ([`strategy.annual_excess_t`](../birkin_index/strategy.py)) and a **Newey-West (HAC)** *t*
  of the monthly maison alpha vs SPY ([`strategy.newey_west_alpha_t`](../birkin_index/strategy.py)).
  `REAL` would require a HAC *t* ≥ 2 **in the bag's favour** — the best leg (RMS.PA) is 1.93.
- **Cost realism (beat 6).** The consignment-spread + carry haircut charged once on NAV
  ([`strategy.net_of_carry_cagr`](../birkin_index/strategy.py)).
- **Deterministic synthetic control.** A fixed-seed steady-compounder generator
  ([`data.synthetic_compounder`](../birkin_index/data.py)) proving the engine recovers a
  planted signal — runs with no network.

## Data sources used here

- **yfinance** (Yahoo Finance) month-end Adj Close for `RMS.PA`, `MC.PA`, `KER.PA`, `SPY`,
  `GLD`, cached under `_cache/`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).
- **Hardcoded Birkin resale-index series** as above (public reporting; approximate; a proxy).

## Related desk studies

- **[Study 358 — Watch-Index](../../358-watch-index/)**: the same "luxury object as an asset
  class" teardown for wristwatches — real spikes, brutal carry, equities win net of cost.
- **[Study 301 — Triple-RSI](../../301-triple-rsi/)**: the survivorship/selection signature —
  a few winners narrated as a system.
