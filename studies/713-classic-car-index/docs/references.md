# References & literature map — Study 713 ("Classic cars are an asset class")

## The claim under test

- **The pitch.** A recurring wealth-management and auction-house claim that **collector
  cars are an investable asset class** — that a Ferrari 250 GTO, a Porsche 911, a Mercedes
  300 SL is a "store of value" that **beats the S&P**, with the 2009–2015 melt-up and the
  indices' apparent low volatility offered as proof. The testable version: (H₁) the
  collector-car *index* out-returns the S&P; (H₂) its low measured risk survives
  de-smoothing; (H₃) you can actually *buy* the return; (H₄) it survives the cost of owning
  and transacting the metal.
- **The boom, in the reporting.** The **Knight Frank Luxury Investment Index (KFLII)**
  repeatedly named classic cars the best-performing luxury asset of the 2010s (on the order
  of **+185%** over a decade in its early-2020s reports), with cars up **~25%** in 2022 (a
  top KFLII performer that year) before cooling to roughly flat-to-negative through 2024.

## The collector-car indices (the "real tape" we proxy)

- **HAGI — Historic Automobile Group International.** The HAGI Top Index and its marque
  sub-indices (HAGI Ferrari, HAGI Porsche) are the most-cited institutional collector-car
  benchmarks; they show the strong 2009–2015 run and a long subsequent plateau.
  https://www.historicautogroup.com/ . **Not freely API-available** — hence our hardcoded,
  cited, *approximate* annual reconstruction.
- **Knight Frank — Luxury Investment Index (KFLII).** The classic-car sleeve of Knight
  Frank's luxury-asset index, published in *The Wealth Report*. Public reporting of its
  full-year figures (cars best-of-decade; +25% in 2022; cooling into 2024) anchors our
  series. https://www.knightfrank.com/wealthreport
- **Hagerty — Price Guide & Market Rating.** The largest US collector-car valuation
  service; its Market Rating (a 0–100 heat gauge) peaked ~2015–16, dipped, and re-firmed
  into 2022 before softening. https://www.hagerty.com/media/market-trends/
- **Historic Automobile Group / auction results (RM Sotheby's, Bonhams, Gooding).** The
  underlying transaction record for the blue-chip end.

> **Transparency.** Our `classic_car_index.data.load_car_index` is a **small, hardcoded,
> approximate** annual series (base 100 @ 2005) whose *path* matches the public anchors above
> (2009–2015 melt-up, 2016–2020 plateau, 2022 bump, 2023–24 cooling). It is a **labelled
> proxy for the real index, never the real index**, and the study's verdict reflects that
> limitation.

## The tradable equity proxies (what a public investor can actually buy)

- **Ferrari NV (`RACE`, NYSE / Borsa Italiana).** The bluest-chip marque; IPO Oct-2015. A
  *labelled proxy*: a high-margin luxury-goods manufacturer's equity, **not** the auction
  price of a vintage Ferrari.
- **Aston Martin Lagonda (`AML.L`, LSE).** The listed British marque; IPO Oct-2018, a
  well-documented value destroyer since (repeated dilution, near-total drawdown). A *labelled
  proxy*: single-company equity risk, not the collector-car market.
- **`SPY`** (SPDR S&P 500 ETF, dividend-adjusted → total return) and **`^GSPC`** (S&P 500
  price-only index) — the benchmarks the claim invokes, on both a total-return and a
  price-only clock.

## Why "an asset class that beats stocks" is the wrong default — the finance

- **Collectibles under-perform equities net of carry.** Dimson & Spaenjers, *Ex Post: The
  Investment Performance of Collectible Stamps* (2011) and the broader emotional-assets
  literature (with Mei & Moses on art): collectibles earn lower risk-adjusted returns than
  equities once **storage, insurance and transaction costs** are charged, and carry large
  idiosyncratic risk. Cars are the same shape — high carry, wide auction spreads.
- **Appraisal smoothing fakes a low risk / high Sharpe.** Geltner, *Smoothing in Appraisal-
  Based Returns* (1991) and *Estimating Market Values from Appraised Values* (1993);
  Getmansky, Lo & Makarov, *An Econometric Model of Serial Correlation and Illiquidity in
  Hedge-Fund Returns* (2004). Sparse, lagged, appraisal-based indices are serially correlated
  and understate true volatility and market correlation — the direct cause of the car index's
  flattering Sharpe, undone here by AR(1) un-smoothing.
- **Illiquidity & transaction costs.** Amihud & Mendelson, *Asset Pricing and the Bid-Ask
  Spread* (1986). Auction-house buyer's premiums (~12–15%) plus seller's commissions make a
  physical car's round-trip an order of magnitude wider than an ETF's — the spread, not the
  headline appreciation, decides the net.
- **Booms and plateaus.** Shiller, *Irrational Exuberance*; Kindleberger & Aliber, *Manias,
  Panics, and Crashes*. A stimulus/low-rate-fuelled 2009–2015 melt-up followed by a decade of
  drift is a speculative re-rating that spent itself, not a permanent "asset class."
- **Survivorship in the success stories.** The viral "my 911 tripled" is a holder of a
  *specific* blue-chip chassis who bought before 2015; the median post-boom buyer ate the
  plateau. Selecting on winners manufactures the asset-class narrative.

## Method lineage (the desk's shared engine)

- **Risk/return primitives.** CAGR, annualised vol, Sharpe, max-drawdown
  ([`strategy.summarize`](../classic_car_index/strategy.py)).
- **Robust inference.** A paired annual-excess *t* for the index vs the S&P (both TR and
  price-only) ([`strategy.annual_excess_t`](../classic_car_index/strategy.py)) and a
  **Newey-West (HAC)** *t* of the monthly proxy alpha vs `SPY`
  ([`strategy.newey_west_alpha_t`](../classic_car_index/strategy.py)). `REAL` would require a
  HAC *t* ≥ 2 **in the cars' favour** — none clears it.
- **Appraisal de-smoothing.** Geltner AR(1) un-smoothing
  ([`strategy.desmooth_returns`](../classic_car_index/strategy.py)) — the vol/Sharpe de-bias.
- **Cost realism (beat 6).** The auction-spread + carry haircut charged once on NAV
  ([`strategy.net_of_carry_cagr`](../classic_car_index/strategy.py)).
- **Deterministic synthetic control.** A fixed-seed boom-then-plateau generator
  ([`data.synthetic_boom`](../classic_car_index/data.py)) proving the engine recovers a
  planted signal — runs with no network.

## Data sources used here

- **yfinance** (Yahoo Finance) month-end levels for `RACE`, `AML.L`, `SPY` (total return)
  and `^GSPC` (price only), cached under `_cache/`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).
- **Hardcoded car-index series** as above (public reporting; approximate; a proxy).

## Related desk studies

- **[Study 358 — Watches are an asset class?](../../358-watch-index/)** — this exact "passion
  asset beats stocks" teardown in a different garage.
- **[Study 144 — Permanent-Portfolio](../../144-permanent-portfolio/)** and the
  inflation-hedge / real-assets family: "a store of value that beats stocks" tested honestly.
