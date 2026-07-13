# References & literature map — Study 712 ("CGC-graded key comics are an asset class")

## The claim under test

- **The pitch.** A recurring collectibles and finance-media claim that **CGC-graded key
  comic books are an investable asset class** — that a slabbed Action Comics #1, Amazing
  Fantasy #15, or the first appearance of a soon-to-be-movie character is a "store of
  value" that **beats the S&P**, with the 2020–2022 secondary-market melt-up offered as
  proof. The testable version: (H₁) the graded-comic price *index* out-returns SPY;
  (H₂) you can actually *buy* that return; (H₃) it survives the cost of grading,
  transacting and holding the slab.
- **The mania, in the reporting.** Record slabs printed at the top: **Amazing Fantasy #15
  CGC 9.6 sold for ~$3.6M** (Heritage Auctions, Sept 2021), briefly the most expensive
  comic ever; **Action Comics #1** later set a record at **~$6.0M** (Heritage, 2024). The
  first-appearance "key" is the unit the pitch is built on.

## The graded-comic indices (the "real tape" we proxy)

- **GoCollect — comic price indices & market reports.** The most-cited graded-comic price
  guide, tracking CGC census + realised sales for tens of thousands of issues, with a
  paid index/analytics product. https://gocollect.com/ · market reports:
  https://blog.gocollect.com/category/market-reports/ . **Not freely API-available** (the
  indices sit behind a subscription) — hence our hardcoded, cited, *approximate* annual
  reconstruction.
- **Heritage Auctions — comics & comic art archives.** The dominant auction house for
  high-end keys; every realised lot is public, but as a per-lot archive, not a
  downloadable index. https://comics.ha.com/ . Anchors the record blue-chip prints.
- **GPAnalysis (Comics Price Guide / GPA).** Realised-sale database for CGC/CBCS-graded
  books (subscription). https://www.gpanalysis.com/ .
- **CGC census & registry.** Certified Guaranty Company population data underlying the
  supply side of every key. https://www.cgccomics.com/census/ .

### Press / market anchors used to pin the level/shape (cited, approximate)

- Heritage Auctions press release, *Amazing Fantasy #15 (CGC 9.6) sells for $3.6M*
  (2021-09): the record first-Spider-Man book. https://comics.ha.com/
- Widely-reported *Action Comics #1 sets a record at ~$6.0M* (Heritage, 2024) — the
  first-Superman book and the market's ultimate blue chip.
- GoCollect market reports (2022–2025) documenting the post-peak cooling of the
  speculative middle and the two-tier (keys firm / middle soft) recovery:
  https://blog.gocollect.com/category/market-reports/

> **Transparency.** Our `comic_book_index.data.load_comic_index` is a **small, hardcoded,
> approximate** annual series (base 100 @ 2018) whose *path* matches the public anchors
> above (2019 base, 2020–21 pandemic melt-up, early-2022 peak, 2022–23 softening, 2024–25
> blue-chip stabilisation). It is a **labelled proxy for the real index, never the real
> index**, and the study's verdict reflects that limitation.

## The tradable equity proxy — and why there essentially is none

- **There is no pure-play listed comic-book equity.** The clean expressions are all
  private or delisted:
  - **CGC / Certified Collectibles Group (CCG)** — the grader whose slab defines the
    trade — is majority-owned by **Blackstone** (2021) and **private**.
  - **PSA / Collectors Universe** — the other big grader — traded as `CLCT` until it was
    **taken private in Feb-2021** by a group led by Nat Turner, D1 Capital and Steve
    Cohen; it is no longer listed.
  - **Heritage Auctions** — the dominant venue — is **private**.
- **The nearest *listed* proxy: Funko (`FNKO`, Nasdaq).** A pop-culture licensed-
  collectibles maker (Pop! figures), IPO'd Nov-2017. It is a **labelled, and frankly
  poor, proxy**: a toy company's equity is not the resale price of a CGC 9.8 key. We use
  it precisely because it is the only listed thing even adjacent to the trade — and its
  own record (a losing CAGR, an ~88% drawdown) is part of the finding.
- **`SPY`** — SPDR S&P 500 ETF, the benchmark the claim invokes.

## Why "an asset class that beats stocks" is the wrong default — the finance

- **Collectibles as investments underperform equities net of carry.** Dimson & Spaenjers
  (2011, *Ex Post: The Investment Performance of Collectible Stamps*; and the broader
  emotional-assets literature with Mei & Moses on art): collectibles earn lower
  risk-adjusted returns than equities once **storage, insurance and transaction costs**
  are charged, and carry large idiosyncratic risk. Graded comics add a **grading fee** on
  top of the usual wide spreads — the same shape, worse frictions.
- **Illiquidity & transaction costs.** Amihud & Mendelson (1986), *Asset Pricing and the
  Bid-Ask Spread*. Auction buyer's premiums + seller's commissions (or a dealer's
  buy/sell margin) make the round-trip spread on a slab an order of magnitude wider than
  an ETF's — the spread, not the headline appreciation, decides the net.
- **Bubbles and round-trips.** Shiller, *Irrational Exuberance*; Kindleberger & Aliber,
  *Manias, Panics, and Crashes*. A 2020–22 melt-up driven by stimulus, low rates, movie
  hype and social-media flipping, followed by a multi-year mean-reversion in the
  speculative middle, is the textbook speculative round-trip — *not* a permanent re-rating
  into an "asset class."
- **Survivorship in the success stories.** The viral "my slab doubled" is a holder who
  *bought a specific key before 2021 and sold near the top*; the median post-peak buyer of
  the speculative middle ate the round-trip. Selecting on winners manufactures the
  asset-class narrative.

## Method lineage (the desk's shared engine)

- **Risk/return primitives.** CAGR, annualised vol, Sharpe, max-drawdown
  ([`strategy.summarize`](../comic_book_index/strategy.py)).
- **Robust inference.** A small-sample annual-excess *t* for the index vs SPY
  ([`strategy.annual_excess_t`](../comic_book_index/strategy.py)) and a **Newey-West
  (HAC)** *t* of the monthly proxy alpha vs SPY
  ([`strategy.newey_west_alpha_t`](../comic_book_index/strategy.py)). `REAL` would require
  a HAC *t* ≥ 2 **in the proxy's favour** — it does not clear it.
- **Cost realism (beat 6).** The CGC-grading + dealer-spread + carry haircut charged once
  on NAV ([`strategy.net_of_costs_cagr`](../comic_book_index/strategy.py)).
- **Deterministic synthetic control.** A fixed-seed bubble-and-round-trip generator
  ([`data.synthetic_bubble`](../comic_book_index/data.py)) proving the engine recovers a
  planted signal — runs with no network.

## Data sources used here

- **yfinance** (Yahoo Finance) month-end Adj Close for `FNKO`, `SPY`, cached under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and
  reproduced by [`examples/verify.py`](../examples/verify.py).
- **Hardcoded comic-index series** as above (public reporting; approximate; a proxy).

## Related desk studies

- **[Study 358 — Watches](../../358-watch-index/)**: the exact same shape in a different
  collectible — a cited resale index, one listed proxy, a cost haircut that turns the
  return negative.
- **[Study 144 — Permanent-Portfolio](../../144-permanent-portfolio/)** and the
  inflation-hedge / collectibles family: "real assets as a store of value" tested honestly.
