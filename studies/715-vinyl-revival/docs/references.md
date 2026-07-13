# References & literature map — Study 715 ("Vinyl is back — a trend to trade?")

## The claim under test

- **The pitch.** A recurring lifestyle-and-markets story: the **vinyl revival** is a
  durable, tradable trend — records are "back," sales grow every year, so put money on it
  and ride the boom. The testable version: (H₁) the vinyl-revenue *trend* out-grows the
  S&P; (H₂) you can actually **buy** that trend with an edge (alpha in a listed proxy);
  (H₃) it survives the cost of expressing it — and, separately, **is the boom already
  priced?**
- **The revival, in the reporting.** U.S. vinyl revenue climbed for **18 straight years**,
  from a ≈ **$89M** low in 2010 to ≈ **$1.4B** in 2024. Vinyl **overtook CDs in revenue in
  2020** (the first time since the 1980s) and **outsold CDs in units in 2022**. It is the
  music industry's most-cited comeback story.

## The vinyl-revenue series (the "trend" we proxy)

- **RIAA — U.S. Recorded Music Revenue Statistics (year-end).** The authoritative source
  for the estimated *retail* value of vinyl LP/EP sales. The database is a public web app,
  **not a freely-pullable API**, hence our hardcoded, cited, *approximate* annual
  reconstruction. https://www.riaa.com/u-s-sales-database/ · year-end reports:
  https://www.riaa.com/reports/
- **RIAA 2024 year-end (Matthew Bass / Jackie Jones).** Vinyl revenue **+7% to ≈ $1.4B**,
  43.6M units — 18th straight year of growth; streaming ≈ **84%** of the total.
  https://www.riaa.com/wp-content/uploads/2025/03/2024-Year-End-Music-Industry-Revenue-Report.pdf
- **RIAA 2022 year-end.** Vinyl **+17% to ≈ $1.2B**; vinyl **units > CD units** for the
  first time since 1987. https://www.riaa.com/wp-content/uploads/2023/03/2022-Year-End-Music-Industry-Revenue-Report.pdf
- **RIAA 2020 year-end.** Vinyl revenue ≈ **$643M**, **exceeding CDs** for the first time
  since the 1980s. https://www.riaa.com/wp-content/uploads/2021/02/2020-Year-End-Music-Industry-Revenue-Report.pdf

> **Transparency.** Our `vinyl_revival.data.load_vinyl_index` is a **small, hardcoded,
> approximate** annual series (base 100 @ 2010) whose *path* matches the public RIAA
> anchors above. It is a **labelled proxy for the trend, never a tradable price** — and it
> measures *industry revenue*, not the resale price of any individual record. The study's
> verdict reflects that limitation.

## The tradable equity proxies (what a public investor can actually buy)

- **Warner Music Group (`WMG`, NASDAQ).** A "big three" major label — presses and owns much
  of the catalogue vinyl sells. IPO June 2020. A *labelled proxy*: a label's equity, not a
  stack of LPs, and its P&L is dominated by streaming.
- **Universal Music Group (`UMG.AS`, Euronext Amsterdam).** The largest major label. Listed
  September 2021 (spun out of Vivendi). A *labelled proxy* — same streaming-dominated shape.
- **Spotify (`SPOT`, NYSE).** The streaming pure-play the vinyl revival is supposedly a
  *reaction against* — included as the "other side" of the format war. Direct listing April
  2018. A *labelled proxy*.
- **`SPY`** — SPDR S&P 500 ETF, the benchmark the claim invokes.

## Why "a tradable trend that beats stocks" is the wrong default — the finance

- **Revenue growth is not an investable return.** A category the pitch elides: the RIAA
  series is industry *revenue*, and there is no security, ETF or index that pays it out.
  You cannot custody "the vinyl trend." Cochrane (2011, *Presidential Address: Discount
  Rates*) on the gap between a growing cash-flow *story* and a priced, tradable *claim*.
- **A revived market expands supply.** New pressings and reissues meet the demand, so
  per-record resale appreciation is far below industry revenue growth — the boom does not
  make *your* copy scarce. Standard supply-response logic; contrast with the fixed-supply
  fantasy behind most "collectible as asset" pitches.
- **Collectibles under-perform equities net of carry.** Dimson & Spaenjers (2011, *Ex Post:
  The Investment Performance of Collectible Stamps*) and the emotional-assets literature
  (Mei & Moses 2002 on art): collectibles earn lower risk-adjusted returns once storage,
  insurance and transaction costs are charged.
- **Illiquidity & transaction costs.** Amihud & Mendelson (1986), *Asset Pricing and the
  Bid-Ask Spread*. A physical record round-trip (marketplace fees + condition/grading
  discount) is an order of magnitude wider than an ETF's — the spread, not the headline
  growth, decides the net.
- **Is it already priced?** Fama (1970), *Efficient Capital Markets*. A widely-reported,
  years-long trend is public information; if it mattered to the tradable proxies it would
  already be in their prices — and at ~7% of a ~84%-streaming business, it barely moves the
  needle either way.

## Method lineage (the desk's shared engine)

- **Risk/return primitives.** CAGR, annualised vol, Sharpe, max-drawdown
  ([`strategy.summarize`](../vinyl_revival/strategy.py)).
- **Robust inference.** A small-sample annual-excess *t* for the trend vs SPY
  ([`strategy.annual_excess_t`](../vinyl_revival/strategy.py)) and a **Newey-West (HAC)** *t*
  of the monthly proxy alpha vs SPY ([`strategy.newey_west_alpha_t`](../vinyl_revival/strategy.py)).
  `REAL` would require a HAC *t* ≥ 2 **in the trade's favour** — none of the three proxies
  clears it.
- **Cost realism (beat 6).** The collector round-trip spread + storage haircut charged once
  on NAV ([`strategy.net_of_collector_carry`](../vinyl_revival/strategy.py)).
- **Deterministic synthetic control.** A fixed-seed revival-path generator
  ([`data.synthetic_revival`](../vinyl_revival/data.py)) proving the engine recovers a
  planted signal — runs with no network.

## Data sources used here

- **yfinance** (Yahoo Finance) month-end Adj Close for `WMG`, `SPOT`, `UMG.AS`, `SPY`,
  cached under `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md)
  and reproduced by [`examples/verify.py`](../examples/verify.py).
- **Hardcoded RIAA vinyl-revenue series** as above (public year-end reports; approximate; a
  proxy for the trend, not a price).

## Related desk studies

- **[Study 358 — Watches are an asset class?](../../358-watch-index/)** — the same
  "passion asset beats stocks" shape (a cited, approximate index + listed proxies vs SPY +
  the carry haircut), tested on luxury watches.
- **[Study 144 — Permanent-Portfolio](../../144-permanent-portfolio/)** and the
  inflation-hedge / real-assets family: "a store of value that beats stocks" tested honestly.
