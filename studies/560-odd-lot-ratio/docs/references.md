# References & literature map — Study 560 (Odd-Lot-Ratio)

## The claim, at full strength

- **Garfield A. Drew (1955)**, *New Methods for Profit in the Stock Market.* The canonical
  statement of the **odd-lot theory**: the small odd-lot (< 100 share) trader is chronically wrong —
  buying near tops and selling near bottoms — so a spike in odd-lot **buying** (a high odd-lot ratio)
  is a contrarian *sell* and a spike in odd-lot **selling** a contrarian *buy*. The 'Drew odd-lot
  index' popularised fading the odd-lot crowd.
- **Kewley & Stevenson (1967)** / **Klein (1974)**, early academic tests of the odd-lot statistics.
  Even in the 1960s–70s the evidence for a *profitable* odd-lot fade was mixed to negative — the
  theory was folklore that never cleanly survived out-of-sample testing.
- **Kavajecz & Odders-White**, and the market-microstructure literature on odd-lot flow, document
  that the *composition* of odd-lot volume changed fundamentally with electronic markets — odd lots
  became a footprint of algorithmic order-slicing, not retail sentiment.

## Why the signal is 'mostly dead' — the structural breaks

- **Decimalization (2001).** Sub-penny/penny tick sizes and the collapse of the round-lot convention
  made it cheap and routine for algorithms to slice large parent orders into many small child orders.
  Post-decimalization, a large fraction of odd-lot volume is *institutional/HFT* order-slicing, not
  small retail — so the odd-lot ratio no longer measures 'dumb money'.
- **Odd-lot tape reporting (2013–2014).** Odd-lot transactions were historically **excluded from the
  consolidated tape / SIP**; they were only added to the public tape in the December 2013 – 2014
  odd-lot transparency changes (and studied in the SEC/academic *odd-lot rate* literature that
  followed). For much of the theory's supposed life the series was literally unmeasurable to the
  public, and by the time it was measurable it no longer meant what the theory assumed.
- **O'Hara, Yao & Ye (2014)**, *"What's Not There: Odd Lots and Market Data."* *Journal of Finance*
  69(5). Shows odd-lot trades carry substantial (often *informed*) volume and were missing from the
  public record — direct evidence that odd-lot flow is not the naive-retail signal the theory needs.

## The odd-lot ratio we build

- The synthetic tape's `odd_lot_ratio` is the fraction of odd-lot volume that is *buying* (an
  odd-lot buy/sell imbalance in [0, 1]), generated as a persistent AR(1) in logit space so it
  clusters and autocorrelates like real sentiment. The contrarian signal is the *negated*,
  standardised **prior-week** ratio (a one-week execution lag). The single knob `fade_alpha` plants
  the old 'dumb-money' fade edge (`> 0`) or the modern null (`= 0`).

## Neighbours on this bench (the dedup map)

- **[Study 257 — AAII-Sentiment](../../257-aaii-sentiment/)** — the AAII bull-bear *survey* as a
  contrarian timing tool. Same *contrarian-retail-sentiment* family, but a survey of stated opinion,
  not order-flow; Study 560 is the odd-lot *flow* gauge and is synthetic-only (no free series).
- **[Study 335 — Buzz-Sentiment-ETF](../../335-buzz-sentiment-etf/)** /
  **[Study 392 — Glassdoor-Sentiment](../../392-glassdoor-sentiment/)** — alt-data sentiment proxies;
  Study 560 is the *classic tape-era* retail-flow gauge, and its story is *structural death*
  (decimalization + order-slicing), not weak-but-alive.
- **[Study 275 — Whisky-Cask](../../275-whisky-cask/)** /
  **[Study 273 — Lego-Returns](../../273-lego-returns/)** /
  **[Study 276 — Sneaker-Resale](../../276-sneaker-resale/)** — the desk's other *synthetic-only,
  no-free-real-tape* studies. Study 560 shares their SIGNAL-axis cap: synthetic-only can never be
  `REAL`.

## Shared method

- **Newey & West (1987)** — the HAC (heteroskedasticity-and-autocorrelation-consistent) standard
  error used for the fade-slope *t*. The odd-lot ratio is highly autocorrelated, so a naive OLS *t*
  overstates significance; the HAC *t* at a few weekly lags is the honest statistic (the house
  inference bar).
- **Welch (1947)** — the unequal-variance two-sample *t* for the panic-vs-euphoria regime spread.
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005) — the placebo null: shuffle the
  odd-lot-ratio labels against forward returns and read the slope's tail probability.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a robust
  *t* ≥ 2 on a *real* tape for `REAL`, else `WEAK`), the synthetic-only cap, one execution lag,
  gross/net labeled everywhere, and shorts paying borrow.
