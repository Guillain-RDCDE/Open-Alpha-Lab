# References & literature map — Study 727 ("the maple-syrup reserve as a trade")

## The claim under test

- **The pitch.** A recurring "did you know" that treats the **Quebec strategic maple-syrup
  reserve** as evidence that maple is a real, ownable soft commodity — a cartel and a
  price-defending stockpile, therefore surely a *trade* (a store of value, or a
  weather-driven sugaring-season seasonal). The testable version: (H₁) the administered
  bulk price out-returns stocks; (H₂) a listed proxy carries maple-attributable alpha;
  (H₃) there is a tradable Feb–Apr seasonal.

## The reserve, the cartel, the price (the "real tape" we proxy)

- **Producteurs et productrices acéricoles du Québec (PPAQ)** — formerly the *Fédération
  des producteurs acéricoles du Québec (FPAQ)* — the mandatory-membership producers' body
  that sets Quebec bulk-syrup quotas and negotiates the bulk price, and operates the
  **Réserve stratégique mondiale de sirop d'érable** (Global Strategic Maple Syrup
  Reserve). https://ppaq.ca/ · reserve overview: https://ppaq.ca/en/selling-buying-maple-syrup/the-strategic-reserve/
- **Quebec's share of world output ~72%** (and the large majority of Canadian production);
  the reserve stores up to ~100 million lb across facilities (Laurierville / Plessisville).
  Government of Canada / Statistics Canada maple statistics:
  https://www.statcan.gc.ca/ (maple products; annual production & farm value).
- **The bulk price is administered, not exchange-traded.** There is **no maple futures
  contract and no live price feed**; the PPAQ *convention de mise en marché* sets the
  per-pound price by grade each season. Our `data.load_maple_price` is a small, hardcoded,
  **approximate** annual reconstruction of that negotiated price (CAD/lb, 2008–2024) — a
  **labelled proxy for the real tape, never the tape**. See the transparency note below.

### Press / reporting anchors used to pin the level & shape (cited, approximate)

- **The Great Canadian Maple Syrup Heist (2011–2012)** — ≈ **C$18.7 million** / ~3,000
  tonnes (~9,571 barrels) siphoned from the reserve at Saint-Louis-de-Blandford. Widely
  reported; see e.g. CBC coverage of the theft and 2016–2017 trial, and the *Vanity Fair*
  long-read "The Great Canadian Maple Syrup Heist" (2016).
  https://www.cbc.ca/news/canada/montreal/maple-syrup-heist (theft & trial coverage).
- **Bulk-price reporting** for the flat-then-firming path (a modest 2022–2024 catch-up as
  reserves drew down after record 2021–22 demand): PPAQ annual price bulletins and
  agricultural press (e.g. *La Terre de chez nous*, Financial Post / Reuters maple-market
  pieces). Used only to anchor the *shape*; exact year values are approximate.

> **Transparency.** `maple_syrup_reserve.data.load_maple_price` is a **small, hardcoded,
> approximate** annual CAD/lb series whose *near-flat shape* matches the public fact that the
> price is committee-set and reserve-defended. It is a **labelled proxy, never a live feed**,
> and the study's verdict reflects that limitation.

## The tradable proxies (what a public investor can actually buy)

- **Rogers Sugar Inc. (`RSI.TO`, TSX).** The parent of **Lantic Inc.**, Canada's largest
  sugar refiner, which also owns the **maple** businesses **L.B. Maple Treat** and
  **Decacer** (acquired 2017) — i.e. the only listed company with a real maple-products
  segment. A *labelled proxy*: a diversified, dividend-paying sugar refiner's equity is not
  the resale price of bulk syrup, and maple is a minority of revenue.
  Company: https://www.lanticrogers.com/
- **No. 11 sugar futures (`SB=F`).** The nearest freely-traded sweetener commodity — a pure
  **placebo**, included precisely to show that the closest tradable soft to "maple" (a) has
  nothing to do with it and (b) is far more volatile and unrewarding.
- **`^GSPTSE`** — S&P/TSX Composite, the CAD-home benchmark (RSI.TO is CAD-denominated, so
  the benchmark is CAD too — no FX mismatch).

## Why "a reserve implies a trade" is the wrong default — the finance

- **Managed/administered markets are *engineered* toward low volatility.** A buffer stock or
  marketing board exists to *dampen* price swings — the opposite of the volatility a trader
  needs. Classic buffer-stock and commodity-price-stabilisation analysis: **Newbery &
  Stiglitz (1981), *The Theory of Commodity Price Stabilization***; **Deaton & Laroque
  (1992), *On the Behaviour of Commodity Prices*, Review of Economic Studies**. A defended
  price is calm *by design*.
- **Seasonality & multiple comparisons.** Testing all 12 calendar months guarantees a false
  positive by chance; a robust seasonal needs a multiple-testing correction (Bonferroni ⇒
  ~|t| ≥ 3) or an out-of-sample confirmation. Cf. **Sullivan, Timmermann & White (2001),
  *Dangers of data mining: the case of calendar effects in stock returns*, J. Econometrics**,
  and the White (2000) Reality Check the desk uses elsewhere.
- **Robust inference for autocorrelated returns.** **Newey & West (1987)** HAC standard
  errors; the **circular block bootstrap** (Politis & Romano, 1994) for a seasonal spread CI
  that respects the annual structure.
- **Misattribution / factor exposure.** A defensive, low-beta, dividend stock earns a
  premium documented in the low-volatility-anomaly and dividend literatures
  (**Frazzini & Pedersen (2014), *Betting Against Beta***); attributing Rogers Sugar's
  outperformance to *maple* rather than to that exposure is a category error the study makes
  explicit.

## Method lineage (the desk's shared engine)

- **Risk/return primitives.** CAGR, annualised vol, Sharpe, max-drawdown
  ([`strategy.summarize`](../maple_syrup_reserve/strategy.py)).
- **Robust inference.** A small-sample annual-excess *t* for maple vs TSX
  ([`strategy.annual_excess_t`]), a **Newey-West (HAC)** *t* of the monthly proxy alpha
  ([`strategy.newey_west_alpha_t`]), per-month HAC *t* ([`strategy.month_stats`]), a Welch
  season test ([`strategy.season_tstat`]) and a **circular block-bootstrap** CI on the
  season-minus-rest spread ([`strategy.season_bootstrap_ci`]). `REAL` would require a HAC
  *t* ≥ 2 **tied to maple** — nothing clears it.
- **Cost realism (beat 6).** A calendar-known sugaring timer, costs one-way × NAV, flat
  months earning the benchmark ([`strategy.seasonal_timer`], [`strategy.apply_costs`]).
- **Deterministic synthetic control.** A fixed-seed planted-season generator
  ([`data.synthetic_world`]) proving the engine recovers a planted signal — runs with no
  network.

## Data sources used here

- **yfinance** (Yahoo Finance) month-end Adj Close for `RSI.TO`, `SB=F`, `^GSPTSE`, cached
  under `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and
  reproduced by [`examples/verify.py`](../examples/verify.py).
- **Hardcoded PPAQ bulk-price series** as above (public reporting; approximate; a proxy).

## Related desk studies

- **[Study 307 — Coffee-Seasonality](../../307-coffee-seasonality/)** — the same seasonality
  machinery on a *real* futures market with a *real* (tiny) frost/harvest seasonal.
- **[Study 358 — Watch-Index](../../358-watch-index/)** — the same labelled-proxy shape for a
  collectible with no tradable index; maple sits a rung below (no index *and* no volatility).
