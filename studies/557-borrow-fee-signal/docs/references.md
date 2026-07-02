# References & literature map — Study 557 (Borrow-Fee-Signal)

## The claim, at full strength

- **Cohen, Diether & Malloy (2007)**, *"Supply and Demand Shifts in the Shorting Market."*
  *Journal of Finance* 62(5). The canonical decomposition of the shorting market into a demand and
  a **supply** (loan-fee) curve: when the *cost to borrow* rises (a demand shift into a fixed
  supply), subsequent returns are **negative** and abnormally so — the borrow-fee signal this
  study proxies. The paper shows the fee (price) carries information the *quantity* shorted does
  not.
- **D'Avolio (2002)**, *"The Market for Borrowing Stock."* *Journal of Financial Economics* 66.
  The foundational description of the equity-lending market: most stocks are *general collateral*
  (a low, floor-y fee), a right tail is *special* / hard-to-borrow with fees that can spike into
  double digits, and specialness predicts low returns. The structure our synthetic fee
  distribution mirrors (a GC floor plus a demand/scarcity-driven special tail).
- **Jones & Lamont (2002)**, *"Short-Sale Constraints and Stock Returns."* *JFE* 66. Stocks that
  are expensive to short (high borrow cost) earn *low* subsequent returns — the historical (1920s
  loan-crowd) precedent for the fee premium.
- **Drechsler & Drechsler (2016)**, *"The Shorting Premium and Asset Pricing Anomalies."* NBER
  w20282. A cross-sectional **shorting-fee factor**: high-fee stocks earn low returns and the fee
  spread prices a swath of anomalies. The clearest modern statement that the *fee itself* is a
  priced signal.
- **Engelberg, Reed & Ringgenberg (2018)**, *"Short-Selling Risk."* *Journal of Finance* 73(2).
  The borrow fee is not just a level but a *risk* (fees can spike, forcing buy-ins) — part of why
  the short leg is `FRAGILE` to hold even when the level signal is right.

## Distinct from short interest — the dedup argument

- The borrow **fee** (price of shorting) and short **interest** (quantity, short-% of float) are
  correlated but *distinct*: two names with identical short interest can have very different fees
  depending on lendable **supply** (float, holder concentration, ETF/index inclusion). Cohen-
  Diether-Malloy (2007) show the fee/price move is the informative one. This study injects an
  independent supply-scarcity driver into the synthetic fee and confirms — via a **joint
  regression** — that the fee's slope survives controlling for short interest, while short
  interest's does not.

## Neighbours on this bench (the dedup map)

- **[Study 262 — Short-Interest](../../262-short-interest/)** — sorts a basket on short **% of
  float** (the *quantity* demanded) and finds a coin-flip (`None`/`Mirage`). Study 557 sorts on the
  borrow **fee** (the *price* of that demand) and explicitly tests whether the fee adds signal
  *beyond* short interest. Different variable, different mechanism (supply vs quantity).
- **Synthetic-only cousins on data-availability** — [273 Lego-Returns](../../273-lego-returns/),
  [275 Whisky-Cask](../../275-whisky-cask/), [276 Sneaker-Resale](../../276-sneaker-resale/): each
  tests a claim for which no clean free tape exists, plants the effect synthetically, proves the
  engine, and caps the verdict below REAL. This study is the microstructure member of that family
  (private lending-market data).

## Shared method

- **Welch (1947)** — the unequal-variance two-sample *t* used for the cheap-minus-expensive bucket
  spread.
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005) — the placebo null: shuffle the
  fee labels against forward returns and read the spread's tail probability.
- **Partial / incremental regression** (Frisch–Waugh–Lovell) — the joint OLS of forward return on
  both fee and short interest, isolating the fee's marginal signal.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (REAL needs a
  robust *t* ≥ 2 on the **real** tape; synthetic-only caps at WEAK/NONE), the explicit
  data-availability caveat on the SIGNAL axis, one documented execution lag, and costs one-way ×
  NAV with shorts paying **the actual observed borrow**, not a placeholder.
