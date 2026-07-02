# References & literature map — Study 587 (NFT-Floor-Beta)

## The claim under test

The folk claim, common in crypto-Twitter and alt-data marketing decks: **blue-chip NFT floor
prices (Bored Apes, CryptoPunks, Azuki, …) are a leading indicator of crypto *risk appetite* — when
floors run hot, ETH is about to run.** The sceptical prior this study encodes (the "expected lean"):
NFT floors are ETH-denominated, liquidity-driven and dominated by wash trading and reflexivity, so
they are a **lagged, high-beta echo** of ETH rather than a leading signal — floor momentum should
have no predictive edge on *forward* ETH returns.

## Academic literature on NFT pricing and its ETH linkage

- **Nadini, M., Alessandretti, L., Di Giacinto, F., Martino, M., Aiello, L. M. & Baronchelli, A.
  (2021).** "Mapping the NFT revolution: market trends, trade networks, and visual features."
  *Scientific Reports* 11, 20902. The first large-scale empirical map of the NFT market:
  heavy-tailed prices, a small set of collections dominating volume, and strong co-movement with the
  broader crypto cycle. Establishes that NFT valuations are tightly coupled to the crypto market
  they are denominated in.
- **Dowling, M. (2022).** "Is non-fungible token pricing driven by cryptocurrencies?" *Finance
  Research Letters* 44, 102097. Finds low-to-moderate *co-movement* between NFT valuations
  (Decentraland LAND) and major cryptocurrencies, with volatility spilling **from** crypto **into**
  NFTs — i.e. crypto leads, NFTs follow. Directly relevant to the lead-vs-follow question here.
- **Dowling, M. (2022).** "Fertile LAND: Pricing non-fungible tokens." *Finance Research Letters*
  44, 102096. Documents speculative, bubble-like NFT price dynamics with pronounced serial
  correlation and inefficiency — consistent with floors *echoing* and amplifying the crypto cycle.
- **Ante, L. (2022).** "The non-fungible token (NFT) market and its relationship with Bitcoin and
  Ethereum." *FinTech* 1(3), 216-224. Uses VAR/Granger-style analysis and finds that **ETH price
  shocks drive NFT sales/activity**, while the reverse effect is weak — the empirical basis for
  "floors follow ETH, not lead it".
- **Ko, H., Son, B., Lee, Y., Jang, H. & Lee, J. (2022).** "The economic value of NFT: Evidence
  from a portfolio analysis using mean-variance framework." *Finance Research Letters* 47, 102784.
  NFTs add little diversification once you account for their crypto beta and transaction frictions —
  the tradability half of the sceptical prior.

## Why the "floors lead crypto" headline does not survive

- **Denomination beta.** Floor prices are quoted *in ETH*. A floor index therefore inherits ETH's
  moves mechanically; a rising floor is often just ETH rising, restated. This study makes that
  explicit: the synthetic floor is a lagged high-beta function of ETH (beta ≈ 1.6).
- **Lag, not lead.** NFT markets are thin and slow to clear relative to spot ETH; listings and
  sales react *after* ETH moves. The lead-lag cross-correlation in this study peaks at lag −1
  (floors correlate with *yesterday's* ETH), the signature of a follower.
- **Wash trading & survivorship.** Real floor tapes are contaminated by wash trades (to farm
  token/points incentives) and by dead collections vanishing from indices — both flatter and distort
  any apparent signal. Named as the data-availability limitation on the SIGNAL axis.

## Data-availability limitation (why this study is synthetic-only)

There is no clean, free, no-key retail tape for blue-chip NFT floor prices. Floors live behind
rate-limited, key-gated marketplace / aggregator APIs (OpenSea, Blur, Reservoir, NFTGo, DappRadar),
are discontinuous, wash-traded, ETH-denominated and only a few years long. Following the desk's
other synthetic-tape alt-data studies (**273 Lego-Returns**, **275 Whisky-Cask**, **276
Sneaker-Resale**), this study builds a deterministic synthetic world and validates the engine as a
detector; the signal axis is capped at `NONE`/`WEAK` because a `REAL` stamp requires a robust
*t* ≥ 2 on a **real** tape.

## Method lineage

- **Predictive regression with Newey-West HAC *t*.** OLS of the forward ETH return on floor
  momentum; the HAC variance (Bartlett weights, rule-of-thumb lag `floor(4*(n/100)^(2/9))`) corrects
  the *t*-stat for the autocorrelation induced by overlapping forward windows. The bar for a signal
  is a predictive HAC *t* ≥ 2 the right way. Cf. **Newey & West (1987)**, "A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix,"
  *Econometrica* 55(3).
- **Lead-lag cross-correlation.** `corr(floor_t, eth_{t+k})` across k — a peak at negative k means
  floors lag ETH (follow); a peak at positive k would mean floors lead.
- **Block / circular-shift placebo.** Rotating the signal against the target preserves each series'
  own autocorrelation while destroying any true lead; the placebo *p* is the tail probability of the
  observed HAC *t*. Cf. permutation testing (**Fisher 1935**; **Good 2005**).
- **Cost model.** One-way cost × turnover on the long/flat ETH overlay; the short variant pays a
  pro-rated annual borrow — the house convention (costs one-way × NAV, shorts pay borrow).
- **Seed-robust synthetic control.** The mean HAC *t* over ≥ 20 seeds for a planted `lead_alpha`, so
  no single lucky seed can manufacture significance (the desk house rule for synthetic claims).

## Related desk studies

- **Synthetic-tape alt-data folklore** — [273 Lego-Returns](../../273-lego-returns/),
  [275 Whisky-Cask](../../275-whisky-cask/), [276 Sneaker-Resale](../../276-sneaker-resale/): the
  same gross-vs-net, survivorship, and "no clean free tape" traps, capped below REAL on data
  availability.
- **Lead-lag / alt-data-as-signal** studies elsewhere on the bench that test whether a noisy
  auxiliary series forecasts a liquid asset — the same predictive-regression + placebo protocol.

## House methodology

- [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a predictive *t* ≥ 2 on a **real**
  tape, plus a placebo null and seed-robustness), the data-availability caveat on the SIGNAL axis,
  one documented execution lag, gross/net labelled everywhere, and costs one-way × NAV with shorts
  paying borrow.
