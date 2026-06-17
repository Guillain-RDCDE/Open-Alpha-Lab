# References & literature map -- Study 295 (Stablecoin-Supply)

## The claim

> *Does stablecoin supply growth fuel the next leg of Bitcoin?*

The "dry powder" thesis, popular in crypto commentary and on-chain analytics
desks: stablecoins (USDT, USDC, DAI, ...) are cash parked on-chain.  When the
*aggregate stablecoin supply grows*, new fiat capital has entered the crypto
ecosystem and is waiting to be deployed into BTC/ETH -- so supply growth should
*lead* the next leg up in price.  The mirror claim is that a contracting supply
(net redemptions) drains buying power and precedes drawdowns.

## On-chain / practitioner sources

- **DefiLlama -- Stablecoins dashboard.** https://defillama.com/stablecoins
  The canonical aggregate of circulating stablecoin market cap by chain and
  issuer; the curated series in `data.py` tracks this aggregate (USDT + USDC +
  DAI + BUSD + the long tail), rounded to a small monthly table.
- **Glassnode / CryptoQuant -- Stablecoin Supply Ratio (SSR).** The SSR
  (BTC market cap / stablecoin supply) is the most widely cited practitioner
  framing of the dry-powder idea: a low SSR is read as "lots of stablecoin
  buying power relative to BTC".  Our supply-growth signal is a first-difference
  cousin of the SSR thesis.
- **Circle (USDC) and Tether (USDT) attestation reports.** Issuer-level reserve
  and circulation disclosures used to cross-check the aggregate.
- **The Block Research -- stablecoin supply charts.** Independent reconstruction
  of issuer-level supply used as a second source for the curated table.

## Why reflexivity is the central hazard

- **Soros, G. (1987). *The Alchemy of Finance.*** The reflexivity framework:
  in markets, the indicator and the price are *jointly* driven by sentiment.
  Stablecoin minting surges precisely when risk appetite is high and prices are
  already rising -- so supply growth is far more a *coincident* than a *leading*
  variable.  A large contemporaneous correlation that **collapses under a
  one-month execution lag** is the textbook signature of a non-tradable,
  reflexive series; our study finds exactly this (contemporaneous HAC t = +2.15
  vs lagged t = +1.59).

- **Lo, A. W. (2004). *The Adaptive Markets Hypothesis.* J. Portfolio Mgmt.**
  Anomalies built on a single hot regime decay as the regime ends.  Our
  sub-period split shows the lagged predictive slope falling from t = +3.05
  (2018-2020) to t = +0.01 (2023-2026) -- the apparent edge is a 2018-2020
  artefact, not a stable structural relationship.

## On stablecoins, flows, and crypto prices (academic)

- **Griffin, J. M. & Shams, A. (2020). *Is Bitcoin Really Untethered?*
  Journal of Finance, 75(4), 1913-1964.** Documents that Tether issuance was
  associated with BTC price movements in 2017 -- but the mechanism is
  *manipulation / supply-side push*, and the result is about contemporaneous
  co-movement, not a clean tradable forward signal.  A cautionary tale about
  reading causation into stablecoin-supply / price correlation.
- **Lyons, R. K. & Viswanath-Natraj, S. (2023). *What Keeps Stablecoins
  Stable?* Journal of International Money and Finance.** Stablecoin supply
  responds endogenously to arbitrage and demand shocks -- reinforcing that
  supply is an endogenous, demand-driven quantity rather than an exogenous
  forcing variable for price.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance
  Matrix* (Econometrica).  Used on both the timing return and the predictive
  regression slope.
- **Block-bootstrap Sharpe CI.** Politis & Romano (1994), *The Stationary
  Bootstrap* (JASA).  Used on the timed-minus-buy-and-hold excess Sharpe.

## Related desk studies

- **[Study 209 -- ETH-BTC-Ratio](../../209-eth-btc-ratio/)** and
  **[Study 210 -- Crypto-Trend](../../210-crypto-trend/)**: other crypto
  cross-asset / timing studies in the Crypto family.
- **[Study 158 -- Super-Bowl](../../158-super-bowl/)**: the folklore baseline
  -- a binary "signal" that gets credit for a base rate (here: BTC's enormous
  unconditional drift) rather than genuine forward information.
- **[Study 223 -- Same-Month-Seasonality](../../223-same-month-seasonality/)**:
  the structural template this study mirrors (synthetic positive control +
  cached real tape, HAC t-stats, sub-period decay).
