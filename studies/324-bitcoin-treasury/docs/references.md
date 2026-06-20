# References & literature map — Study 324 (Bitcoin-Treasury)

## The claim under test

- **"MSTR is the best way to get leveraged Bitcoin exposure — and the premium pays."**
  Michael Saylor (MicroStrategy / Strategy, Inc.) has framed the company since
  **2020-08-11** as a Bitcoin acquisition vehicle ("a Bitcoin development company"), with
  shareholder letters and conference talks arguing the equity offers *intelligent
  leverage* to Bitcoin plus an operating-company premium. Retail and sell-side notes
  routinely call MSTR "leveraged Bitcoin," and a cottage industry of mNAV ("market-cap to
  net-asset-value") trackers argues the premium itself is a source of value. The testable
  hypotheses: (H₁) MSTR carries genuine alpha beyond its Bitcoin beta; (H₂) holding MSTR
  beats holding (levered) Bitcoin; (H₃) the NAV premium is harvestable by timing it.

## The mechanics — leverage, beta, and convexity

- **The CAPM/market-model decomposition.** Sharpe (1964); Jensen (1968), *The Performance
  of Mutual Funds in the Period 1945–1964* (Journal of Finance) — Jensen's alpha is exactly
  the intercept of an asset's excess return regressed on its factor; a beta near the
  leverage and an intercept indistinguishable from zero is "you were paid for the exposure,
  not the manager." We run MSTR-on-BTC and read the intercept's HAC *t*.
- **Levered-ETF / constant-leverage decay.** Cheng & Madhavan (2009), *The Dynamics of
  Leveraged and Inverse ETFs*; Avellaneda & Zhang (2010), *Path-Dependence of Leveraged ETF
  Returns* (SIAM J. Financial Math.) — a constant-leverage position on a volatile underlier
  suffers volatility drag, so a >1 beta on Bitcoin is *not* a free lunch even before
  financing. Our levered-BTC replica makes the fair comparison explicit.
- **Financing & the cost of leverage.** Frazzini & Pedersen (2014), *Betting Against Beta*
  (Journal of Financial Economics) — leverage is not costless; the borrow must be charged
  once, which our replica does (and which a "MSTR vs BTC" eyeball never does).

## Closed-end-fund premiums — the NAV-premium analogue

- **The closed-end fund puzzle.** Lee, Shleifer & Thaler (1991), *Investor Sentiment and the
  Closed-End Fund Puzzle* (Journal of Finance) — funds trade at large, persistent, *time-
  varying* premia/discounts to NAV driven by sentiment, not fundamentals; the premium is a
  risk factor, not a reliable return. MSTR's mNAV premium is the same phenomenon on a
  Bitcoin-holding shell, and the same caution applies: a wide, mean-reverting premium is
  variance you carry, not a coupon you collect.
- **GBTC as the direct cousin.** The Grayscale Bitcoin Trust traded at a +100% premium in
  2017 and a −50% discount in 2022 — the canonical demonstration that a "wrapper on
  Bitcoin" premium can swing violently and is not a dependable edge.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat & coefficient covariance.** Newey & West (1987), *A Simple,
  Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance
  Matrix* (Econometrica) — used both for the mean-difference *t* and the regression
  intercept's sandwich SE in [`strategy.hac_tstat_resid`](../bitcoin_treasury/strategy.py).
- **Circular block bootstrap.** Politis & Romano (1992) / Künsch (1989) — block resampling
  preserves the autocorrelation that i.i.d. resampling destroys; the MSTR−replica edge CI
  uses it.
- **Excess-of-cash Sharpe in a levered race.** A levered leg implicitly earns the risk-free
  rate on its own capital; comparing raw Sharpe to excess Sharpe manufactures a verdict, so
  the race is excess-of-cash on both sides (METHODOLOGY → *House rules*).

## Data sources used here

- **Yahoo! Finance daily closes** (via `yfinance`), auto-adjusted (total return): **MSTR**
  (from 2014) and **BTC-USD** (from 2014-09). Headline numbers are the **treasury era**
  (≥ 2020-08-11), inner-joined on common days, pinned with an as-of date and content
  fingerprint (see [`docs/results.md`](results.md)). The offline reproducible core and
  test-suite run on the deterministic [`data.synthetic_pair`](../bitcoin_treasury/data.py)
  generator — BTC random walk + an MSTR built *as* levered beta with a tunable planted
  alpha and a return-free NAV-premium wobble — never the network.

## Related desk studies

- **[Study 70 — Digital-Gold](../../70-digital-gold/)**: is Bitcoin a haven or a high-octane
  risk asset? The risk-asset finding is the foundation MSTR levers up.
- **[Study 210 — Crypto-Trend](../../210-crypto-trend/)** & **[Study 209 — ETH-BTC-Ratio](../../209-eth-btc-ratio/)**:
  the two crypto signals on this bench that *did* stamp Real/Fragile — useful contrast for a
  "leverage is not alpha" null.
- **[Study 221 — Mayer-Multiple](../../221-mayer-multiple/)** & **[Study 293 — MVRV-Ratio](../../293-mvrv-ratio/)**:
  Bitcoin "valuation"/timing ratios that came up Mirage — the same fate as timing MSTR's NAV
  premium here.
