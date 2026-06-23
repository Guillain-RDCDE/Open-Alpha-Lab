# References & literature map — Study 378 (ETF NAV Premium/Discount)

## The claim under test

- **The folk version.** A staple of ETF "value" lore: an ETF trading **below its NAV** is
  "on sale" — the share price is cheaper than the basket it owns, so the discount is a free
  gap that *must* close, and patient buyers harvest it. The bond-ETF dislocations of
  **March 2020** (HYG, LQD, EMB, MUB all printing large intraday discounts to NAV) are
  cited both as proof the discounts are real and as the textbook "buy-the-discount" setup.
- **Where it comes from.** ETF prem/disc is a genuine, *reported* quantity: every fund
  publishes daily NAV and the closing-price premium/discount, and regulators require it
  (e.g. SEC Rule 6c-11, the "ETF Rule", 2019, and the fund-by-fund premium/discount
  disclosures). The lore bolts a *trading* claim onto that reported gap.

## Why an ETF can trade away from NAV — the structural mechanism

- **The creation/redemption arbitrage.** Authorized Participants (APs) keep ETF price ≈ NAV
  by creating shares when the ETF is rich and redeeming when cheap (Gastineau, *The
  Exchange-Traded Funds Manual*, 2010; Ben-David, Franzoni & Moussawi, "Exchange-Traded
  Funds", *Annual Review of Financial Economics*, 2017). The arbitrage is **not free**: it
  costs the AP the basket's bid-ask, balance-sheet, and inventory risk, so a band of
  prem/disc persists — wider for **illiquid underlyings** (high yield, EM, munis) than for
  large-cap equity.
- **Stale-NAV and price-discovery in bond ETFs.** For bond ETFs the *NAV itself* is partly
  stale — many underlying bonds don't trade daily and are matrix-priced — so during stress
  the **ETF price leads NAV** and the "discount" is in large part NAV catching down to a
  price the ETF already discovered (Madhavan & Sobczyk, "Price Dynamics and Liquidity of
  Exchange-Traded Funds", *Journal of Investment Management*, 2016; Pan & Zeng, "ETF
  Arbitrage Under Liquidity Mismatch", 2019). Buying the "discount" can mean buying *ahead*
  of a real markdown, not at a bargain.
- **March 2020.** The bond-ETF discounts of March 2020 are the canonical case: HYG/LQD/EMB
  traded at multi-percent discounts to (stale) NAV until the Fed's credit facilities
  (PMCCF/SMCCF) backstopped the underlying. Studies of the episode (e.g. Aramonte &
  Avalos, BIS Bulletin, 2020; Falato, Goldstein & Hortaçsu, 2021 on bond-fund fragility)
  document that the arbitrage **broke** exactly when the discount was largest — the worst
  time to be relying on "the gap closes."

## Why the gap rarely pays — the statistics & microstructure

- **Mean reversion vs. harvestable return.** A stationary, mean-reverting basis (our
  prem/disc proxy) guarantees the *gap* shrinks on average, but the **investor's** return is
  the ETF's move net of the basket — most of a "discount" is the basket itself dropping, so
  the hedged harvest is far smaller than the headline gap. We isolate it with a hedge-basket
  residual and test the mean with a **Newey-West (HAC) t** (Newey & West, 1987) because
  overlapping holds are autocorrelated, plus a **placebo / randomization null** (Fisher's
  randomization logic; Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993).
- **Costs dominate a few-bp edge.** Half-spreads on HYG/EMB/MUB run ~1-3 bp in calm markets
  and blow out in stress; a hedged round-trip pays the spread on **two** legs. When the gross
  edge is ~5 bp, the round-trip *is* the edge — the classic result that bond-ETF
  prem/disc "arbitrage" is an AP business, not a retail one (Ben-David et al., 2017).
- **Selection on a famous episode.** "Buy the discount" is selected on March 2020, when it
  happened to work *after the Fed*; Harvey, Liu & Zhu (2016), "…and the Cross-Section of
  Expected Returns" (*Review of Financial Studies*) and Bailey & López de Prado (2014),
  *The Deflated Sharpe Ratio*, formalise why a rule discovered on one vivid episode needs a
  far higher bar than an in-sample *t*.

## Method lineage (the desk's shared engine)

- **Hedged harvest + HAC t.** [`strategy.hedged_harvest`](../etf_nav_premium/strategy.py)
  and [`strategy.hac_t`](../etf_nav_premium/strategy.py) — the residual earned back after a
  discount, with a Newey-West SE for overlapping holds.
- **Placebo null.** [`strategy.placebo_pvalue`](../etf_nav_premium/strategy.py) — draw the
  same number of random entry dates many times; the honest small-edge test.
- **NAV proxy + deterministic synthetic control.**
  [`data.nav_basis`](../etf_nav_premium/data.py) builds the transparent prem/disc proxy;
  [`data.synthetic_basis`](../etf_nav_premium/data.py) plants a known per-day earn-back the
  inference must recover (and, at zero edge, must NOT manufacture).
- **Costs.** [`strategy.net_of_costs`](../etf_nav_premium/strategy.py) — one-way half-spread
  charged on the ETF and the hedge leg(s).

## Data sources used here

- **yfinance** daily adjusted closes for HYG, EMB, MUB + hedge legs (JNK, EMLC, LQD, TFI,
  IEF), 2012-01-03 → 2026-06-18, cached under `_cache/etf_prices.csv`. All headline numbers
  are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py). Official iNAV/NAV is **not** on the free
  endpoint — see the proxy caveat in `results.md`.

## Related desk studies

- Buy-the-dip and mean-reversion folklore (price "snaps back" after a drop) is the
  equity-side cousin of "buy the discount"; the same cost-vs-edge arithmetic decides both.
- Stress-conditional "free lunch" claims (a gap that only appears in a crisis and is
  hardest to trade then) recur across the bench's structural/microstructure family — the
  pattern is always: real gap, real reversion, no harvestable net.
