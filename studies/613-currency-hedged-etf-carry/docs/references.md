# References & literature map — Study 613 (Currency-Hedged ETF Carry)

## The claim under test

- **The share-class carry story.** A currency-hedged equity ETF holds the foreign basket and
  **sells the foreign currency forward** (typically one-month forwards, rolled). Covered
  interest parity prices that forward at the short-rate differential, so the hedged class earns
  the local equity return **plus (r_US − r_foreign)** — "free carry hidden in a share class".
  The claim circulated widely in the 2013–2015 hedged-ETF boom (WisdomTree's DXJ/HEDJ marketing,
  the "hedged share class" debate) and again in 2022–2024 when the Fed-vs-BOJ gap hit ~5 %/yr.
- **Covered interest parity.** Keynes (1923, *A Tract on Monetary Reform*) for the original
  forward-points arithmetic; any textbook: the 1-month forward premium on USDJPY ≈ the 1-month
  USD−JPY rate differential. The hedge P&L therefore *is* the differential, mechanically —
  which is why this is a mechanics study, not an anomaly hunt.
- **CIP deviations / cross-currency basis.** Du, Tepper & Verdelhan (2018, *Deviations from
  Covered Interest Rate Parity*, JF): post-2008 the JPY basis is persistently **negative**,
  meaning a USD investor selling JPY forward earns the differential **plus** the basis. That is
  the right sign and rough size (~20–60 bp) for the ~0.5 %/yr excess of our measured HEWJ/EWJ
  carry (+2.39 %/yr) over the coarse policy differential (+1.89 %/yr).

## Why HEWJ/EWJ is the decisive pair

- **iShares Currency Hedged MSCI Japan (HEWJ)** holds **EWJ itself plus one-month JPY forwards**
  (iShares fund page & prospectus, ishares.com) — same basket, same manager, so the return
  differential isolates the hedge overlay almost purely. DXJ (WisdomTree Japan Hedged Equity,
  wisdomtree.com) is dividend-weighted and exporter-tilted, so its differential vs EWJ mixes
  hedge carry with basket alpha even though its hedge itself is complete (β = 1.05 vs HEWJ's
  0.97 on the hedged-mandate windows). HEDJ (WisdomTree Europe Hedged Equity) vs VGK (Vanguard
  FTSE Europe) has the same basket-mismatch problem on the EUR leg — plus a 0.58 % vs 0.09 %
  ER gap.
- **DXJ history.** DXJ launched June 2006 as the *unhedged* WisdomTree Japan Total Dividend
  Fund and only adopted the currency-hedged mandate on **April 1, 2010** (fund literature,
  wisdomtree.com) — our DXJ window starts 2010-04 for that reason. The tape confirms the date:
  DXJ−EWJ's regression β on −fx is **−0.15** over 2006-07..2010-03 (no hedge) and **+0.94**
  after.
- **HEDJ history.** HEDJ traded before 2012 as a different (unhedged international dividend)
  fund; it switched to the hedged Europe mandate in **January 2012** — our HEDJ window starts
  2012-02 for that reason (fund literature, wisdomtree.com).

## Rates & data sources

- **US short rate** — `^IRX` (13-week T-bill discount, %) via yfinance.
- **Bank of Japan** — uncollateralized overnight call-rate target chronology
  (https://www.boj.or.jp/en/mopo/mpmdeci/index.htm): ZIRP → 0.25 (Jul-2006) → 0.50 (Feb-2007) →
  cuts to 0.10 (Dec-2008) → 0–0.1 (Oct-2010) → **−0.10 (Jan-2016)** → 0–0.1 (Mar-2024) → 0.25
  (Jul-2024) → 0.50 (Jan-2025). Hardcoded as a step table in
  [`data.py`](../currency_hedged_etf_carry/data.py).
- **ECB** — deposit facility rate, "Key ECB interest rates"
  (https://www.ecb.europa.eu/stats/policy_and_exchange_rates/key_ecb_interest_rates/):
  0.25 (Dec-2011) … −0.50 (Sep-2019) … hikes Jul-2022→4.00 (Sep-2023) → cuts to 2.00 (Jun-2025).
  Hardcoded step table, same file.
- **Prices/FX** — yfinance total-return closes for DXJ, EWJ, HEWJ, HEDJ, VGK; spot `JPY=X`
  (yen per dollar) and `EURUSD=X`; cached once under `_cache/chc_prices.csv`.
- **Expense ratios** (fund pages, 2026): DXJ 0.48 %, EWJ 0.50 %, HEWJ 0.50 %, HEDJ 0.58 %,
  VGK 0.09 %.

## Method citations

- **Newey & West (1987)**, *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix*, Econometrica — the HAC t's on means and on the
  hedge regression (6 Bartlett lags; 3/12 sensitivity reported).
- **Fama (1984)**, *Forward and Spot Exchange Rates*, JME — the forward-premium regression
  lineage behind decomposing hedged-minus-unhedged into carry and FX legs.
- **Sialm & Zhu (2022)**, *Currency Management by International Fixed Income Mutual Funds* (JF,
  earlier NBER wp) and **Israel, Maloney & Villalon** (AQR practitioner literature on hedged
  share classes) — hedged-vs-unhedged share-class mechanics and the case that the hedge is a
  rate-differential transfer, not a return enhancer per se.

## Related desk studies (dedup)

- [364-fx-carry-trade](../364-fx-carry-trade/) — trades the carry **directly in currencies**
  (long high-yielders / short funders, a risk premium with crash skew). **This study is
  different**: no currency positions are chosen; we test whether an equity ETF's *hedging
  wrapper* mechanically passes the rate differential into a share class you can buy at ETF
  spreads. There the question is "is carry a rewarded risk?"; here it is "does the wrapper
  really deliver the arithmetic?"
- [611-mreit-carry](../611-mreit-carry/) and [612-em-debt-carry](../612-em-debt-carry/) — the
  neighboring packaged-carry studies in this lot (leverage-funded MBS carry; EM sovereign yield
  carry). This one is the cleanest of the three mechanically: the transfer is a forward
  contract, not a levered balance sheet.
- [147-fx-momentum](../147-fx-momentum/) — the FX time-series cousin; no overlap in claim.
