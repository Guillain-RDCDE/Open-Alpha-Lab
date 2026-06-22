# References & literature map — Study 359 (Farmland)

## The claim under test

- **The pitch.** *"Farmland is the best inflation hedge — a smooth, low-volatility, near-zero-
  correlation real asset that quietly compounds; that's why billionaires buy it."* The two folk
  anchors are (1) the **NCREIF Farmland Index**, whose appraisal-based total returns are famously
  steady and lightly correlated with stocks/bonds, and (2) the widely-reported fact that **Bill
  Gates is the largest private owner of US farmland** (~270,000 acres; see *The Land Report* /
  NBC/Forbes coverage, 2021). The testable claims: (H₁) listed farmland is a low-beta diversifier;
  (H₂) farmland's return rises with inflation (a real inflation hedge); (H₃) the smooth, uncorrelated
  NCREIF numbers describe the risk you actually bear.

## What's actually tradable

- **NCREIF Farmland Index** — a quarterly, *appraisal-based* total-return index of institutionally-
  owned US farmland (income + appreciation), maintained by the National Council of Real Estate
  Investment Fiduciaries (`ncreif.org`). The full index is members-only/paywalled; we hardcode the
  widely-republished **annual** total returns (1992–2023) compiled from NCREIF press summaries and
  the farmland reviews published by TIAA/Nuveen and Manulife/Hancock Agricultural Investment Group.
- **Listed farmland REITs (the only retail vehicles).** **LAND** — Gladstone Land Corp. (NASDAQ,
  IPO Jan-2013); **FPI** — Farmland Partners Inc. (NYSE, IPO Apr-2014). Both own row-crop / permanent-
  crop farmland and lease it; both carry leverage and trade thinly. We pull auto-adjusted (dividend-
  reinvested) month-end closes from **yfinance** and use **SPY** as the market benchmark.
- **CPI.** US CPI-U annual inflation (BLS), hardcoded alongside the NCREIF series.

## Why the smooth NCREIF numbers overstate the case — appraisal smoothing

- **Appraisal smoothing / un-smoothing.** Geltner, D. (1991), *Smoothing in Appraisal-Based Returns*,
  Journal of Real Estate Finance and Economics; Fisher, Geltner & Webb (1994), *Value Indices of
  Commercial Real Estate*. Appraisal-based indices are moving averages of true values, which
  mechanically **shrinks measured volatility and beta and manufactures positive autocorrelation**.
  The first-order reversal `r_true[t] = (r_app[t] − ρ·r_app[t−1])/(1−ρ)` recovers the implied true
  series. Our synthetic control reproduces the closed-form variance ratio `a/(2−a)`, lag-1 autocorr
  `1−a`, and the beta-shrink exactly.
- **Stale pricing and serial correlation of real-asset returns.** Getmansky, Lo & Makarov (2004),
  *An Econometric Model of Serial Correlation and Illiquidity in Hedge Fund Returns* (JFE) — the same
  smoothing logic: illiquid, marked-to-model assets report autocorrelated, vol-suppressed returns,
  so reported Sharpe/correlation are flattering artifacts.
- **NCREIF vs listed real estate.** The appraisal-vs-transaction gap is well documented for
  commercial property (NCREIF NPI vs NAREIT); farmland inherits the same wedge — the listed proxies
  (LAND/FPI) are the transaction-priced mirror and are far more volatile and market-correlated.

## Inflation hedging — the empirical content

- **Inflation-hedge regressions.** Fama & Schwert (1977), *Asset Returns and Inflation* (JFE) — an
  inflation hedge is an asset whose nominal return loads positively and significantly on realised/
  expected inflation. We run the annual return on CPI with HAC standard errors.
- **Farmland as an inflation hedge — the academic prior.** Painter (2010s) and the TIAA/Nuveen and
  Hancock farmland reviews report positive long-run real returns and a positive (but noisy) inflation
  correlation for farmland; the effect is real-economy plausible (land + food prices co-move with the
  price level) but statistically weak in short samples. Our short annual sample finds a *positive but
  insignificant* CPI loading — consistent with "real, weak."

## Method lineage (the desk's shared engine)

- **OLS with Newey-West HAC standard errors.** Newey & West (1987). Guards against the
  autocorrelation that appraisal smoothing and overlapping/annual returns inject into the residuals;
  `strategy.ols_hac` returns HAC *t*-stats for every beta.
- **CAPM market-beta / diversification test.** Sharpe (1964). A real low-beta diversifier shows a low,
  stable, significant beta; the listed proxies instead show ~0.7–0.9 beta with ~30% annual vol.
- **Deterministic synthetic control.** A fixed-seed appraisal-smoothing generator
  ([`data.synthetic_smoothing`](../farmland/data.py)) checked against the closed-form variance/
  autocorrelation/beta shrink ([`strategy.expected_var_ratio`](../farmland/strategy.py)) — the offline
  core runs with no network.

## Data sources used here

- **yfinance** auto-adjusted month-end closes: LAND, FPI, SPY (Feb-2014 → Jun-2026), cached under
  `_cache/prices_monthly.csv`.
- **NCREIF Farmland Index** annual total returns + **BLS CPI** + **S&P 500** annual total returns
  (1992–2023), hardcoded public annual figures (cited above) — a transparent, clearly-labelled cited
  series standing in for the paywalled quarterly index, never fabricated to flatter a tradable result.
  All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 144 — Permanent-Portfolio](../../144-permanent-portfolio/)** and **[Study 151 —
  Stocks-for-the-Long-Run](../../151-stocks-for-long-run/)**: real-but-fragile real-asset / long-horizon
  diversification claims, the same "real low-beta diversifier, fragile to hold" shape.
- **[Study 120 — Excess-CAPE-Yield](../../120-excess-cape-yield/)**: another "real signal, mirage once
  you account for the measurement artifact" teardown.
