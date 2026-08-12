# References & literature map — Study 891 (Insurance Float Engine)

## The claim under test

- **Float as a compounding engine.** A property-and-casualty insurer collects premiums today
  and pays claims later; the money held in between — the **float** — can be invested for the
  insurer's own account. Warren Buffett's Berkshire Hathaway chairman's letters (berkshirehathaway.com,
  esp. 1997, 2002, 2009 and the "float" sections he revisits almost every year) frame this as
  near-zero-cost (sometimes *negative*-cost) leverage: if underwriting breaks even or better, the
  investor keeps the full return on other people's money. The popular extrapolation — "so a broad
  basket of insurers should be a quiet, market-beating, structurally-advantaged compounder" — is
  the folklore this study tests.
- **Why it might *not* generalise.** Berkshire's edge came from *disciplined underwriting* plus a
  world-class investment operation on top of the float, not from being an insurer per se. A
  cap-/equal-weighted basket of listed insurers holds the average underwriter, whose float is only
  valuable if the underwriting cost of that float is low — and the basket also carries the sector's
  interest-rate and credit sensitivity, i.e. **financial-sector beta**. The academic case that
  Buffett's return is leverage + quality/low-vol exposure, not magic, is **Frazzini, Kabiller &
  Pedersen (2018), "Buffett's Alpha," Financial Analysts Journal** — directly relevant to whether a
  plain insurer basket inherits any of it.

## What we measured, and the rails

- **Excess-vs-excess Sharpe race.** Both the insurer basket and SPY are measured *excess-of-cash*
  (minus BIL) before annualising the Sharpe, so a rising short rate (2022-26) cannot flatter a thin
  edge and the comparison is like-for-like. The "advantage" is the folklore's predicted-positive
  number.
- **CAPM & two-factor decomposition.** The CAPM alpha asks whether any out-performance survives
  subtracting market beta; the two-factor regression on [market, **KBE − SPY** bank spread] is the
  decisive test — a *float* premium would survive the financial-sector control, whereas plain sector
  beta is absorbed by it. Method: **Newey & West (1987)**, *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica — HAC *t* on
  the mean difference and on every regression coefficient (6 Bartlett lags).
- **Bootstrap Sharpe CI.** Circular block bootstrap — **Politis & Romano (1994)**, *The Stationary
  Bootstrap*, JASA 89 — via `quantlab.stats.sharpe_ci_bootstrap`, so the interval respects the serial
  dependence in monthly returns. **Lo (2002)**, *The Statistics of Sharpe Ratios*, FAJ, is the
  standard reference for why a Sharpe needs a standard error at all.
- **Era cut / drawdowns / calendar-year table.** A structural premium should show up in calm eras,
  not only in the crash where insurers happened to fall less; the drawdown and per-year tables make
  the risk profile concrete.

## Data sources

- **ETFs (yfinance, total-return `auto_adjust=True` closes, no key):**
  - **KIE** — SSGA SPDR S&P Insurance ETF, **equal-weight** the S&P Insurance Select Industry Index
    (ssga.com); inception 2005-11-08. The headline basket.
  - **IAK** — iShares U.S. Insurance ETF, tracks the Dow Jones U.S. Select Insurance Index
    (ishares.com); inception 2006-05-01. The second wrapper.
  - **SPY** — SPDR S&P 500 ETF Trust, the market benchmark.
  - **KBE** — SSGA SPDR S&P Bank ETF, equal-weight the S&P Banks Select Industry Index; the
    financial-sector control (banks lever a spread, not float).
  - **BIL** — SPDR Bloomberg 1-3 Month T-Bill ETF (inception 2007-05-25); the cash leg. Its
    ~0.1354 % expense ratio makes it a tradable, mildly-conservative cash proxy.
- **Sample.** The common window where all five exist is **2007-06 → 2026-06** (BIL binds the start),
  229 complete months, cached once under `_cache/insfloat_prices.parquet`.

## Related desk studies (dedup)

- [628-buffetts-alpha](../../628-buffetts-alpha/) — decomposes *Berkshire itself* into leverage +
  quality/low-vol factors (Frazzini-Kabiller-Pedersen). **This study is different**: we do not touch
  Berkshire; we ask whether a plain *listed-insurer basket* (KIE/IAK) inherits any market-beating
  edge from the float story, and find it is financial-sector beta.
- [51-blue-chip-quality](../../51-blue-chip-quality/) (gross-profitability / quality) — a
  cross-sectional *quality* factor across all sectors. Here the "quality" angle is sector-specific
  (float-funded insurers vs spread-levered banks) and tested as a *sector-basket vs market* race, not
  a cross-sectional sort.
- [246-defensive-sectors](../../246-defensive-sectors/) — the low-vol / defensive-sector rotation
  claim. Insurers are *not* defensive here (β = 1.1, drawdown −70 %); the overlap is only that both
  ask "does a sector tilt beat the market risk-adjusted?" — this one answers no, via the float lens.
- [340-bank-loans](../../340-bank-loans/) — floating-rate *bank-loan credit* as an asset class; a
  different financial-sector exposure (senior secured loans, not equity), and a yield/credit study
  rather than an equity Sharpe race. KBE (bank *equity*) is our control leg, not the subject.

## Method citations

- **Frazzini, Kabiller & Pedersen (2018)** — *Buffett's Alpha*, Financial Analysts Journal 74(4):
  leverage + quality/low-beta, not alchemy.
- **Newey & West (1987)** — HAC covariance for the *t*-statistics.
- **Politis & Romano (1994)** — the stationary/circular block bootstrap behind the Sharpe CI.
- **Lo (2002)** — *The Statistics of Sharpe Ratios*, Financial Analysts Journal 58(4).
- **Berkshire Hathaway shareholder letters** — the primary source for the float-as-leverage claim.
