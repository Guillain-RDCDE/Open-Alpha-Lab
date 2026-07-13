# References & literature map — Study 752 (TGA-Drawdown)

## The claim under test

- **The Treasury General Account as a "hidden liquidity lever."** The macro-liquidity
  thesis holds that the TGA — the U.S. Treasury's operating checking account at the
  Federal Reserve — is a swing factor for bank reserves and, through them, for risk
  assets. When the Treasury *draws down* the TGA (spends faster than it borrows), the
  cash lands in private bank accounts and shows up as an increase in **reserve balances**
  at the Fed; when the Treasury *rebuilds* the TGA (borrows and parks the cash), reserves
  fall. Because reserves are a large component of "net liquidity," a TGA drawdown is read
  as a stealth **liquidity injection** that lifts equities over the following weeks — and
  a rebuild as a drain. This is the "**net liquidity**" framework popularised by macro
  strategists and widely amplified on financial social media.
- **The "net liquidity" identity behind it.** The most-quoted operational proxy is
  *Fed balance sheet − TGA − Reverse Repo (RRP)*, tracked as a driver of the S&P 500.
  Michael Howell, *Capital Wars: The Rise of Global Liquidity* (Palgrave Macmillan, 2020)
  and CrossBorder Capital's *Global Liquidity Index* work is the fullest articulation of
  the "liquidity leads markets" view; the specific TGA/RRP-adjusted-reserves version was
  popularised in 2021–2023 markets commentary (e.g. numerous Bloomberg / *Financial
  Times* pieces on how the post-debt-ceiling TGA rebuild would "drain liquidity").
- **The data series.** U.S. Department of the Treasury, *Daily Treasury Statement (DTS)*,
  "Federal Reserve Account" operating cash balance; and Federal Reserve **H.4.1** release
  → FRED series **`WTREGEN`** (*"Liabilities and Capital: Liabilities: Deposits with F.R.
  Banks, Other Than Reserve Balances: U.S. Treasury, General Account: Week Average"*,
  weekly Wednesday level, $ millions). Reserve balances are FRED `WRESBAL`.

## Why the TGA data isn't fetched live here — and what we do

- **FRED CSV endpoint firewalled, and the series is weekly.** The free
  `fred.stlouisfed.org/graph/fredgraph.csv?id=WTREGEN` endpoint times out in this build's
  network sandbox, and the underlying series is **weekly** (~1,000 prints over our
  window). Hardcoding a thousand weekly values by hand would be *false precision*.
  Following the desk convention for a small labelled proxy — **Study 358 (watch-index)**
  and **Study 708 (eurovision-effect)** use exactly this device — we hardcode an
  **approximate monthly end-of-month proxy** of the TGA balance ($B), 2005–2026, and name
  it a **PROXY** on every axis. The landmark moves are faithful (the 2008 Supplementary
  Financing surge, the 2020 COVID balloon toward ~$1.8T, the 2021 / mid-2023 / 2025
  debt-ceiling drawdowns toward near-zero); the exact monthly levels are approximate.
- **Equities.** SPY daily adjusted close via **yfinance** (no key), month-end sampled,
  total-return adjusted — labelled as such.

## Why identification is the crux — reserves, the RRP, and confounding

- **Reserves ≠ TGA one-for-one.** The TGA→reserves pass-through is mediated by the
  Reverse Repo Facility (RRP) and by the size of the Fed balance sheet; in 2022–2023 much
  of the post-debt-ceiling TGA rebuild was absorbed by a *shrinking RRP* rather than by
  falling reserves, muting the predicted liquidity drain. Lorie Logan (FRBNY / FRB Dallas)
  and Fed staff notes on **reserve demand and the RRP** document why the mechanical
  "TGA down ⇒ stocks up" link is contingent, not automatic. This is the mechanism the
  lead/lag test probes.
- **Debt-ceiling episodes confound the sign.** The largest TGA *drawdowns* in the sample
  (2021, mid-2023, 2025) are **debt-ceiling** episodes — the Treasury runs its cash to
  near-zero because it legally *cannot borrow*, a macro-stress regime, not a stimulative
  one. So the biggest "injections" cluster in exactly the months a liquidity bull would
  least expect them, scrambling any clean conditional mean.
- **The market as its own leading indicator.** Equity prices lead the real economy and
  co-move with the same fiscal-and-monetary backdrop that moves the TGA, so a
  contemporaneous association need not be a *lead*. We run an explicit **lead/lag
  cross-correlation** to locate where the injection actually sits relative to SPY returns.
- **Predictive regressions and small-sample caution.** Welch & Goyal (2008), *A
  Comprehensive Look at the Empirical Performance of Equity Premium Prediction* (Review of
  Financial Studies) — most macro predictors that look significant in-sample fail out of
  sample; the bar for a tradable macro-liquidity signal is high.

## Why the inference is HAC + placebo-based

- **Newey-West (HAC) standard errors.** Newey & West (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*
  (Econometrica) — the headline predictive regression of overlapping forward returns on
  the injection uses HAC errors with lags set to the return horizon.
- **Welch two-sample t.** Welch (1947), *The generalization of "Student's" problem when
  several different population variances are involved* (Biometrika) — unequal-variance
  test of the DRAWDOWN-set forward mean against the unconditional mean.
- **Randomization / placebo null.** Because regime months are autocorrelated and the
  effective sample is modest, we resample random same-size month sets and ask how often
  chance is as bullish as the DRAWDOWN set (Fisher's randomization logic; Efron &
  Tibshirani, *An Introduction to the Bootstrap*, 1993).
- **One coincident balloon dominates.** The 2020 COVID TGA surge-then-drawdown is one
  enormous event; we report results with and without 2020–mid-2021 so the verdict doesn't
  ride on it.

## Method lineage (this study's engine)

- **Signal + inference.** [`strategy.injection`](../tga_drawdown/strategy.py),
  [`strategy.summarize`](../tga_drawdown/strategy.py) (Welch *t* + placebo *p*),
  [`strategy.hac_regression`](../tga_drawdown/strategy.py) (Newey-West predictive slope),
  [`strategy.lead_lag`](../tga_drawdown/strategy.py) (the liquidity-lever test),
  [`strategy.timing_overlay`](../tga_drawdown/strategy.py) (hold-when-drawing-down,
  one-month lag, one-way costs).
- **Deterministic synthetic control.**
  [`data.synthetic_tga`](../tga_drawdown/data.py) plants a known drawdown→returns link in
  the capturable (1-month-lagged) window; `edge = 0` must not manufacture significance, a
  large `edge` must light up the test.

## Data sources used here

- **FRED `WTREGEN`** (hardcoded monthly PROXY, $B) + **yfinance SPY** daily adjusted
  close, 2005-01 → 2026-06, cached under `_cache/spy_prices.csv`. All headline numbers are
  pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 385 — Jobless-Claims-Momentum](../385-jobless-claims-momentum/)**: the same
  hardcoded-macro-snapshot + SPY method applied to a famous "leading" labour series — a
  sibling test of whether a celebrated macro gauge actually leads and pays.
- **[Study 358 — watch-index](../358-watch-index/)** and
  **[Study 708 — eurovision-effect](../708-eurovision-effect/)**: the labelled-proxy
  device for a series that can't be fetched live, done honestly and named as a proxy.
