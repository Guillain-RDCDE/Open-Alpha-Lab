# References & literature map — Study 901 (Profitable Small-Caps)

## The claim under test

- **Asness, Frazzini, Israel, Moskowitz & Pedersen (2018)**, *"Size Matters, If You Control
  Your Junk"*, Journal of Financial Economics 129(3). The headline finding: the raw size
  premium (small minus big) is weak, unstable and concentrated in microcaps — **because
  small caps are on average "junkier"** (lower profitability, lower quality, higher beta).
  Once you control for a quality/junk factor (QMJ), the size premium becomes **large, stable,
  monotonic across size deciles, robust across 30 countries and pervasive across time**. In
  their words, size "works" only after you clean it of junk. This study asks the *tradable*
  version: does a live **profitable / high-quality small-cap ETF** actually beat a plain
  small-cap index (and hold up against large caps) on excess-of-cash Sharpe, net of costs?
- **Banz (1981)**, *The Relationship Between Return and Market Value of Common Stocks*, JFE —
  the original size effect. **Fama & French (1993, 2015)** — SMB as a factor and the
  five-factor model that adds profitability (RMW) and investment (CMA), the academic parents
  of "quality small caps".
- **Novy-Marx (2013)**, *The Other Side of Value: The Gross Profitability Premium*, JFE — the
  profitability leg that CALF's free-cash-flow screen and XSHQ's quality composite proxy.
- **The disappointment literature.** Since Banz, the naive small-cap premium has largely
  vanished out-of-sample (Horowitz, Loughran & Savin 2000; McLean & Pontiff 2016 on
  post-publication decay). AFMP's junk-control is the leading rescue; whether an ETF can
  *harvest* the rescued premium is exactly what a live-fund Sharpe race can falsify.

## The ETFs

- **CALF** — Pacer US Small Cap Cash Cows 100 (paceretfs.com): the 100 highest trailing
  free-cash-flow-yield names in the S&P SmallCap 600, FCF-weighted, reconstituted quarterly.
  A pure "profitable small-cap" expression. Inception **2017-06-15**; ER **0.59 %**.
- **XSHQ** — Invesco S&P SmallCap Quality (invesco.com): S&P SmallCap 600 names ranked on a
  quality composite (return on equity, accruals ratio, financial leverage), quality-weighted.
  Inception **2017-04-06**; ER **0.29 %**.
- **IWM** — iShares Russell 2000 (ishares.com): the plain, "junk-and-all" small-cap beta.
  Inception **2000-05-22**; ER **0.19 %**.
- **IJR** — iShares Core S&P Small-Cap 600 (ishares.com): plain small caps, but the **S&P
  SmallCap 600 index already imposes a mild positive-earnings screen at construction** (S&P
  Dow Jones methodology) — a "half-cleaned" baseline. Inception **2000-05-22**; ER **0.06 %**.
- **SPY** — SPDR S&P 500 (ssga.com): the large-cap yardstick. Inception **1993-01**; ER
  **0.09 %**.
- **BIL** — SPDR Bloomberg 1-3 Month T-Bill (ssga.com): the **cash leg**. Every Sharpe here
  is **excess of BIL's total return**. Inception **2007-05**; ER **0.14 %**. `^IRX` (13-week
  T-bill discount) is carried as a fallback proxy.
- **Short-history caveat.** CALF and XSHQ began trading in **2017**, so the profitable-small
  read has ~9 years of live tape — one COVID crash, one 2022 bear, no full pre-GFC cycle.
  Named on the Signal axis; the era cut (pre-/post-2021) checks robustness within that span.

## Data & method

- **Prices** — yfinance **total-return** closes (`auto_adjust=True`) for all six ETFs plus
  `^IRX`, cached once under `_cache/psc_prices.csv`. As-of **2026-06-30** (partial month
  dropped). Fingerprint stamped in [`results.md`](results.md).
- **Excess-of-cash** — every leg is measured minus BIL's daily total return, so a Sharpe race
  is a race of risk-adjusted *premia*, not of cash-rate luck.
- **Newey & West (1987)**, Econometrica — HAC (Bartlett) t on the daily return difference and
  on the size/market regression coefficients (10 daily lags).
- **Politis & Romano (1994)**, *The Stationary Bootstrap*, JASA — the circular-block
  bootstrap behind the Sharpe CI and the **paired** Sharpe-difference CI.
- **Lo (2002)** / **Mertens (2002)** — Sharpe-ratio standard errors (mirrored in
  `quantlab.analytics.sharpe_with_se`, cited for the inference lineage).

## Related desk studies (dedup)

- [513-size-effect](../../513-size-effect/) — tests the **raw** size premium (small minus
  big) directly. **This study is different**: it does not re-litigate whether small beats big;
  it asks whether the *AFMP quality-cleaned* version is harvestable through a live **profitable
  small-cap ETF** on a risk-adjusted, costed basis.
- [657-larry-portfolio](../../657-larry-portfolio/) — small-**value** tilt (the Larry Swedroe
  portfolio). Overlaps in "tilt small caps toward a factor" but the factor there is **value**,
  not **profitability/quality**; here the value tilt is a *control* (stripped in the beta
  decomposition), not the thesis.
- [242-quality-minus-junk](../../242-quality-minus-junk/) — the QMJ factor itself, across the
  whole cap spectrum. This study is the **small-cap slice** of that idea, packaged as a
  buyable ETF and raced against plain small caps and large caps.
- [362-piotroski-f-score](../../362-piotroski-f-score/) — a specific accounting-quality score
  for stock selection. Cousin in spirit (quality screening) but a single-name fundamental
  signal, not a small-cap-ETF allocation race.
