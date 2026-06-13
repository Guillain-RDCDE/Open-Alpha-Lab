# References & literature map — Study 100 (Melting-Ice)

## The claim under test

Leveraged ETFs (3x like **TQQQ**, **UPRO**, **SPXL**) are routinely described as *toxic*:
the daily-rebalancing mechanic supposedly bleeds the fund to zero through **"volatility
decay"**, so the folk rule is *"never hold a 3x ETF more than one day."* The strong,
sold-at-full-strength version is that decay is a **law** — that over any long horizon a
constant-leverage fund is *guaranteed* to lose to the naive "3x the index return" you
expected, and ultimately to approach zero.

- Popular framing, e.g. Investopedia, *"Why Leveraged ETFs Are Not a Long-Term Bet"*:
  <https://www.investopedia.com/articles/exchangetradedfunds/07/leveraged-etf.asp>
- The "decay to zero" warning is a fixture of broker disclosures and finance media every
  time a 3x ETF is mentioned.

## Why the steelman is almost coherent — the real mechanism

The kernel of truth is exact and not folklore: a fund that **resets to k-times exposure
every day** does not deliver k-times the *period* return. The two foundational papers:

- **Cheng, M. & Madhavan, A. (2009).** *The Dynamics of Leveraged and Inverse
  Exchange-Traded Funds.* Journal of Investment Management. Derives the constant-leverage
  rebalancing identity and shows the realised return depends on the **path**, not just the
  endpoints — the source of the variance-drag term.
- **Avellaneda, M. & Zhang, J. (2010).** *Path-Dependence of Leveraged ETF Returns.*
  SIAM Journal on Financial Mathematics. Proves the closed-form
  `L_T / S_T^k ≈ exp(−(k(k−1)/2) · realized-variance)` — the levered NAV equals
  k-power of the underlying times a discount that grows with **realised variance**. This
  is the "decay" term, made precise.

To leading order, with mean daily log-return `g` and daily variance `s2`, the levered
log-CAGR is `k·g − (k(k−1)/2)·s2`: **3x the drift, minus 3x the variance** (for k=3). So
decay is a *race* between drift (which compounding amplifies in a trend) and the variance
drag — not a one-way street.

## Why the strong claim ("always decays to zero") is false *as stated*

- **Compounding cuts both ways.** In a strong, low-vol **uptrend**, daily resetting
  compounds gains on gains: the realised 3x NAV ends *above* naive-3x, not below. The
  2010s NASDAQ was exactly this regime — and TQQQ massively beat the "3x the period
  return" line over 2010–2021 (this study quantifies it).
- **The break-even is computable.** Drag overwhelms drift only when realised variance
  exceeds `s2* = g / ((k−1)/2)` (for k=3, `s2* = g`). Above that volatility, decay wins
  (2018 chop, 2022 selloff); below it, compounding wins. "Always" is the error.
- **The real risk is drawdown, not a decay law.** A 3x sleeve's honest danger is a
  catastrophic peak-to-trough loss (−77% to −82% on the real funds), which can take a
  decade to recover, plus ~5%/yr all-in cost (0.95% expense + financing on 2x notional) —
  *not* a guarantee of bleeding to zero in a trending market.

## Method lineage

- **Constant-leverage simulator** — the exact daily identity
  `L_t = L_{t−1}·(1 + k·r_t − fee)` (Cheng-Madhavan), validated against the *real* TQQQ /
  UPRO tapes (daily-return correlation > 0.998, RMS tracking error < 20 bps, final NAV
  within ~2%).
- **Drag/drift decomposition** — the second-order log-return expansion
  `E[log(1+k·r)] ≈ k·g − (k(k−1)/2)·s2` (Avellaneda-Zhang), with the break-even variance.
- **Newey–West HAC standard errors** for the mean of an autocorrelated return series:
  Newey & West (1987), Econometrica.

## Data sources used

- **TQQQ** (3x) & **QQQ**; **UPRO** (3x) & **SPY** — daily, **total-return adjusted**
  (dividends folded in) via `quantlab.data` (Yahoo Finance), cached to parquet under
  `_cache/`. The levered fund and its underlying are compared on the same footing. TQQQ
  history begins **2010-02-11**, UPRO **2009-06-25** (each fund's inception); we state
  these windows honestly rather than back-casting.

## Related desk studies

- [Study 68 — All-Weather](../../68-all-weather/) — leverage and risk-parity mechanics on
  a real tape.
- [Study 91 — Death-Cross](../../91-death-cross/) — another "the textbook warning is real
  but the strong claim is false" teardown, same two-sided synthetic-control pattern.
