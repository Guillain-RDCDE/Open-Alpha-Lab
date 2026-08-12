# References & literature map — Study 906 (EM Local Bonds, FX-Hedged — a proxy)

## The claim under test

- **The pitch.** Emerging-market **local-currency** government bonds yield far more than
  their US-dollar counterparts because they pay the *local* short rate (Brazil, Mexico,
  Indonesia, South Africa have run 6–13 % policy rates while US bills paid 0–5 %). The
  standard complaint is that the **currency** eats the carry: an unhedged US investor in
  EMLC earns `local_bond_return + EM_FX_return`, and the EM-FX leg is so volatile (and, over
  2010–2026, so *negative* as the dollar rose structurally) that the fat local rate
  disappears. The natural fix a desk reaches for: **hedge the FX** and keep the local rate.
- **The problem.** There is **no clean, liquid FX-hedged EM-local ETF** on US tape (the
  hedged-EM-local funds that exist are tiny, short-lived, or OTC). So we build a **proxy
  hedge** and label it as one throughout: a long position in **UUP** (Invesco DB US Dollar
  Bullish, long USD futures vs the DXY basket EUR/JPY/GBP/CAD/SEK/CHF). When the broad
  dollar rallies, EM currencies almost always sell off together, so a long-UUP overlay gains
  roughly when EMLC's FX leg loses. This is a **developed-market-basket** proxy for an
  **EM-basket** hedge — high correlation, far from 1 — and that gap is the honest limitation.
- **The specific test here.** Estimate the variance-min hedge ratio `b` of EMLC on UUP
  (excess-of-cash), form the hedged series `(EMLC−BIL) − b·(UUP−BIL)`, and race it
  excess-vs-excess against the USD-EM sibling **EMB** and cash: does stripping the dollar
  leave a local-rate carry that (a) clears a HAC *t* ≥ 2, (b) has a bootstrap Sharpe CI clear
  of zero, (c) holds across eras, and (d) beats the simpler USD-EM ETF? A costed overlay and
  a planted-carry synthetic control complete the teardown.

## What we measure, and the honesty rails

- **Excess-vs-excess only.** Every leg is netted against BIL (the tradable 1-3m T-bill ETF),
  so a Sharpe advantage is a like-for-like race, not a cash-rate illusion — decisive when
  US bills paid ~5 % over 2023–2025.
- **Total-return tape.** `auto_adjust=True` — coupons and distributions reinvested; no
  price-only leakage.
- **In-sample `b` is an upper bound; a walk-forward `b` is the honest cross-check.** The
  headline hedge ratio is fit on the full sample (best-case FX-strip). A 36-month **rolling**
  `b` applied with a one-month lag (`rolling_hedge_series`) is the implementable version with
  **no look-ahead**; it gives the same thin result, and it exposes the ~+0.10 residual EM-FX
  beta the DXY proxy cannot reach.
- **Robust inference.** Newey-West (HAC, Bartlett, 6-lag) *t* on the hedged excess and on the
  hedged−EMB premium — monthly bond returns are serially correlated, so a plain *t* would
  overstate significance. A circular-block bootstrap Sharpe CI and a two-era cut cross-check.
- **Short-history / young-ETF caveat, on the Signal axis.** EMLC starts 2010-08, LEMB
  2011-11, EBND 2011-03 — ~15 years, ~190 months. That is a short sample for a
  macro-conditional carry story dominated by a single dollar super-cycle; the era split says
  so explicitly.
- **The timer is graded separately.** The UUP overlay is re-struck each rebalance to
  `|b|`·NAV notional and pays a one-way spread; the net decides tradability.
- **The synthetic control proves the machinery only.** A planted local-minus-US carry is
  recovered (HAC *t* = +3.25); the null fires on ~1/20 seeds (nominal). It never supports the
  real-tape stamp.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* on the hedged excess and the premium series).
- **Politis, D. & Romano, J. (1994)** — the stationary / circular block bootstrap used for
  the Sharpe confidence intervals (via `quantlab.stats.sharpe_ci_bootstrap`).
- **Lo, A. (2002)** — "The Statistics of Sharpe Ratios" (the delta-method SE behind
  `quantlab.analytics.sharpe_with_se`).
- **Covered interest parity** — the identity that a currency-hedged foreign bond earns the
  local rate minus the forward points ≈ the local-vs-US short-rate differential; the reason a
  clean FX hedge *should* leave the local carry behind.

## Data sources

- **yfinance daily closes** (`auto_adjust=True`, total-return): EMLC, LEMB, EBND, EMB, UUP,
  BIL, 2010 → 2026-06-30, cached under `_cache/em_prices.parquet`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [612-em-debt-carry](../../612-em-debt-carry/) — the carry inside **USD-denominated** EM
  debt (EMB), no currency question at all. This study starts from the **local-currency** tape
  and asks whether *removing* the FX beats that USD sibling (it does not).
- [662-em-local-bonds](../../662-em-local-bonds/) — the **unhedged** EM-local premium as-is.
  This study adds the **FX-hedge proxy** overlay and re-races the *hedged* leg.
- [364-fx-carry-trade](../../364-fx-carry-trade/) — a **direct** long-high-yield /
  short-low-yield **currency** carry basket (the FX *is* the trade). Here the FX is the thing
  we try to **strip out**, keeping only the bond's local rate.
- [889-dollar-hedge-overlay](../../889-dollar-hedge-overlay/) — a dollar overlay on a
  different underlying. This study uses the UUP overlay specifically as an **EM-FX proxy
  hedge** on local-EM bonds, and grades the residual EM-basket beta it fails to cover.

None of the siblings race an **FX-hedged (UUP-proxy) local-EM leg against USD-EM debt** —
this study's own axis.
