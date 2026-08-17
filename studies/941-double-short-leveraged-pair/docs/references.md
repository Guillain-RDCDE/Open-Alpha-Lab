# References & literature map — Study 941 (Short Both Legs)

## The claim under test

- **The double-short folklore.** A daily-reset levered fund loses roughly
  `0.5 * L * (L - 1) * sigma^2` per year to its own rebalancing — about 3σ² for a 3x long
  and 6σ² for a −3x inverse, which on the Nasdaq-100's ~21% volatility is ~13% and ~26% a
  year. Since TQQQ and SQQQ track the *same* index with opposite signs, the retail-forum
  conclusion is irresistible: short both in equal dollars, the index exposure cancels, and
  you are left holding the decay of both — a market-neutral money machine. The claim
  circulates continuously on r/investing, Seeking Alpha and Bogleheads, usually with a
  chart of TQQQ+SQQQ held long going to zero as the "proof".
- **The steelman.** The decay is arithmetically undeniable, the pair is liquid, and the
  hedge is exact by construction (both funds reset at the same close on the same index).
  The trade's failure modes are therefore mechanical, not statistical: borrow cost, the
  rebalance schedule, and — the one the folklore never states — whether a *constant-dollar*
  short collects a *geometric* effect at all.

## Why the mechanism is not what it looks like

- **Cheng & Madhavan (2009), *The Dynamics of Leveraged and Inverse Exchange-Traded
  Funds*, Journal of Investment Management.** The canonical derivation of the daily-reset
  path dependency and the end-of-day rebalancing flow. The decay term is a property of the
  *compounded* return path; the fund's *expected daily* return remains L times the index
  less fees and financing. That distinction is exactly why a book held at constant dollars
  and reset daily collects the fee load and not the decay.
- **Avellaneda & Zhang (2010), *Path-Dependence of Leveraged ETF Returns*, SIAM Journal on
  Financial Mathematics.** Derives and empirically verifies the log-return decomposition
  `log(L-fund) = L * log(index) - 0.5 * L * (L - 1) * sigma^2 * T` and shows the tracking
  is excellent at the daily horizon and poor at the multi-month horizon — the arithmetic
  behind our leg regressions (β = +2.96 / −2.96 on daily returns).
- **Lu, Wang & Zhang (2012), *Long Term Performance of Leveraged ETFs*, Financial Services
  Review** — documents that holding periods matter more than leverage, and that pairs of
  long/inverse funds do not net to a free lunch once carry and financing are credited.
- **Trainor & Baryla (2008), *Leveraged ETFs: A Risky Double That Doesn't Multiply by
  Two*, Journal of Financial Planning** — early, practitioner-facing statement of the same
  compounding arithmetic.
- **Shum, Hejazi, Haryanto & Rodier (2016), *Intraday Share Price Volatility and Leveraged
  ETF Rebalancing*, Review of Finance** — the funds' own end-of-day rebalancing is costly
  and costlier in stressed markets. The real-tape harvest widens to +6.8% in 2020 and +3.6%
  in 2022, which is *consistent with* the residual being the funds' own costs. It is
  circumstantial, not proof: this study measures a residual and does not decompose it into
  expense ratio, financing spread, internal trading and tracking.

## Why borrow decides it

- **D'Avolio (2002), *The Market for Borrowing Stock*, Journal of Financial Economics**;
  **Geczy, Musto & Reed (2002), *Stocks Are Special Too*, JFE**; **Kolasinski, Reed & Ringgenberg
  (2013), *A Multiple Lender Approach to Understanding Supply and Search in the Equity
  Lending Market*, Journal of Finance.** The lending market prices exactly the arbitrage
  the borrower is trying to capture: when a security is worth shorting for a mechanical
  reason, the fee rises to meet it. Our breakeven of 2.06%/yr sits in the range these
  papers describe as ordinary for a crowded but liquid name — which is why the borrow
  assumption is swept end to end rather than assumed.
- **Duffie, Gârleanu & Pedersen (2002), *Securities Lending, Shorting, and Pricing*, JFE.**
  The theory of the lending fee as the price of the short's edge. Applied here: the fee an
  inverse ETF commands is, to first order, the fee load an arbitrageur would otherwise
  collect from it.
- **No public borrow-fee tape.** Historical stock-loan rates are a paid dataset. The rate
  in this study is therefore an explicit **ASSUMPTION**, and so are the account terms (the
  short rebate). Both are swept, and the breakeven is reported as the headline number.
  Consequence to keep in view: **every harvest figure in this study is gross of borrow**,
  which is the one cost the trade cannot avoid and the one this desk cannot price.

## Related desk studies (dedup)

- **[Study 154 — Leverage Anomaly](../../154-leverage-anomaly/)**: the *cross-sectional*
  low-beta/leverage anomaly in single stocks. Nothing to do with the daily-reset mechanic.
- **[Study 593 — HFEA Leveraged 60/40](../../593-hfea-leveraged-6040/)** and
  **[Study 594 — Leverage Rotation](../../594-leverage-rotation-200sma/)**: leveraged ETFs
  used as an *allocation* (long TQQQ/UPRO in a portfolio, timed or not). Study 941 is the
  opposite side of the same instruments — a *short* book whose whole return is the funds'
  cost load — and it is the only one of the three that is market-neutral by construction.
- **[Study 943 — Reset Frequency](../../943-leverage-reset-frequency/)** asks what a
  *monthly-reset* levered fund would have done; Study 941 asks what happens when the
  *investor's* reset frequency changes on a short of the existing daily-reset pair. The
  overlap is deliberate and the results are complementary: 943 prices the fund's schedule,
  941 prices the arbitrageur's.
- **[Study 375 — VXX Roll Decay](../../375-vxx-roll-decay/)** and
  **[Study 661 — USO Roll Decay](../../661-uso-roll-decay/)**: shorting a *structurally
  decaying* product, but the decay there is a **roll yield** (a real, collectable cash flow
  from the futures term structure), not a compounding artefact. Study 941 is the case where
  the "decay" turns out **not** to be collectable — the contrast with 375/661 is the point.
- Also in this lot: **942 (inverse-ETF structural loss)** and **945 (leverage financing
  cost)** measure the two halves of the load this study collects — the inverse fund's own
  drag and the financing rate embedded in a levered fund. Study 941 is the only one that
  asks whether an outside investor can *take* it.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite …
  Covariance Matrix*, Econometrica — [`strategy.newey_west_t`](../short_pair/strategy.py)
  and [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Return-difference (Sharpe comparison) t-stat.** Jobson & Korkie (1981), *Performance
  Hypothesis Testing with the Sharpe and Treynor Measures*, Journal of Finance —
  [`strategy.sharpe_diff_tstat`](../short_pair/strategy.py).
- **Circular block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap*, JASA —
  [`strategy.bootstrap_sharpe_ci`](../short_pair/strategy.py) and
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).

## Data sources

- **TQQQ** (ProShares UltraPro QQQ, 3x daily long), **SQQQ** (ProShares UltraPro Short QQQ,
  −3x daily), **QQQ** (the unlevered index ETF, beta benchmark) and **BIL** (1-3 month
  T-bill ETF, the cash leg) — daily **total-return** closes via `yfinance`
  (`auto_adjust=True`), common window 2010-02-11 (both levered funds' inception) →
  **2026-06-30**. Nothing here is price-only.
- **Non-tape inputs, all labelled and swept:** the **borrow fee** (0–15%/yr on gross short
  notional), the **short rebate** (full / none), the **one-way trading cost** (0–10 bps) and
  the **reset schedule** (daily / weekly / monthly / never). No borrow-fee or securities-
  lending history is used, because none is freely available.
- **As-of 2026-06-30.** The partial current month is dropped so the sample never creeps;
  the 2026 row of the calendar table is labelled as a partial (H1) year.
