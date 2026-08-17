# References & literature map — Study 957 (Holdco Discount)

## The claim under test

- **The holdco-discount thesis.** A listed holding company whose value is dominated by one
  *listed* stake trades, almost always, below the marked value of that stake. The retail
  version of the pitch is arithmetic: "you are buying EUR 1 of LVMH for 79 cents through
  Christian Dior, so you get the same asset plus a free 21 cents." The professional version
  adds a mechanism: the gap is a *deviation*, it has a long-run average, and deviations from
  a long-run average revert — so buy when the discount is unusually wide, hedge out the
  underlying, and collect the narrowing.
- **What must be true for that to pay.** Two things, and this study tests them in that
  order. First the discount must be **stationary enough to revert** — a wide gap today must
  predict a narrower gap in six months. Second the reversion must be **big enough to survive
  the trade**: a hedged pair costs two legs of commission and a borrow fee on the short, and
  it earns nothing at all if the discount merely wanders.
- **The steelman for the sceptic.** Every standard explanation of the discount is a *level*
  story, not a *convergence* story: holding-company tax leakage on dividends passed up the
  chain, the controlling family's voting block making a takeover impossible, the corporate
  overhead of the holding vehicle, and the market's discount for a manager who might reinvest
  your capital badly. None of these has any reason to shrink over the next six months. If the
  discount is a *price of control* rather than a mispricing, it can stay wide, or widen, for
  decades — and the hedged buyer just pays borrow while waiting.

## Where the idea comes from

- **Closed-end fund discounts.** Lee, Shleifer & Thaler (1991), *Investor Sentiment and the
  Closed-End Fund Puzzle*, Journal of Finance — the canonical treatment of a listed vehicle
  trading away from an observable NAV, and the origin of the "discounts mean-revert with
  sentiment" reading. Pontiff (1996), *Costly Arbitrage: Evidence from Closed-End Funds*,
  Quarterly Journal of Economics — deviations are larger precisely where arbitrage is
  costlier, which is the mechanism our borrow sweep is aimed at.
- **Conglomerate and holding-company discounts.** Berger & Ofek (1995), *Diversification's
  Effect on Firm Value*, Journal of Financial Economics — the sum-of-the-parts discount as a
  valuation fact. Cornell & Liu (2001), *The Parent Company Puzzle*, Journal of Corporate
  Finance, and Lamont & Thaler (2003), *Can the Market Add and Subtract? Mispricing in Tech
  Stock Carve-Outs*, Journal of Political Economy (the 3Com/Palm case) — the extreme cases
  where a parent trades below its listed subsidiary stake alone, and where the "obvious"
  arbitrage was blocked by exactly the frictions we charge here.
- **Control and voting value.** Nenova (2003), *The Value of Corporate Voting Rights and
  Control*, Journal of Financial Economics — the discount as a *price*, not an error. This is
  the reading our result is most consistent with.
- **Limits of arbitrage.** Shleifer & Vishny (1997), *The Limits of Arbitrage*, Journal of
  Finance — a converging trade that can diverge first is not a free lunch, and the divergence
  is precisely what our always-on hedged arm measured.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica —
  [`strategy.newey_west_t`](../holdco_nav/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Panel standard errors under overlap and cross-name correlation.** Driscoll & Kraay
  (1998), *Consistent Covariance Matrix Estimation with Spatially Dependent Panel Data*,
  Review of Economics and Statistics — [`strategy.hac_ols_panel`](../holdco_nav/strategy.py).
  Two of our seven names (Naspers and Prosus) sit on the *same* underlying, Tencent, so a
  pooled t-stat that assumed independence would be flatly wrong. Hodrick (1992) and Valkanov
  (2003) on overlapping long-horizon regressions are the reason the forward window is
  HAC-corrected at 1.5x its own length rather than trusted naively.
- **Circular block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap*, JASA —
  [`strategy.bootstrap_sharpe_ci`](../holdco_nav/strategy.py) and
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).

## Related desk studies (dedup)

- **[Study 367 — CEF-Discount](../../367-closed-end-fund-discount/)** and
  **[Study 910 — Managed-Distribution CEF](../../910-managed-distribution-cef/)**: closed-end
  *funds*, whose NAV is **published by the manager** every evening. Study 957 is the case
  where nobody publishes a NAV and we have to *build* one from the tape — which is why the
  assumption table, not the statistics, is the hard part here.
- **[Study 378 — ETF-NAV-Premium](../../378-etf-nav-premium/)**: an intraday premium/discount
  proxy on an ETF with a **creation-redemption mechanism** that forces convergence within a
  day. A holdco has no such mechanism, and that absence is the whole point.
- **[Study 620 — A-H Share Premium](../../620-a-h-premium/)**: the *same company* in two
  venues, kept apart by capital controls. Ours is *two different companies* in the same venue,
  kept apart by control and tax.
- **[Study 239 — Spinoffs](../../239-spinoffs/)**: what happens *after* a parent separates
  from a subsidiary. Study 957 is the period *before* — while the chain is still intact.
- **[Study 366 — Merger-Arbitrage](../../366-merger-arbitrage/)**: a spread with a contractual
  convergence date. The holdco discount has none, which is the difference that decides it.

## The NAV proxy — every assumption, named

The discount is `1 - P_holdco / (k * P_stake + other_per_share)`, built from **price-only**
(split-adjusted, dividend-unadjusted) closes. Returns in the backtest come from
**total-return** closes. `k` and `other_per_share` are frozen constants; in reality both drift
as holdcos buy back shares and sell down stakes, so the **level** of every series below is an
assumption and only its **trailing-standardised variation** is used in the tests.

| Holdco | Stake | `k` | source of `k` | truncated at | why |
|---|---|---|---|---|---|
| Heineken Holding (HEIO.AS) | Heineken NV (HEIA.AS) | 0.9995 | 50.005% of HEIA over HEIO's own count | — | — |
| Christian Dior (CDI.PA) | LVMH (MC.PA) | 1.1745 | ~42.4% of LVMH over Dior's count | — | — |
| Liberty Broadband (LBRDK) | Charter (CHTR) | 0.3161 | 45.6m CHTR over ~144m LBRD shares; `other` = −20.1/share (GCI Alaska minus net debt) | 2024-11-12 | Charter and Liberty Broadband signed a definitive all-stock merger agreement on 2024-11-13; from that date the gap is a **merger spread** with a contractual exchange ratio and a convergence date — Study 366's subject, not a holdco discount |
| Naspers (NPSNY) | Tencent (TCEHY) | 0.30062 | *anchored* to the reported ~40% discount at 2026-06-30 | — | ADR ratios and the chain through Prosus make a bottom-up count unverifiable from the tape |
| Prosus (PROSY) | Tencent (TCEHY) | 0.22507 | *anchored* to ~30% at 2026-06-30 | — | as above; overlaps Naspers by construction |
| Bollore (BOL.PA) | Vivendi (VIV.PA) | 1.34292 | *anchored* to ~50% at 2024-11-29 | 2024-11-30 | Vivendi split into four listed pieces in December 2024; the NAV identity dies there |
| SoftBank (SFTBY) | Alibaba (BABA) | 0.09203 | *anchored* to ~45% at 2021-12-31 | 2021-12-31 | SoftBank disposed of essentially the whole stake through 2022-23 prepaid forwards |

All three truncations are chosen on **corporate events announced in advance**, never on
performance, and all three *shorten* the sample. The NAV-calibration sweep re-runs every
headline with `k` scaled from 0.8x to 1.2x and the `other` term from 0.5x to 1.5x.

The Liberty Broadband cut is the one an audit had to force. Cutting it removes 406 trading
days (14% of that name's sample) in which the "discount" was a live deal spread — the one
window in this panel where convergence was *contractual*, i.e. the only window where the
thesis was guaranteed to work — so keeping it would have been the generous choice, not the
conservative one. It changes nothing that matters (the timed pair's net and gross Sharpes are
identical to three decimals) and it flips the pooled mean-reversion slope from +0.0005 to
−0.0002, still with a *t* of essentially zero. Both versions are in `docs/results.md`.

## Free parameters, and the sweep that covers them

`enter = 1.0` / `exit = 0.0` (in trailing-z units) are the study's only genuinely free
parameters — `k` and the hedge ratio are derived, costs and borrow are priced inputs, and the
504-day standardisation window and 126-day forward horizon are conventions. All four are now
swept in [`examples/verify.py`](../examples/verify.py) (`strategy.threshold_sweep`,
`strategy.zwindow_sweep`, and the `fwd_days` argument to `mean_reversion_test`), and the grid
is printed whole rather than at the reported cell only. The headline cell is *not* the best
cell in the grid, and no cell in the grid produces a net *t* of 2.

**Named exclusions**, so the panel is a rule and not a taste:

- **Loews (L) / CNA Financial (CNA)**, **Icahn Enterprises (IEP) / CVR Energy (CVI)** — the
  listed stake is well under three-quarters of NAV, so the "observable" NAV would have been
  mostly an invented constant.
- **Strategy (MSTR) / bitcoin** — the tightest observable NAV on the tape, and unusable here:
  bitcoin-per-share more than tripled through at-the-market issuance, so no constant `k` is
  defensible over any useful window.
- **Liberty TripAdvisor (LTRPA) / Tripadvisor (TRIP)**, **Cannae (CNNE) / Dun & Bradstreet
  (DNB)** — both classic wide-discount names, both dropped because the take-privates of 2025
  left no usable price history on Yahoo. Their absence is a **survivorship bias in favour of
  the thesis**, since a discount that ends in a buyout is a discount that closed, and this
  panel cannot see any of them.

## Other non-tape inputs

- **Costs.** 10 bps one-way per leg on traded notional (an ASSUMPTION; swept 0 → 50 bps).
  Realistic for HEIA/MC/CHTR, optimistic for the ADR legs and for BOL.PA.
- **Borrow.** 100 bps/yr on the short leg (an ASSUMPTION; swept 0 → 600 bps). General
  collateral for CHTR or MC.PA, generous for TCEHY.
- **Cash leg.** `^IRX`, the 13-week T-bill *discount rate*, compounded daily — a quoted rate,
  not a tradable fund, used only to extend the excess-of-cash comparison back to 2004. It is
  cross-checked against BIL's actual total return over BIL's own window.

## Data sources

All series are daily Yahoo! Finance closes, kept in **two** conventions: `auto_adjust=True`
total-return closes for every *return*, and `auto_adjust=False` split-adjusted closes for
every *valuation*. Both legs of every pair quote in the same currency, so no FX conversion
enters any ratio. As-of **2026-06-30**; the partial current month is dropped so the sample
cannot creep between reruns.
