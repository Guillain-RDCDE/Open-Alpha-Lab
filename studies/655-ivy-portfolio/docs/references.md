# References & literature map — Study 655 (Ivy Portfolio)

## The claim under test

- **The book.** Mebane Faber & Eric Richardson, *The Ivy Portfolio: How to Invest Like the
  Top Endowments and Avoid Bear Markets* (Wiley, 2009) — the retail-facing sequel to Faber's
  2007 SSRN paper *A Quantitative Approach to Tactical Asset Allocation* (the 200-day/10-month
  SMA timing rule tested single-asset in [Study 110](../110-faber-timing/)). The book's pitch:
  Yale- and Harvard-style endowments beat retail portfolios not through stock-picking but
  through **broad asset-class diversification** — domestic equity, foreign equity, real
  estate, bonds, commodities — and retail investors can approximate it with five liquid ETFs
  at 20% each, optionally timed with the same 10-month SMA rule Faber applied to a single
  asset in the SSRN paper.
- **The mechanism claimed.** Two separable claims, tested separately here: (1) the five asset
  classes are imperfectly correlated, so an equal-weight blend should show a better
  risk-adjusted profile (Sharpe, drawdown) than a plain stock/bond mix; (2) applying the
  10-month SMA independently to each of the five sleeves should further cut drawdowns by
  sidestepping each asset's own bear markets, the same mechanism Faber demonstrated for a
  single asset (SPY) in the SSRN paper.
- **The adjacent (distinct) results on this desk.** [68-all-weather](../68-all-weather/) is
  risk-*parity* weighting (inverse-vol, not equal-weight) on a different quartet (SPY/IEF/GLD/
  DBC). [144-permanent-portfolio](../144-permanent-portfolio/) and
  [203-golden-butterfly](../203-golden-butterfly/) are equal-weight recipes too, but with
  **no timing overlay** and a different asset mix (cash/gold/short+long bonds, no REITs or
  broad commodities). [110-faber-timing](../110-faber-timing/) is the *same* 10-month
  (200-day) SMA rule, but applied to **one asset** (SPY) — this study is the composite,
  5-sleeve version of that exact rule. [592-dual-momentum-gem](../592-dual-momentum-gem/) is
  a **relative**-momentum switcher between two equities plus an absolute-momentum bond gate —
  a decision tree, not a diversified equal-weight blend with an independent-sleeve timer. None
  of the five is this study's specific pairing: **20% × 5 (US equity / foreign equity / REITs
  / bonds / commodities), with and without a per-sleeve 10-month SMA.**

## What we measure, and the honesty rails

- **Two claims, kept separate.** The static allocation (does diversification help
  risk-adjusted return?) and the timing overlay (does the SMA add alpha or just cut risk?) are
  measured and stamped **independently** — collapsing them into one number was the single
  biggest way earlier drafts of this kind of study can mislead.
- **Excess-of-cash throughout.** Every Sharpe — static vs 60/40, timed vs static, timed vs
  60/40 — is computed excess of BIL (the same instrument that plays the timer's cash leg), so
  no race compares a raw Sharpe to an excess Sharpe.
- **One documented execution lag.** The SMA signal uses the price through month *t-1*'s
  close; the resulting position earns month *t*'s return. Applied once, at the source
  (`strategy.sma_signal`), never re-shifted downstream.
- **Costs are one-way × NAV** on the total absolute weight change at each monthly rebalance —
  the same convention used desk-wide, charged to both arms so the comparison is fair.
- **A matched-exposure random-timing control**, not just a raw active-return test, isolates
  whether the 10-month rule's specific *timing* — not merely time spent out of the market —
  carries information (mirrors the random-switching control in
  [592-dual-momentum-gem](../592-dual-momentum-gem/) and the random-timing control in
  [110-faber-timing](../110-faber-timing/)).
- **Bootstrap Sharpe-difference CIs** (circular block bootstrap, stable across 3/6/12-month
  block sizes) carry the Signal-axis claim about the static allocation, since the raw HAC
  *t* on active return alone doesn't capture a risk-adjusted (not just mean-return) claim —
  the same convention as [144-permanent-portfolio](../144-permanent-portfolio/) and
  [203-golden-butterfly](../203-golden-butterfly/).
- **Sub-period check (ex-GFC).** Both findings — the static underperformance and the timer's
  drawdown cut — are re-run with 2007-07→2009-06 dropped, so neither verdict rests on one
  crisis quarter alone.

## Data sources

- **VTI, VEU, VNQ, AGG, DBC, BIL** daily auto-adjusted (total-return) closes — yfinance (no
  key), cached under `_cache/ivy_prices.csv`, 2003-01-02 → 2026-06-30. BIL's 2007-05-30
  inception is the binding constraint on the joint monthly window (2007-06-30 start).
- Mebane Faber, *A Quantitative Approach to Tactical Asset Allocation*, SSRN 962461 (2007,
  updated 2013) — the single-asset SMA-timing mechanism this study applies per-sleeve.
- Mebane Faber & Eric Richardson, *The Ivy Portfolio* (Wiley, 2009) — the five-asset
  endowment-style allocation and its own reported timing overlay.
- Geetesh Bhardwaj, Gary Gorton & K. Geert Rouwenhorst, *Facts and Fantasies about Commodity
  Futures Ten Years Later* (2015) and Erb & Harvey, *The Strategic and Tactical Value of
  Commodity Futures* (2006) — the commodity-futures roll-yield literature explaining DBC's
  weak standalone decade on this tape.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [68-all-weather](../68-all-weather/) — risk-**parity** (inverse-vol weighted), a different
  four-asset mix (SPY/IEF/GLD/DBC), no timing overlay. This study: **equal-weight**, five
  assets including REITs and foreign equity, **with** an optional per-sleeve timer.
- [110-faber-timing](../110-faber-timing/) — the identical 10-month/200-day SMA rule, but on
  **one** asset (SPY). This study runs the same rule **independently across five sleeves** and
  asks whether the single-asset risk-reduction-not-alpha finding generalises to a diversified
  book — it does.
- [144-permanent-portfolio](../144-permanent-portfolio/) — Harry Browne's 25/25/25/25
  (stocks/long bonds/gold/cash), **no** timing, no REITs, no broad commodities, no foreign
  equity.
- [203-golden-butterfly](../203-golden-butterfly/) — the Permanent Portfolio plus a
  small-cap-value fifth leg (IWN), also **no** timing overlay.
- [592-dual-momentum-gem](../592-dual-momentum-gem/) — a **relative + absolute momentum
  decision tree** switching between two equity indices and a bond fallback; not a diversified
  equal-weight blend, and the timing logic picks *which single asset to hold*, not whether to
  hold each of several sleeves independently.

None of the five siblings tests Faber's specific pairing — the endowment-style five-asset
equal weight, and its independent per-sleeve 10-month timer — together. That combination is
this study's own axis.
