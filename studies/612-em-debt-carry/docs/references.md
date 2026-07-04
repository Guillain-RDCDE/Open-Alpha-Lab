# References & literature map — Study 612 (EM Debt Carry)

## The claim under test

- **The pitch.** Emerging-market sovereign bonds yield hundreds of basis points over
  Treasuries (the EMBI spread has averaged roughly 300-450 bp since the index began) — a "fat
  carry" that allocators are told compounds into outperformance. The packaged retail versions
  are **EMB** (iShares J.P. Morgan USD Emerging Markets Bond ETF, tracking the EMBI Global
  Core, launched 2007-12, https://www.ishares.com/us/products/239572/) and **EMLC** (VanEck
  J.P. Morgan EM Local Currency Bond ETF, tracking the GBI-EM, launched 2010-07,
  https://www.vaneck.com/us/en/investments/emerging-markets-local-currency-bond-etf-emlc/).
- **The classic literature on whether the spread overpays for default.**
  - Klingen, Weder & Zettelmeyer (2004), *How Private Creditors Fared in Emerging Debt
    Markets, 1970–2000* (IMF WP 04/13): long-run EM debt returns were roughly Treasury-like
    once defaults are counted — the spread compensated for realised losses, little more.
  - Meyer, Reinhart & Trebesch (2022), *Sovereign Bonds since Waterloo* (QJE 137(3)): 200
    years of external sovereign bonds earned an excess return of ~3-4%/yr *including*
    defaults — a real but crisis-punctuated premium. The two papers bracket the honest debate
    this study drops onto a modern, investable tape.
  - Borri & Verdelhan (2011), *Sovereign Risk Premia* (working paper/AFA): EM sovereign
    returns co-move with US equity/consumption risk — the spread is compensation for
    *systematic* risk taken exactly at the wrong time, the formal version of our
    equity-beta finding.
- **The carry-crash lens.** Brunnermeier, Nagel & Pedersen (2008), *Carry Trades and Currency
  Crashes* (NBER Macro Annual): carry returns are negatively skewed and crash when funding
  liquidity dries up — "going up by the stairs, down by the elevator." Our monthly EMB−IEF
  spread (skew −1.72, worst-decile-SPY gap −509 bps/mo) is the bond-fund incarnation.
- **Local currency vs hard currency.** Du & Schreger (2016), *Local Currency Sovereign Risk*
  (JF 71(3)): local-currency EM debt carries a distinct currency-depreciation risk on top of
  credit — the mechanism behind EMLC's coupon (+5.6%/yr) failing to reach the total return
  (−0.6%/yr vs IEF).

## What we measure, and why

- **Total-return vs price-only, twice.** yfinance `auto_adjust=True` closes are total-return;
  `auto_adjust=False` closes are split-adjusted price-only. The monthly difference is the
  fund's **coupon/distribution component** — the *promised* carry — with no index data needed.
  The TR spread EMB−IEF is the *collected* carry: same currency (USD), matched duration
  (~7y vs 7-8y), so it isolates the EM credit premium rather than a duration bet.
- **HAC everywhere.** Monthly bond spreads are serially correlated; every mean is tested with
  a Newey-West *t* (6 lags; 3/12 shown as robustness). Newey & West (1987, Econometrica).
- **Welch on the group split.** The risk-off conditional gap (worst-decile SPY months vs the
  rest) uses Welch (1947) — unequal variances by construction in crash months.
- **Excess-vs-excess.** Sharpe races subtract BIL from both legs (house rule; no
  cash-vs-total mismatches).
- **Fixed crisis windows.** The five windows (GFC 2008-09→2009-03, taper tantrum
  2013-05→2013-09, EM stress 2018-04→2018-11, COVID 2020-02→2020-04, rate/Russia
  2022-01→2022-10) are documented calendar dates from the events themselves, not fitted to
  the spread series.

## Method lineage (the desk's shared engine)

- **Dual-tape coupon decomposition.** [`data.fetch_panel`](../em_debt_carry/data.py) /
  [`strategy.dividend_component`](../em_debt_carry/strategy.py) — the same TR-minus-price-only
  construction as the packaged-carry siblings (below).
- **HAC spread + NW regression.** [`strategy.nw_tstat`](../em_debt_carry/strategy.py),
  [`strategy.nw_regression`](../em_debt_carry/strategy.py) — the carry-premium alpha strips
  duration (IEF) *and* the hidden equity beta (SPY).
- **Risk-off conditional split.** [`strategy.riskoff_profile`](../em_debt_carry/strategy.py) —
  the carry-crash profile, Welch *t* on the worst-decile-SPY gap.
- **Crisis ledger.** [`strategy.crisis_ledger`](../em_debt_carry/strategy.py) — the third-axis
  arithmetic: cumulative spread surrendered inside the five windows vs collected outside.
- **Deterministic synthetic control.** [`data.synthetic_world`](../em_debt_carry/data.py) —
  planted spread / coupon / risk-off beta knobs; the null must not manufacture significance.

## Data sources used here

- **yfinance** daily closes (total-return and price-only) for EMB, EMLC, IEF, SPY, BIL,
  2007-01 → as-of 2026-06-30, cached under `_cache/edc_tr.csv` and `_cache/edc_px.csv`. All
  headline numbers are pinned in [`docs/results.md`](results.md) (fingerprint `1bff8a946cdf`)
  and reproduced by [`examples/verify.py`](../examples/verify.py).
- Index/fund facts: iShares EMB and VanEck EMLC product pages (expense ratios 0.39% / 0.30%,
  inception dates, effective durations); J.P. Morgan EMBI Global Core / GBI-EM index
  methodology (issue-size and liquidity screens).

## Related desk studies (the packaged-carry family — the dedup map)

- [610-fallen-angels-premium](../../610-fallen-angels-premium/) and
  [611-mreit-carry](../../611-mreit-carry/) — the **same packaged-carry family**: a fat quoted
  yield delivered through an index fund, decomposed on the same dual TR/price-only tape. 610
  is *credit selection inside high yield*; 611 is *levered MBS net-interest-margin*. This
  study is the **sovereign credit** member: the coupon is real, but the collected spread is
  equity beta plus a crisis left tail.
- [364-fx-carry-trade](../../364-fx-carry-trade/) — the **currency** side of the EM carry:
  our EMLC−EMB contrast (−2.8%/yr, the FX leg's bill) is exactly where that study's subject
  enters this one.
- [69-safe-haven](../../69-safe-haven/) — the flight-to-quality mechanics that sit on the
  other side of every one of our five crisis windows.
