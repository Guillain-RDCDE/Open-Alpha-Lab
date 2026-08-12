# References & literature map — Study 911 (REIT Quality Screen)

## The claim under test

- **The idea.** REITs are not one asset. **Equity REITs** own property and collect rents;
  their leverage is moderate and their income is a real, durable cash flow. **Mortgage
  REITs** (mREITs) own mortgage assets funded with short-term repo, running a *levered
  carry* between long-asset yields and short funding — a book that pays a fat dividend in
  calm times and detonates when funding tightens or spreads gap. A **"quality REIT" screen**
  therefore holds the durable-income equity sleeve and screens the levered-carry sleeve out,
  aiming for a better *risk-adjusted, net-of-cost* return than the broad REIT index.
- **The durable-income / low-leverage side.** The REIT literature documents that
  lower-leverage, higher-quality-cash-flow property REITs carry different risk than
  high-leverage names; residential and "core" property (apartments, healthcare, storage)
  have the stickiest rents. See the NAREIT / property-factor work on REIT leverage and
  quality (e.g. the "REIT quality factor" and low-leverage-REIT discussions in the
  practitioner literature) and Fama–French-style quality/profitability tilts applied to
  the REIT cross-section.
- **The mortgage-REIT / levered-carry side.** mREITs' total returns are notoriously
  poor relative to their dividend yields because principal erodes: the carry is periodically
  wiped out by rate/spread shocks (2008, 2013 taper, 2020). This is the "yield trap" —
  a high distribution yield masking a flat-to-negative total return. The comparison of
  mREIT total return vs equity-REIT total return is the direct test here.
- **The specific test.** We race the live liquid vehicles on an **excess-vs-excess Sharpe**
  basis (all minus the BIL T-bill): the residential quality sleeve **REZ** and an
  equal-weight equity-REIT book **(VNQ, REZ, RWR)** against the broad index **VNQ**, and the
  mortgage-REIT sleeve **REM** against both — with a Newey-West HAC *t* on monthly spreads, a
  paired block-bootstrap Sharpe-advantage CI, a two-era cut, daily drawdowns, a costed book,
  and a seeded synthetic control.

## What we measure, and the honesty rails

- **Excess-vs-excess only.** Every Sharpe race and every advantage is over the tradable BIL
  T-bill leg — no comparing a levered book's raw Sharpe to cash-heavy alternatives.
- **Total-return, net-of-fee tape.** yfinance `auto_adjust=True` closes fold dividends and
  fees back in — essential here, because the whole mortgage-REIT story is *price* decaying
  while the *distribution* looks fat.
- **HAC inference.** Newey-West (Bartlett, 6-lag) *t* on monthly spreads — overlapping and
  serially correlated REIT returns make a plain *t* overstate significance.
- **Bootstrap Sharpe-advantage CI.** A paired circular-block bootstrap preserves both the
  serial dependence and the cross-sectional pairing, so "is the Sharpe edge distinguishable
  from zero?" is answered honestly, not with a delta-method point estimate.
- **Short-history / young-ETF caveat, on the Signal axis.** XLRE lists only 2015-10; the
  sample is one deep cycle (GFC) plus 2020 — magnitudes are indicative.
- **One documented rebalance lag; costs graded separately.** The quality book is
  point-in-time (weights set at `t−1` close, held at `t`) and charged one-way × NAV ×
  turnover per monthly rebalance against a buy-and-hold benchmark.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* on the monthly spreads).
- **Politis, D. & Romano, J. (1994)** — the stationary / circular block bootstrap (the paired
  Sharpe-advantage CI).
- **Lo, A. (2002)** — the statistics of the Sharpe ratio (why a Sharpe needs a standard error
  before it means anything).

## Data sources

- **yfinance daily OHLC** (`auto_adjust=True`, total-return), tickers VNQ, REZ, RWR, XLRE,
  REM, SPY, BIL, 2005-01-03 → 2026-06-30, cached under this study's own `_cache/`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [207-reits-diversifier](../../207-reits-diversifier/) — tests whether the **broad REIT
  index diversifies** an equity/bond portfolio (a correlation/allocation question). This
  study instead tests **quality *within* the REIT complex** — equity vs mortgage, durable vs
  levered — not REITs-vs-other-assets.
- [611-mreit-carry](../../611-mreit-carry/) — the **mortgage-REIT carry** trade in its own
  right (the levered-spread mechanics). Here mortgage REITs (REM) appear only as the *foil* —
  the trap the quality screen exists to avoid — not the object of study.
- [341-mlp-pipelines](../../341-mlp-pipelines/) — **energy MLP pipelines**, a different
  high-yield "income" complex (midstream toll roads). Same yield-trap *question*, different
  asset; no REIT overlap.
- [246-defensive-sectors](../../246-defensive-sectors/) — a **low-volatility / defensive
  equity sector** tilt (utilities, staples). Real estate is a GICS sector but this study
  sorts *within* real estate on leverage/income quality, not across defensive sectors.

None of the siblings race a **durable-income equity-REIT sleeve against the broad REIT index
and the mortgage-REIT trap on an excess-vs-excess, net-of-cost basis** — this study's own axis.
