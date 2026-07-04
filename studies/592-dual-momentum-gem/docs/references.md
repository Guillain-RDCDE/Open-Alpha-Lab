# References & literature map — Study 592 (Dual Momentum — GEM)

## The claim under test

- **The source.** Gary Antonacci, *Risk Premia Harvesting Through Dual Momentum* (2012/2013,
  working paper, later *Journal of Management & Entrepreneurship*; SSRN 2042750,
  <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2042750>) and the retail-facing book
  *Dual Momentum Investing: An Innovative Strategy for Higher Returns with Lower Risk*
  (McGraw-Hill, 2014). GEM — **Global Equities Momentum** — is the flagship model: each month,
  (1) an **absolute momentum** gate compares US equities' trailing 12-month return against
  T-bills; (2) if it passes, **relative momentum** picks the better of US vs international
  equities; (3) if it fails, the portfolio hides in aggregate bonds. Antonacci's 1974–2011
  backtest reports roughly S&P-beating returns with about **half the maximum drawdown** — the
  exact composite claim we test.
- **The mechanism claimed.** Time-series (absolute) momentum harvests the persistence of
  multi-month market regimes (sidestepping long bears); cross-sectional (relative) momentum
  adds the US-vs-international rotation. Both legs lean on the canonical momentum literature:
  Jegadeesh & Titman (1993, *JF*), Moskowitz, Ooi & Pedersen (2012, *JFE*, time-series
  momentum), Asness, Moskowitz & Pedersen (2013, *JF*, value & momentum everywhere).

## Named siblings on this desk (the dedup map)

- [`146-country-momentum`](../../146-country-momentum/) tests **cross-sectional momentum across
  country equity ETFs** — a ranked breadth panel. GEM is not that: it is a **two-asset
  composite allocation rule** with an absolute-momentum gate.
- [`518-time-series-momentum`](../../518-time-series-momentum/) tests **time-series momentum on
  a diversified futures panel** (the Moskowitz-Ooi-Pedersen claim). GEM borrows that gate but
  packages it into a **retail switch between three ETFs** — the composite retail allocation
  product, which is what we grade here.

## The critical / post-publication literature

- Zakamulin, *The Real-Life Performance of Market Timing with Moving Average and Time-Series
  Momentum Rules* (2014, *Journal of Asset Management*): out-of-sample and
  transaction-realistic versions of timing rules are far weaker than in-sample backtests.
- Goyal & Welch (2008, *RFS*): predictors that look strong in-sample routinely fail
  out-of-sample — the frame for any post-publication decay test.
- McLean & Pontiff, *Does Academic Research Destroy Stock Return Predictability?* (2016, *JF*):
  published anomalies decay by ~1/3 to 1/2 after publication — our third axis applies exactly
  this question to GEM with a 2013 split.
- Practitioner replications (e.g. Allocate Smartly's live GEM tracking, and the widely-discussed
  post-2013 GEM shortfall vs the S&P 500) document the same whipsaw pattern we find in
  2015-16 / 2020 / 2023-25.

## Data sources

- **yfinance** (public, no key): daily auto-adjusted (total-return) closes for **SPY** (S&P 500,
  1993–), **EFA** (MSCI EAFE, 2001–), **AGG** (US Aggregate bonds, 2003–), **IEF** (7-10y
  Treasuries, 2002–), and **^IRX** (13-week T-bill discount yield). <https://finance.yahoo.com>.
- Bond leg = AGG spliced with IEF before AGG's Sept-2003 inception (both are the standard
  "safe leg" choices in GEM implementations; the splice is documented in
  [`data.py`](../dual_momentum_gem/data.py)).
- Everything cached under [`_cache/`](../_cache/) and re-run cache-first; the headline run is
  sliced to the frozen as-of (2026-06-30) and fingerprinted (`quantlab/repro.py`).

## Shared method citations

- Newey & West (1987, *Econometrica*): HAC standard errors — the *t* on mean monthly active
  return (serial correlation).
- Welch (1947): unequal-variance *t* for group splits (pre- vs post-publication; GEM vs the
  shuffled-timing baseline, averaged over 40 seeds — single-seed baselines are banned on this
  desk).
- The desk's inference bar: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — REAL requires a
  robust *t* ≥ 2 on the real tape; synthetic controls are machinery proofs, never market
  evidence.
