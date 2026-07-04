# References & literature map — Study 594 (Leverage Rotation, TQQQ + 200SMA)

## The claim under test

- **The strategy's canonical write-up.** Michael A. Gayed, *Leverage for the Long Run — A
  Systematic Approach to Managing Risk and Magnifying Returns in Stocks* (2016, SSRN
  2741701; 2016 Dow Award). The rule: hold leveraged equity exposure when the index is
  above its 200-day moving average, de-lever/park in T-bills below — because volatility
  clusters and is regime-dependent on the SMA state, and daily-reset leverage compounds
  best in low-vol uptrends. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2741701
- **The retail phenomenon.** The plan is one of the most-reposted strategies on
  r/LETFs and r/wallstreetbets ("TQQQ + 200SMA", "HFEA's cousin", "the 9Sig alternative"),
  usually shown from 2010+ backtests of real TQQQ — a window that begins *after* the
  2000-02 regime this study restores. Representative community backtests:
  https://www.reddit.com/r/LETFs/ (search "200 MA TQQQ") and testfol.io-style simulators.
- **The vehicle.** ProShares UltraPro QQQ (TQQQ) prospectus — 3x the *daily* return of
  the Nasdaq-100, expense ratio ~0.95%/yr, financing via swaps: the daily-reset design
  is exactly what the constant-leverage identity models.
  https://www.proshares.com/our-etfs/leveraged-and-inverse/tqqq

## Why leverage + SMA in the first place (the mechanism literature)

- **Volatility clustering / regime dependence.** Gayed's own exhibit — and ours: below
  the 200SMA, next-day index vol runs ~2× the above-line vol. The vol channel is
  well-documented: Schwert (1989); Mandelbrot (1963) on clustering. Moskowitz, Ooi &
  Pedersen (2012, *Time series momentum*, JFE) for the trend-following cousin.
- **Leveraged-ETF mechanics.** Cheng & Madhavan (2009, *The Dynamics of Leveraged and
  Inverse Exchange-Traded Funds*, JOIM); Avellaneda & Zhang (2010, *Path-dependence of
  Leveraged ETF Returns*, SIAM J. Fin. Math.): daily-reset compounding = 3× the period
  return **only** path-dependently — variance drag `(k(k−1)/2)·σ²` vs compounding-in-trend.
  The desk's own [100-melting-ice](../100-melting-ice/) validated the constant-leverage
  identity at daily corr 0.999 vs real TQQQ/UPRO; this study reuses that identity with an
  all-in fee calibrated once (2.5%/yr) against the real TQQQ tape.
- **SMA timing on the unlevered index.** Faber (2007, *A Quantitative Approach to Tactical
  Asset Allocation*, JWM) — the desk tested it in [110-faber-timing](../110-faber-timing/):
  a real drawdown shield, not a certified return engine. This study is the 3x-levered
  retail escalation of exactly that rule.

## Why the desk's tests are shaped this way

- **Random-timer baseline (exposure + switch matched, ≥ 20 seeds).** An SMA rule spends
  ~74% of days long; naive comparisons confuse *exposure* with *timing*. We permute the
  strategy's own in/out run lengths (preserving exposure fraction AND switch count
  exactly), rebuild the strategy per seed, and average the Welch *t* over 40 seeds —
  the desk's guard against single-seed luck. Cf. the timing-vs-exposure logic in
  [110-faber-timing](../110-faber-timing/).
- **HAC (Newey-West 1987) t** on daily return differences — daily strategy returns are
  autocorrelated and heteroskedastic; the naive *t* flatters.
- **Welch (1947) t** for the above/below-SMA conditioning splits (mean channel and
  squared-return vol channel).
- **Sub-sample honesty.** McLean & Pontiff (2016, JF) on post-publication selection;
  the strategy's viral 2010+ backtests are a start-date selection — the full-1999 tape
  (with the synthesised 3x) is the honest sample, and the 2010+ *t* = +2.52 vs
  full-tape *t* = +1.66 contrast is reported as exactly that.

## Data sources used here

- **yfinance** daily auto-adjusted (total-return) closes: QQQ (1999-03-10 →), TQQQ
  (2010-02-11 →), plus **^IRX** (13-week T-bill discount yield) for the cash leg and the
  2x financing of the synthesised fund. Cached under `_cache/lr200_prices.csv`; all
  headline numbers pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup frame)

- [100-melting-ice](../100-melting-ice/) — **Real / Mirage / Busted**: the 3x
  decay *mechanic* (the identity this study's synthesis reuses) and why "decays to zero"
  is path-dependent, not a law.
- [110-faber-timing](../110-faber-timing/) — **Real / Fragile**: the un-levered 200SMA
  rule on SPY; a genuine drawdown shield that lags buy & hold on return.
- **This study is the specific retail combination of the two** — Faber's switch driving
  melting-ice's instrument — and tests the combined promise ("3x upside without 3x
  crashes") on the one regime (2000-02) the community backtests never include.
- [593-hfea-leveraged-6040](../593-hfea-leveraged-6040/) — the other Reddit-famous
  leveraged plan (HFEA), same lot.
