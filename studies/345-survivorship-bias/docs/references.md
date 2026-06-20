# References & literature map — Study 345 (Survivorship-Bias)

## The bias under test

- **The canonical definition.** Survivorship bias is the distortion that arises when a
  backtest is run on a universe assembled from *surviving* entities — the firms, funds, or
  securities still in existence at the sample's end — so the ones that died (delisted on
  bankruptcy, were merged away, or were dropped from an index) were never in the sample.
  The losers are deleted *before the test sees them*, biasing every long-biased read
  upward. This study turns that distortion into a *measurement*: run one strategy on a
  panel that keeps the dead names and on the same panel with them removed.

- **Mutual-fund survivorship — the founding empirical work.** Brown, Goetzmann, Ibbotson &
  Ross (1992), *Survivorship Bias in Performance Studies* (Review of Financial Studies
  5(4)); Malkiel (1995), *Returns from Investing in Equity Mutual Funds 1971–1991* (Journal
  of Finance) — dead funds are systematically worse, so a survivors-only sample overstates
  average fund returns by ~1–1.5%/yr. Carhart, Carpenter, Lynch & Musto (2002),
  *Mutual Fund Survivorship* (RFS), gives the modern treatment of multi-period survival.

- **CRSP and the delisting-return problem.** Shumway (1997), *The Delisting Bias in CRSP
  Data* (Journal of Finance), and Shumway & Warther (1999) for Nasdaq: when a stock is
  delisted (often on a near-total loss), naively dropping it — rather than booking the
  delisting return — biases measured returns *upward*, especially for small/distressed
  names. Our synthetic tape books an explicit delisting loss (a −80% terminal bar) for the
  doomed names, exactly the mechanism Shumway warns about.

- **Backtest survivorship in factor/strategy research.** Davis (1994) on the
  pre-Compustat-coverage bias; Kothari, Shanken & Sloan (1995) on survivorship in
  book-to-market tests; and the practitioner-side warnings in López de Prado (2018),
  *Advances in Financial Machine Learning* (Wiley), ch. on backtesting pitfalls — the
  current-membership index reconstruction is named there as a primary source of false
  discoveries.

## Why a contrarian "buy the losers" rule is the sharpest probe

- The bias points *upward* for any rule that ends up holding the names that recovered
  while never paying for the names that died. A contrarian rule buys recent losers — the
  exact population most likely to delist — so removing the dead names (survivors-only)
  rewards it for outcomes it could not have known and did not survive. Momentum/value
  factor work that documents this asymmetry: De Bondt & Thaler (1985) on long-term
  reversal (the original "buy the losers"), against which survivorship is a known confound.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.hac_tstat`](../survivorship_bias/strategy.py).
- **Circular block bootstrap.** Politis & Romano (1992), *A circular block-resampling
  procedure for stationary data* — CIs on the mean that preserve autocorrelation
  ([`strategy.block_bootstrap_ci`](../survivorship_bias/strategy.py)).
- **The survivorship guard.** Mirrors `quantlab.universe` / `quantlab.hf_data`: the
  current-membership real loader refuses to run without `allow_survivorship_bias=True`, and
  the caveat travels into the verdict — here the bias is the *subject*, so the opt-in is the
  point, not a footnote.

## Data sources used here

- **Yahoo! Finance** (via `yfinance` and the shared `quantlab.data` loader), total-return
  monthly closes, 20 US large-caps, 2005–2026 — used only as a *biased illustration* (the
  convenience universe is itself survivors-only). The headline run is pinned with an as-of
  date (2026-05-31, last full month) and content fingerprints (see
  [`docs/results.md`](results.md)). The offline reproducible core and the test-suite run on
  the deterministic [`data.synthetic_panel`](../survivorship_bias/data.py) generator, where
  the dead names can be toggled on and off — never the network.

## Related desk studies

- **[Study 350 — Dartboard-Portfolio](../../350-dartboard-portfolio/)**: the sibling
  research-method demo — same cross-sectional machinery and the same survivorship opt-in
  guard, applied to random selection instead of a deletion sweep.
- **[Study 140 — Amihud-Illiquidity](../../140-amihud-illiquidity/)**: a factor study whose
  premium is partly survivorship; this study is the controlled demonstration of *why* that
  caveat matters.
