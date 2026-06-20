# References & literature map — Study 333 (Recovery-Speed)

## The claim under test

- **Drawdown-recovery momentum / "buy the fast recoverers."** A recurring piece of
  trading lore (and a staple of post-crash market commentary): after a deep, market-wide
  drawdown, the stocks that climb back to their pre-drawdown highs *fastest* are the
  "strong hands," and that strength persists — so a basket of the fastest recoverers
  keeps leading the laggards. It is a cross-sectional cousin of time-series momentum,
  conditioned on a drawdown event. We state it at full strength and test it as a
  market-neutral long-fast / short-slow quintile book over a forward window, pooled
  across every independent drawdown in the sample.

## The real effects it leans on (and is easily confused with)

- **Cross-sectional momentum.** Jegadeesh & Titman (1993), *Returns to Buying Winners and
  Selling Losers* (Journal of Finance) — past 3–12 month winners outperform losers. A
  fast recoverer is, mechanically, a recent winner; any "recovery-speed" edge must be
  shown to be *more* than vanilla momentum measured at an arbitrary post-crash date.
- **Short-term reversal.** Jegadeesh (1990) and Lehmann (1990) — at short horizons, recent
  winners *under*perform. Measured a few weeks after a trough, a "fast recoverer" can just
  as easily mean-revert as continue. The sign of a drawdown-conditioned sort is not
  obvious a priori, which is exactly why it needs testing rather than assertion.
- **52-week-high momentum.** George & Hwang (2004), *The 52-Week High and Momentum
  Investing* (Journal of Finance) — proximity to the 52-week high predicts returns.
  Recovery speed is a different parameterisation of the same "how close are you to your
  old high" idea; the desk has tested the raw 52-week-high anomaly elsewhere.
- **Beta and the recovery rally.** A post-trough rally is a market move; high-beta names
  recover faster *and* keep rallying simply because the market is rising. We therefore
  report the long-short in **excess of the market** (market-neutral) so we never credit
  recovery-rally beta as recovery-speed alpha.

## Why one event is not evidence — selection and multiple testing

- **Single-event significance is a cherry-pick.** Harvey, Liu & Zhu (2016), *…and the
  Cross-Section of Expected Returns* (Review of Financial Studies) — the multiple-testing
  problem. With only a handful of drawdowns in any sample, picking the one that "worked"
  (here the GFC) inflates significance; the honest test pools all independent episodes.
- **Survivorship direction.** Brown, Goetzmann, Ibbotson & Ross (1992), *Survivorship Bias
  in Performance Studies* (Review of Financial Studies). A current-survivor universe omits
  the names that drew down and never recovered — biasing a "short the slow recoverers"
  book *in favour* of the claim. We reason about the direction explicitly: the bias works
  *for* the effect, so a null result on survivors is conservative.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West standard errors.** Newey & West (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*
  (Econometrica) — [`strategy.hac_tstat`](../recovery_speed/strategy.py).
- **Bootstrap confidence intervals.** Efron & Tibshirani (1993), *An Introduction to the
  Bootstrap*. We resample names within each leg for the long-short CI; the synthetic
  control uses a permutation (shuffle) null on the recovery-speed scores.
- **Drawdown definition.** Peak-to-trough on the running maximum, recovery declared when a
  fraction of the loss is regained — the standard maximum-drawdown construction.

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`), auto-adjusted (total-return proxy),
  2004–2026. A small universe of long-lived large-caps + SPY as the market proxy. All
  headline numbers are pinned with an as-of date and content fingerprint
  ([`docs/results.md`](results.md)). The offline reproducible core and test-suite run on
  the deterministic [`data.synthetic_panel`](../recovery_speed/data.py) generator, which
  plants a tunable recovery-momentum edge (the positive control) and a null, and never
  touches the network.

## Related desk studies

- **Mean-reversion & stat-arb family.** The drawdown-conditioned sort is a stat-arb idea;
  it shares the cross-sectional long-short machinery with the desk's other ranking studies
  and the honest-inference discipline (HAC *t*, bootstrap CI, market-neutral race) used
  throughout the bench.
- **Momentum cousins.** Any cross-sectional momentum study on the bench is the natural
  comparison: recovery speed is momentum measured at a drawdown-conditioned date, and the
  question is whether the conditioning adds anything beyond plain momentum (it does not,
  here).
