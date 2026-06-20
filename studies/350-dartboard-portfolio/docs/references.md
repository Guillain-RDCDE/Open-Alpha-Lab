# References & literature map — Study 350 (Dartboard-Portfolio)

## The claim under test

- **The Malkiel dartboard.** Burton G. Malkiel (1973), *A Random Walk Down Wall Street*
  (Norton). The original quip: *"A blindfolded monkey throwing darts at a newspaper's
  financial pages could select a portfolio that would do just as well as one carefully
  selected by experts."* The testable hypothesis: **random stock *selection*, held
  equal-weight, does at least as well as the cap-weighted index and as expert
  concentration.** Note this is about *which names you pick*, distinct from how you weight
  a fixed set (Study 171).
- **The Wall Street Journal dartboard contest** (1988–2002). WSJ staff literally threw
  darts at the stock pages and raced the picks against a panel of professional analysts.
  The pros won the headline tally, but the dartboard's raw returns were close and the
  contest was widely (mis)read as "monkeys beat the pros."
- **The Research Affiliates "monkey" study.** Arnott, Hsu, Kalesnik & Tindall (2013),
  *The Surprising Alpha From Malkiel's Monkey and Upside-Down Strategies* (Journal of
  Portfolio Management). Thousands of *random* portfolios beat the cap-weighted index on
  average — the result that revived the legend. Crucially, they show the outperformance is
  a **size and value tilt** inherited mechanically from departing from cap weights, not
  stock-picking skill. This is the paper our teardown is built to reproduce and explain.

## Why random equal-weight beats cap-weight — the size mechanism

- **The size premium.** Banz (1981), *The Relationship Between Return and Market Value of
  Common Stocks* (Journal of Financial Economics); Fama & French (1992, 1993). Small-cap
  stocks have historically earned a premium. Any equal-weight scheme over-weights the many
  small names a cap index barely holds, so it harvests this premium mechanically — the
  source of the dartboard's "win" when small beats large.
- **Equal-weight vs cap-weight indexing.** Plyakha, Uppal & Vilkov (2012/2021), *Equal or
  Value Weighting? Implications for Asset-Pricing Tests* — the equal-weight portfolio's
  outperformance decomposes into factor exposures plus a rebalancing (contrarian) effect,
  not security selection. The dartboard is a noisy equal-weight portfolio, so it inherits
  exactly these properties.
- **The regime dependence.** When mega-caps lead (e.g. the 2013–2024 US tech run),
  cap-weighting concentrates into the winners and equal-weight/dartboard *under*-perform —
  the documented reversal of the legend (see our real-tape result and the recurring
  "equal-weight S&P 500 lags" coverage of the 2020s).

## The honest controls

- **Random portfolios as a benchmark.** Clare, Motson & Thomas (2013, Cass Business
  School), *An Evaluation of Alternative Equity Indices* — generating thousands of random
  portfolios as a Monte-Carlo distribution against which to judge any weighting scheme.
  Our 500-monkey draw is the same device: one throw is luck, the distribution is the test.
- **Distinguishing tilt from skill.** The decisive control is racing the dartboard against
  the *equal-weight index* (which already holds the size tilt). If the dartboard's edge
  over the cap index vanishes against the equal-weight index, the edge was the tilt, not
  selection — exactly Arnott et al.'s decomposition, run directly.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.hac_tstat_diff`](../dartboard_portfolio/strategy.py).
- **Circular block bootstrap.** Politis & Romano (1992), *A circular block-resampling
  procedure for stationary data* — preserves autocorrelation when CI-ing the mean spread
  ([`strategy.block_bootstrap_ci`](../dartboard_portfolio/strategy.py)).
- **Survivorship guard.** Mirrors `quantlab.universe` — the current-membership panel is
  opt-in (`allow_survivorship_bias=True`) and the caveat travels into the verdict.

## Data sources used here

- **Yahoo! Finance** (via `yfinance` and the shared `quantlab.data` loader),
  total-return monthly closes; market-cap snapshot via `fast_info`. Universe of 20 US
  mega-caps, 2013–2026. All headline numbers are pinned with an as-of date (2026-05-31,
  last full month) and a content fingerprint (see [`docs/results.md`](results.md)). The
  offline reproducible core and the test-suite run on the deterministic
  [`data.synthetic_panel`](../dartboard_portfolio/data.py) generator, never the network.

## Related desk studies

- **[Study 171 — Naive-1-Over-N](../../171-naive-1-over-n/)**: the *allocation* half of
  the question — 1/N equal-weight versus Markowitz optimisers on a *fixed* universe.
  Study 350 is the *selection* half — does it matter *which* names you pick, chosen
  blindfolded? Read together they bracket Malkiel's quip.
- **[Study 97 — Balancing-Act](../../97-balancing-act/)**: the 60/40 portfolio teardown —
  same desk machinery (excess-vs-excess races, block-bootstrap CIs) on an allocation rule.
