# References & literature map — Study 357 (the Coffee-Can portfolio)

## The claim under test

- **The source.** Robert G. Kirby, *The Coffee Can Portfolio*, **Journal of Portfolio
  Management**, Fall 1984. Kirby recounts a client whose husband had bought every stock Kirby
  recommended but ignored every *sell* — the untouched "coffee can" account, with one holding
  ballooned to a fortune, dwarfed the actively-managed ones. The prescription: buy quality
  names, hold for ~10 years, **never trade**.
- **Modern restatements.** Charlie Munger's "sit-on-your-ass investing" (the big money is in
  the *waiting*); the Bogle/index-fund case against turnover and fees; Saurabh Mukherjea,
  *Coffee Can Investing* (2018) for the Indian equity restatement. The popular version fuses
  three promises: it **beats the index**, it does so at **near-zero cost**, and it requires
  **no skill or effort.** We test each.

## Why the apparent edge is partly an artefact

- **Survivorship bias.** Brown, Goetzmann, Ibbotson & Ross (1992), *Survivorship Bias in
  Performance Studies* (Review of Financial Studies); Elton, Gruber & Blake (1996),
  *Survivorship Bias and Mutual Fund Performance*. Selecting names known *ex post* to have
  survived (or won) inflates measured returns by an amount that does **not** vanish with sample
  size — the bias is in the *selection*, not the estimator's variance. A coffee can assembled
  from today's famous names is the textbook case: the bankruptcies and de-listings that a real
  1984 investor faced are silently excluded.
- **Order-statistics / look-ahead.** Choosing the top-*k* names by *terminal* value picks the
  right tail of the cross-section by construction; the conditional mean
  `E[r | top-k terminal]` is biased up even when every firm shares the same true drift. Our
  synthetic control plants exactly that null (identical drift) and measures the manufactured gap.
- **Low-turnover / cost premium (the honest part).** Carhart (1997), *On Persistence in Mutual
  Fund Performance* — high turnover predicts lower net returns; the cost wedge is real but, at a
  low rebalance frequency, small. Arnott, Hsu & Moore (2005) and the broader "fundamental
  indexing" line quantify how rebalancing/trading frictions erode net performance — the tailwind
  a never-trade can genuinely captures.

## Why "buy and hold" wins on *risk*, not necessarily *return*

- **Low-volatility / defensive equity.** Frazzini & Pedersen (2014), *Betting Against Beta*;
  Baker, Bradley & Wurgler (2011), *Benchmarks as Limits to Arbitrage*. A buy-and-hold basket
  of mature, dividend-paying large-caps is a low-beta, defensive tilt: it should show lower
  volatility and shallower drawdowns than the cap-weighted index, with a comparable Sharpe — a
  *risk* edge, not an alpha. This is exactly the MIXED pattern we find.
- **Weight drift / convexity of buy-and-hold.** A never-rebalanced basket lets winners run, so
  terminal wealth `W_T = Σ w_{i,0} Π_t(1+r_{i,t})` is convex in each path and concentrates into
  the survivors automatically — the same mechanism that makes hindsight selection so flattering.

## Method lineage (the desk's shared engine)

- **Paired HAC inference.** The Signal axis is the mean monthly *excess* of the can over the
  index, tested with a Newey-West (HAC) standard error (`strategy.paired_t`, `hac_lag=6`).
  Newey & West (1987), *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix*. `REAL` requires |t| ≥ 2 on the real tape.
- **Deterministic synthetic control.** A fixed-seed panel generator with an explicit death
  process ([`data.synthetic_panel`](../coffee_can/data.py)) and a look-ahead-vs-ex-ante
  selector pair ([`data.famous_names`](../coffee_can/data.py) /
  [`data.honest_names`](../coffee_can/data.py)) — the offline positive control runs with no
  network and plants a *known* (zero) cross-sectional edge so the harness's survivorship
  measurement is calibrated.
- **Cost on NAV.** Turnover defined one-way × NAV (the L1 weight change at each reset),
  charged in basis points ([`strategy.rebalanced`](../coffee_can/strategy.py)) — the house
  cost convention.

## Data sources used here

- **yfinance** (public Yahoo Finance endpoint, no key): monthly **total-return** closes
  (`auto_adjust=True` — splits *and* dividends), 2005-01 → 2026-05, for the basket
  KO/JNJ/PG/PEP/WMT/MMM/MO/XOM/IBM/GE and the benchmark SPY. Cached under
  `_cache/coffee_can_monthly.csv`.
- **Caveat (named on the Signal axis).** The basket is itself a *survivorship sample* — ten
  large-caps that are still household names in 2026. The real-tape result is therefore an
  *upper bound* on what an honest 2005 investor would have captured; the synthetic control
  measures how large that inflation can be. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 151 — Stocks-for-the-long-run](../../151-stocks-for-long-run/)**: the dead-names
  problem at index scale — every "equities always win over decades" chart is drawn on the firms
  that didn't go bankrupt. Same bias, larger canvas.
- **[Study 144 — Permanent-portfolio](../../144-permanent-portfolio/)** and the lazy-portfolio
  family: low-turnover buy-and-hold whose appeal is *risk-adjusted* smoothness, not raw
  out-performance — the same MIXED signature.
