# References & literature map — Study 424 (Ease of Movement)

## The claim under test

- **The folk recipe.** Richard W. Arms Jr. introduced the Ease of Movement (EMV/EOM)
  oscillator in the late 1970s as a companion to his Equivolume charting method
  (*Volume Cycles in the Stock Market*, Dow Jones-Irwin, 1983; and *Profits in Volume:
  Equivolume Charting*, 1971). EOM combines the bar's mid-price move with its
  volume-scaled range into a single "how easily did price move?" number: a large positive
  reading means price rose a lot on little volume (an *effortless advance*), a large
  negative reading the reverse. The rule sold on modern charting sites (StockCharts,
  Investopedia, TradingView): **EOM crossing above zero is a buy, crossing below zero is a
  sell** — "trade with the path of least resistance." We steelman this as: *the EOM-zero
  timing rule, net of costs, delivers a Sharpe edge over buy-and-hold that exceeds what a
  plain price-only moving-average cross already provides.*

## Why the steelman is *almost* coherent — the real effect it leans on

- **Volume confirms trend.** The idea that low-volume drifts continue and high-volume
  reversals exhaust is folk wisdom with a kernel: Karpoff (1987), *"The Relation Between
  Price Changes and Trading Volume: A Survey"* (Journal of Financial and Quantitative
  Analysis), documents a robust price-volume relationship, and Campbell, Grossman & Wang
  (1993), *"Trading Volume and Serial Correlation in Stock Returns"* (Quarterly Journal of
  Economics), show that return autocorrelation is conditional on volume — exactly the kind
  of structure an EOM-style indicator could in principle exploit.
- **Time-series momentum is real.** Moskowitz, Ooi & Pedersen (2012), *"Time Series
  Momentum"* (Journal of Financial Economics), establish that trend persistence at the
  index level is a genuine, pervasive premium. EOM > 0 is, at heart, a slow trend filter,
  so the standalone signal it produces is real *because trend is real* — which is precisely
  why it does not constitute a separate edge.
- **Moving-average timing reduces drawdown.** Faber (2007), *"A Quantitative Approach to
  Tactical Asset Allocation"* (Journal of Wealth Management), shows that a simple SMA
  timing overlay roughly matches buy-and-hold returns at much lower volatility/drawdown —
  the *de-risked beta* result this study reproduces for EOM, and the benchmark EOM must
  beat to earn an alpha claim.

## The failure mode exposed

- **A trend signal in a volume costume.** This study finds EOM's standalone net Sharpe is
  real (*t* = 3.18) but **statistically indistinguishable from an SMA(50/200) cross**
  (difference *t* = −0.61) and it **loses the race against buy-and-hold** (difference
  *t* = −1.71). The volume weighting adds nothing measurable. This is the textbook
  data-snooping / redundant-indicator trap documented by Sullivan, Timmermann & White
  (1999), *"Data-Snooping, Technical Trading Rule Performance, and the Bootstrap"*
  (Journal of Finance), and by Park & Irwin (2007), *"What Do We Know About the
  Profitability of Technical Analysis?"* (Journal of Economic Surveys): once you control
  for the simple benchmark, the dressed-up rule's marginal value vanishes.
- **De-risked beta is not alpha.** Brock, Lakonishok & LeBaron (1992), *"Simple Technical
  Trading Rules and the Stochastic Properties of Stock Returns"* (Journal of Finance),
  warned that a timing rule's lower volatility is mostly a function of being in cash part
  of the time — comparing it to fully-invested buy-and-hold on raw return flatters it. We
  guard against this by running the race **excess-of-cash, on Sharpe and on a difference
  test**, where the flattery disappears.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *"A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix"* (Econometrica) —
  implemented in [`strategy._hac_tstat`](../ease_of_movement/strategy.py).
- **Permutation / sign-shuffle placebo.** The episode-sign-shuffle null follows the
  randomisation-test logic of Masters (2018), *Permutation and Randomization Tests for
  Trading System Development*, applied here to the realised holding episodes.
- **Reproducibility stamp.** As-of freeze + per-tape content fingerprint, mirroring the
  desk-wide `quantlab/repro.py` convention; see [`docs/results.md`](results.md).

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`, `auto_adjust=True`), 20-year window across
  six liquid ETF tapes (SPY, QQQ, IWM, DIA, EFA, EEM). The offline reproducible core and
  the synthetic positive control run on the deterministic
  [`data.synthetic_panel`](../ease_of_movement/data.py) generator, never the network. Each
  headline is pinned to as-of 2026-05-31 (last complete month) with per-tape fingerprints.

## Related desk studies

- **[Study 178 — CCI](../../178-cci/)**: another normalised oscillator turned into a timing
  rule and pinned against a fair control — same "does the indicator add anything?" question.
- **[Study 104 — Bollinger-Reversion](../../104-bollinger-reversion/)**: Bollinger Band mean
  reversion, the reversion counterpart to EOM's trend framing, same honest engine.
- **[Study 106 — Supertrend](../../106-supertrend/)**: a trend-following technical rule —
  the closest family to EOM (a slow trend filter), same treatment.
- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: the 50/200 golden cross — the very
  SMA benchmark EOM is raced against here, on the lagging-indicator question.
- **[Study 401 — Signal-Stacking](../../401-signal-stacking/)**: the desk's method demo on
  why a redundant signal adds no edge — the structural point this EOM teardown instantiates.
