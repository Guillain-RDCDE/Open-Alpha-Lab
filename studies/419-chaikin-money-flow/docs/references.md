# References & literature map — Study 419 (Chaikin Money Flow)

## The claim under test

- **The folk recipe.** Marc Chaikin built the **Accumulation/Distribution line** in the
  1970s and **Chaikin Money Flow (CMF)** on top of it. CMF reads where each bar's close
  sits inside its high–low range (the *Money Flow Multiplier*), weights it by volume, and
  sums over a trailing window (default 20–21 days) divided by total volume, giving a value
  in [−1, +1]. The trading lore, repeated across StockCharts, Investopedia, TradingView
  and most retail platforms: **CMF > 0 is "accumulation" and precedes a rise; CMF < 0 is
  "distribution" and precedes a fall — money flow leads price.** We steelman this as:
  *the sign/level of CMF on daily equity bars carries information about the next return that
  beats a volume-blind trend filter, net of costs.*

## Why the steelman is *almost* coherent — the real effects it leans on

- **Volume genuinely co-moves with returns.** Karpoff (1987), *"The Relation Between Price
  Changes and Trading Volume: A Survey"* (J. Financial & Quantitative Analysis), documents
  a robust positive price-change/volume correlation. Heavy-volume up-days are real — the
  open question CMF stakes is whether they *forecast* the next move or merely *describe*
  the current one.
- **Information-based trading and order flow.** Easley, Kiefer, O'Hara & Paperman (1996),
  *"Liquidity, Information, and Infrequently Traded Stocks"* (J. Finance), and the broader
  PIN literature show informed order flow can lead price intraday. CMF is a crude,
  end-of-day proxy for that idea — but daily close-location is a very noisy read on flow.
- **Close location and the "smart-money" intuition.** Chaikin's premise (close near the
  high = buyers in control) echoes the candlestick body/wick lore. Whether that intrabar
  position predicts the *next* bar is exactly what this study measures.

## The failure mode exposed

- **CMF is contemporaneous, not leading.** The Money Flow Multiplier is computed from the
  *same* bar whose return it is supposed to anticipate; on a broad index the close's
  within-bar location carries no forecast of the next day's return (HAC *t* ≈ 0,
  permutation *p* ≈ 1 here). This is the "indicator confirms what already happened"
  failure mode common to lagging technical overlays.
- **A volume-blind trend filter wins.** The SMA(50/200) and MACD filters — which never
  look at volume — both beat CMF on net excess Sharpe. Park & Irwin (2007), *"What Do We
  Know About the Profitability of Technical Analysis?"* (J. Economic Surveys), review how
  most technical rules' apparent edge fails to survive honest benchmarking.
- **Data-snooping / out-of-sample fragility.** Sullivan, Timmermann & White (1999),
  *"Data-Snooping, Technical Trading Rule Performance, and the Bootstrap"* (J. Finance),
  and Brock, Lakonishok & LeBaron (1992), *"Simple Technical Trading Rules and the
  Stochastic Properties of Stock Returns"* (J. Finance), document how technical-rule
  performance shrinks once selection and the bootstrap are accounted for — and how the
  right control is the *simpler* rule, not a strawman.
- **Weak-form efficiency.** Fama (1970), *"Efficient Capital Markets"* (J. Finance): for
  large, liquid US equities at the daily horizon, public price/volume history should carry
  little exploitable forecast — consistent with the `NONE` here.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *"A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix"* (Econometrica) —
  implemented in [`strategy.hac_tstat`](../chaikin_money_flow/strategy.py).
- **Permutation / placebo inference.** The block-permutation timing test
  ([`strategy.block_permutation_pvalue`](../chaikin_money_flow/strategy.py)) follows the
  randomisation-test logic of Good (2005), *Permutation, Parametric, and Bootstrap Tests
  of Hypotheses*, preserving the marginal exposure while destroying the signal–return alignment.
- **Reproducibility stamp.** As-of freeze + content fingerprint each headline run carries
  (see [`docs/results.md`](results.md)); mirrors the desk's `quantlab/repro.py` discipline.

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`), auto-adjusted OHLCV, SPY headline tape
  2000-01-03 → 2026-06-23 plus a five-name panel (QQQ, AAPL, MSFT, XLE, GLD). The offline
  reproducible core and the synthetic positive control run on the deterministic
  [`data.synthetic_panel`](../chaikin_money_flow/data.py) generator, never the network.
  Each headline is pinned with an as-of date and a per-tape content fingerprint.

## Related desk studies

- **[Study 418 — Money Flow Index](../../418-money-flow-index/)**: the volume-weighted
  RSI — the closest cousin, same harness (long/flat race vs RSI / buy-and-hold), same
  "does volume add information?" question.
- **[Study 178 — CCI](../../178-cci/)**: Lambert's Commodity Channel Index, another
  normalised oscillator overbought/oversold rule, also found to carry no daily edge.
- **[Study 104 — Bollinger-Reversion](../../104-bollinger-reversion/)**: a mean-reversion
  band rule whose apparent edge dissolves into market drift once benchmarked.
- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: the 50/200 golden cross — the same
  trend-filter benchmark CMF is raced against here.
