# References & literature map — Study 423 (Force Index)

## The claim under test

- **The folk recipe.** Alexander Elder introduced the Force Index in *Trading for a Living:
  Psychology, Trading Tactics, Money Management* (Wiley, 1993), and elaborated it in
  *Come Into My Trading Room* (Wiley, 2002). The raw Force Index is
  `(Close − Close_prev) × Volume`, smoothed with an exponential moving average — Elder uses
  **FI(2)** for short-term entries and **FI(13)** for the intermediate trend. The rule sold
  to retail: *"When the 13-day Force Index crosses above zero the bulls are in control — buy;
  when it crosses below zero the bears have taken over — exit or short. It flags reversals
  before the crowd."* We steelman this as: *the FI(13) zero-cross timing rule, applied with a
  one-day lag and net of costs, beats simply holding the asset.*

## Why the steelman is *almost* coherent — the real effects it leans on

- **Price-volume confirmation.** The intuition that a move on heavy volume is "more real"
  has a genuine empirical basis. Karpoff (1987), *"The Relation Between Price Changes and
  Trading Volume: A Survey"* (Journal of Financial and Quantitative Analysis), documents the
  positive price-change/volume correlation. Gervais, Kaniel & Mingelgrin (2001),
  *"The High-Volume Return Premium"* (Journal of Finance), show high-volume days carry
  information. Force Index packages this folklore into one number — but packaging an effect
  is not the same as extracting a *tradable directional* signal from it.
- **Trend / momentum at intermediate horizons.** Moskowitz, Ooi & Pedersen (2012),
  *"Time Series Momentum"* (Journal of Financial Economics), document that a long/flat trend
  filter can add value across asset classes — which is exactly why our **SMA(50/200)**
  benchmark beats buy-and-hold here. The question is whether FI's volume twist *improves* on
  a price-only trend filter; on this tape it strictly degrades it.

## The failure mode exposed

- **Out of the market during the up-legs.** US equity ETFs trend structurally upward, so any
  rule that sits in cash ~40% of the time (FI(13) is invested only 58% of days) sacrifices
  the equity premium it foregoes. The FI zero-cross is a *lagging, noisy* trend proxy whose
  flips cluster around the very advances that pay — so it underperforms buy-and-hold on every
  tape, significantly on five of six.
- **The volume multiplication adds noise, not signal.** Because raw FI is price-change ×
  volume, volume spikes (often around news, gaps, index rebalances) inject large transient
  swings that flip the zero-cross without forecasting direction. Our SMA(50/200) benchmark —
  identical machinery, *no volume term* — beats both FI and buy-and-hold, isolating the
  volume twist as the source of the degradation.
- **Data-snooping on technical rules.** Brock, Lakonishok & LeBaron (1992), *"Simple
  Technical Trading Rules and the Stochastic Properties of Stock Returns"* (Journal of
  Finance), and Sullivan, Timmermann & White (1999), *"Data-Snooping, Technical Trading Rule
  Performance, and the Bootstrap"* (Journal of Finance), show how much apparent technical
  edge evaporates out of sample. Park & Irwin (2007), *"What Do We Know About the
  Profitability of Technical Analysis?"* (Journal of Economic Surveys), survey the mixed-to-
  negative record of volume-based oscillators on equities — consistent with the result here.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West *t*-stat.** Newey & West (1987), *"A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix"* (Econometrica) —
  implemented in [`force_index/strategy.py`](../force_index/strategy.py) (`hac_tstat`).
- **Excess-vs-excess Sharpe race.** Both legs reduced to excess-of-cash before comparison,
  with the test on the daily return difference (`excess_vs_excess`), per the desk's
  house rule that a part-time-in-cash rule must be judged on like-for-like excess returns.
- **Permutation / placebo inference.** The sign-permutation of the position vector
  (`permutation_pvalue`) is a label-shuffle null in the spirit of White (2000)'s Reality
  Check — it preserves the exposure profile and asks whether the *timing* alignment carries
  information.
- **Reproducibility stamp.** Each headline run carries an explicit as-of date (partial bar
  dropped) and a per-tape content fingerprint (`data.fingerprint`); see
  [`docs/results.md`](results.md).

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`, `auto_adjust=True` → total-return-adjusted
  close), full histories of six liquid ETFs (SPY, QQQ, DIA, IWM, XLE, GLD) to 2026-06-23.
  The offline reproducible core, the positive control, and any test run on the deterministic
  [`data.synthetic_panel`](../force_index/data.py) generator, never the network.

## Related desk studies

- **[Study 178 — CCI](../../178-cci/)**: Lambert's Commodity Channel Index — another
  textbook oscillator, same "does a technical rule beat a coin / buy-and-hold?" question.
- **[Study 104 — Bollinger-Reversion](../../104-bollinger-reversion/)**: Bollinger Band
  mean-reversion; the trend-vs-reversion framing recurs here.
- **[Study 106 — Supertrend](../../106-supertrend/)**: a trend-following technical timing
  rule on the same infrastructure — the family Force Index aspires to but underperforms.
- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: the 50/200 golden cross — the price-only
  trend filter that, in this study, *beats* the volume-laden Force Index.
