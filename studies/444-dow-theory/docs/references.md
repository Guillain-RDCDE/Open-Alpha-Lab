# References & literature map — Study 444 (Dow Theory)

## The claim under test

- **The folklore.** "A bull market is only real when the **Industrials** and the **Transports**
  *confirm* each other — both averages must make new **higher highs**. When one average makes a
  new high the other fails to match (a *non-confirmation*), the trend is suspect; when both make
  new **lower lows**, the primary trend has reversed. Trade with the confirmed primary trend;
  stand aside when the averages disagree." This is the oldest piece of US technical-analysis
  doctrine and the conceptual ancestor of the Dow Jones averages themselves.
- **The origin.** Charles H. Dow stated the ideas piecemeal in *Wall Street Journal* editorials
  (1899–1902). They were systematised posthumously by **William Peter Hamilton**, *The Stock
  Market Barometer* (1922), and codified into the canonical "tenets" by **Robert Rhea**, *The
  Dow Theory* (1932): the market has three trends (primary, secondary, minor); the averages must
  *confirm* one another; volume confirms the trend; a trend persists until a clear reversal.
- **The confirmation tenet.** The "two averages must confirm" rule is the falsifiable core and
  the one we mechanise: a primary-trend signal requires **both** the Industrials and the
  Transports to break to new highs (bull) or new lows (bear) — agreement between the two is the
  whole point.

## Why a mechanical encoding (and its limits)

- **Dow Theory is partly subjective** (identifying "the trend," "secondary reactions," and
  significant highs/lows is a matter of judgement). We test the **tightest mechanical rule a
  proponent would accept**: "both averages at/above their trailing high" as the confirmation,
  latched as a primary-trend regime until "both at/below their trailing low." This is the
  objective, falsifiable skeleton of the confirmation tenet — if even the steelmanned mechanical
  version carries no edge, the discretionary version inherits the burden of proof.
- **The academic record.** Alfred Cowles III, *Can Stock Market Forecasters Forecast?* (1933,
  *Econometrica*) scored Hamilton's published Dow-Theory forecasts and found they **underperformed**
  a buy-and-hold benchmark. Stephen Brown, William Goetzmann & Alok Kumar, *The Dow Theory:
  William Peter Hamilton's Track Record Reconsidered* (1998, *Journal of Finance*) re-graded
  Hamilton with a neural-net replication and found his market-timing had **positive risk-adjusted
  but lower absolute** returns than buy-and-hold — i.e. the value, if any, was **drawdown
  reduction from being in cash**, exactly our finding. Our mechanical confirmation rule reproduces
  that pattern and shows it does not require the *confirmation* at all.

## Why the active spread, the placebo, and the content test

- **Excess-vs-excess race.** A timing rule that sits in cash part-time must be compared
  excess-of-cash to excess-of-cash, or a raw-Sharpe vs excess-Sharpe mismatch manufactures a
  verdict (a desk house rule). Cash earns 0 here, so both legs are on the same clock.
- **HAC inference.** Newey & West (1987, *A Simple, Positive Semi-Definite, Heteroskedasticity
  and Autocorrelation Consistent Covariance Matrix*, *Econometrica*) for the one-sample *t* on
  the daily active spread — daily returns are autocorrelated and a naive *t* overstates
  significance.
- **The cash-timing confound.** Any rule that holds cash a third of the time lowers drawdown and
  can match Sharpe in a sample that contained two crashes — without the signal carrying any
  information. We separate the two with (a) a **content test** (confirmed-day vs non-confirmed-day
  forward returns, Welch *t*; Welch 1947) that strips the cash-drag, and (b) a **random-regime
  placebo** matched to the real regime's on-fraction and run length (a two-state Markov chain) —
  the honest "is it the Transports, or just being in cash?" null.

## Method lineage (the desk's shared engine)

- **Confirmation regime + latched primary trend.**
  [`strategy.confirmation_flag`](../dow_theory/strategy.py) /
  [`strategy.latched_regime`](../dow_theory/strategy.py) — both-averages trailing-high/low rule.
- **HAC one-sample t + Welch content test.** [`strategy.hac_t`](../dow_theory/strategy.py),
  [`strategy.confirmed_vs_not`](../dow_theory/strategy.py).
- **Random-regime placebo.** [`strategy.placebo_pvalue`](../dow_theory/strategy.py) — matched
  on-fraction and persistence; p = P[random Sharpe ≥ real].
- **Deterministic synthetic control.** [`data.synthetic_panel`](../dow_theory/data.py) plants a
  confirmation edge proportional to a knob; with the edge set to zero the content test must NOT
  manufacture significance — the offline core runs with no network.

## Data sources used here

- **yfinance** auto-adjusted daily closes for **DIA** (Industrials) and **IYT** (Transports),
  2004-01-02 → 2026-05-29, cached under `_cache/dow_theory_etf.parquet`; and the price-only
  index levels **^DJI / ^DJT**, 1992 → 2026, cached under `_cache/dow_theory_idx.parquet`. All
  headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- The **moving-average / trend-timing** teardowns (e.g. [Faber timing](../110-faber-timing),
  [death-cross](../91-death-cross), [Supertrend](../106-supertrend)) share the lesson that most
  trend-timing's apparent benefit is **drawdown reduction from cash exposure**, not signal.
- The **research-method demos** ([data-mining-roulette](../343-data-mining-roulette),
  [multiple-testing](../346-multiple-testing)) frame why a *t* alone is not enough and why a
  matched-persistence placebo is the right null for a part-time-in-cash timing rule.
