# References & literature map — Study 361 (Zweig Breadth Thrust)

## The claim under test

- **The rule (Marty Zweig).** Martin Zweig, *Winning on Wall Street* (1986, rev. 1994). Zweig
  defines the **Breadth Thrust**: take the 10-day exponential moving average of *NYSE advancing
  issues / (advancing + declining issues)*; a thrust occurs when this EMA rises **from below
  0.40 to above 0.615 within ~10 trading days**. Zweig reports that every such thrust since 1945
  was followed by a powerful bull move — a famously rare, famously "never wrong" buy signal.
- **The folklore.** The signal is repeated across market-technician lore and financial media as
  the rare "all-clear" with a perfect or near-perfect record (commonly cited as ~a dozen
  occurrences in 70+ years, each followed by strong gains, e.g. the widely-noted 2009, 2011,
  2015, 2018, 2019, 2020 fires). The "never wrong" reputation is precisely what invites a
  sample-size audit: ~a dozen events is far too few to establish an edge.

## Why true breadth is not on yfinance — and what we do instead

- **NYSE advance/decline data.** The canonical inputs (daily NYSE advancing/declining issues,
  ticker symbols `$ADV`/`$DECL`/`$ADD` on some terminals) are **not** available through the free
  yfinance endpoint, which serves per-ticker OHLCV only. We therefore **construct a transparent
  proxy**: the daily advance ratio across a fixed, long-listed 40-name US large-cap basket
  (advancers / (advancers + decliners)). This is a *narrower, noisier* breadth gauge than the
  ~3,000-issue NYSE — and we say so on the Signal axis. The proxy is a methodological choice,
  not a fabrication: every input is a public adjusted close.
- **Breadth indicators generally.** Geoffrey H. Moore and the NBER tradition on diffusion
  indexes; Richard Russell's and Norman Fosback's writing on advance/decline breadth (Fosback,
  *Stock Market Logic*, 1976) — breadth as the share of the market participating in a move. The
  advance ratio is the simplest diffusion index; the EMA is Zweig's smoother.

## Why ~a dozen events cannot be an edge — the statistics

- **Small-sample inference / power.** With *k* ≈ 12–22 events, the standard error of a
  conditional-mean estimate is large; an excess return of a few percent over a ~10–20% base
  cannot be distinguished from luck. We test the conditional mean against the unconditional mean
  with a **Welch two-sample t** (Welch, 1947, *The generalization of "Student's" problem*) and,
  because *k* is tiny, with a **placebo / randomization test** — the share of random same-size
  date draws whose mean forward return beats the thrust set (Fisher's randomization logic;
  Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993).
- **Base rates and the win-rate illusion.** US equities rise in the large majority of rolling
  12-month windows, so a high post-signal win-rate is *expected under the null*. The right
  comparison is the **excess** over the unconditional base rate, not the raw win-rate — the
  classic base-rate fallacy (Kahneman & Tversky, 1973, *On the psychology of prediction*).
- **Multiple testing / selection on a famous rule.** Rules that survive into folklore are
  selected on their in-sample record; Harvey, Liu & Zhu (2016), *…and the Cross-Section of
  Expected Returns* (Review of Financial Studies) and Bailey & López de Prado (2014),
  *The Deflated Sharpe Ratio*, formalise why a single "never wrong" rule discovered ex-post needs
  a far higher bar than a naive t-stat.

## Method lineage (the desk's shared engine)

- **Welch t + placebo p-value.** [`strategy.welch_t`](../zweig_breadth_thrust/strategy.py) and
  [`strategy.placebo_pvalue`](../zweig_breadth_thrust/strategy.py) — the Signal-axis tests:
  conditional vs unconditional forward returns, and a 20,000-draw randomization null sized to
  the event count.
- **Deterministic synthetic control.**
  [`data.synthetic_breadth`](../zweig_breadth_thrust/data.py) injects a known number of thrusts
  and (optionally) a known forward edge; the offline core runs with no network. The control
  confirms the detector is faithful *and* that ~a dozen events cannot reach significance unless
  the planted edge is implausibly large.
- **Forward-return measurement with execution lag.**
  [`strategy.event_returns`](../zweig_breadth_thrust/strategy.py) enters one day after the signal
  (no look-ahead) and holds a fixed horizon; costs applied in
  [`strategy.net_of_costs`](../zweig_breadth_thrust/strategy.py).

## Data sources used here

- **yfinance** daily adjusted closes for SPY + a fixed 40-name large-cap basket, 1995-01-04 →
  2026-06-22, cached under `_cache/basket_prices.csv`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 167 — Hindenburg-Omen](../../167-hindenburg-omen/)**: the mirror image. A rare
  breadth-based signal said to "always" precede *crashes*; the Zweig thrust is the same rare
  breadth pattern said to "always" precede *rallies*. Both are sample-size illusions — a dozen
  events cannot establish either a curse or a blessing.
- **[Study 168 — Advance-Decline](../../168-advance-decline/)**: the breadth raw material —
  whether the advance/decline line itself carries information distinct from price.
