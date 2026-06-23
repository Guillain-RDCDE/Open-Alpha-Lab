# References & literature map — Study 363 (Post-Earnings-Announcement Drift)

## The claim under test

- **The folklore.** "After a company beats (or misses) on earnings, the stock keeps **drifting
  the same way** for weeks — so buy the beats, short the misses, and ride the drift." It is the
  retail/technician version of one of the most-cited anomalies in academic finance.
- **The seminal evidence.** Ray Ball & Philip Brown, *An Empirical Evaluation of Accounting
  Income Numbers* (1968, Journal of Accounting Research) first documented that prices keep
  adjusting *after* an earnings announcement. Victor Bernard & Jacob Thomas, *Post-Earnings-
  Announcement Drift: Delayed Price Response or Risk Premium?* (1989, JAR) and *Evidence that
  Stock Prices Do Not Fully Reflect the Implications of Current Earnings for Future Earnings*
  (1990, Journal of Accounting & Economics) established PEAD as a robust, persistent effect:
  stocks in the top **standardized unexpected earnings (SUE)** decile outperform the bottom
  decile for ~60 trading days after the print.
- **Why it is the rare survivor.** Eugene Fama (1998, *Market efficiency, long-term returns, and
  behavioral finance*, JFE) called PEAD the "granddaddy of anomalies" — the one most resistant
  to the usual methodological objections. Surveys: Richardson, Tuna & Wysocki (2010,
  *Accounting anomalies and fundamental analysis*, JAE) and Kishore et al. on the SUE factor.

## Surprise proxies — what we measure, and why two of them

- **Reported EPS surprise (headline proxy).** We use the vendor's `Surprise(%)` =
  (reported EPS − consensus estimate) / |estimate|, the closest free analogue to academic
  **SUE**. This is the *fundamental* surprise, orthogonal to the price reaction.
- **Post-announcement price gap (myth-check proxy).** The one-day reaction return the session
  after the print — the "surprise the tape reveals," and exactly what a chart-watcher would
  sort on. Brandt, Kishore, Santa-Clara & Venkatachalam (2008, *Earnings announcements are full
  of surprises*) and the SUE-vs-return literature show the *price* surprise and the *earnings*
  surprise are related but distinct; our myth-check confirms only the latter drifts.
- **Timing / no look-ahead.** Most large-caps report after the close, so the first full session
  that prices the news is the next day. We observe the gap at that reaction-session close, then
  enter **one day later** and hold — so the drift we measure is strictly *after* the reaction is
  public (the standard event-study convention; see Bernard & Thomas 1989).

## Why the effect is small on a large-cap basket

- **Limits to arbitrage.** Chordia, Goyal, Sadka, Sadka & Shivakumar (2009, *Liquidity and the
  post-earnings-announcement drift*, FAJ) show PEAD concentrates in **small, illiquid** names
  and shrinks among liquid large-caps once trading costs are charged. Our universe is 30 of the
  most liquid US large-caps **by construction**, so a thin, cost-sensitive drift is exactly what
  theory predicts — the effect is real but at the conservative end of its range.
- **Costs and turnover.** Each event is a fresh round trip on both legs; we apply one-way costs
  × turnover (4 legs) and short-leg borrow. Frazzini, Israel & Moskowitz (2018, *Trading costs*)
  on the gap between paper and net anomaly returns motivates the net-vs-gross discipline.

## Why a high *t* still needs a placebo + clustering check

- **Welch / one-sample t** (Welch, 1947, *The generalization of "Student's" problem*) for the
  long-short mean against zero. Earnings cluster in **seasons**, so naive *t*-stats overstate
  significance; we add a **label-shuffle placebo** (Fisher's randomization logic; Efron &
  Tibshirani, *An Introduction to the Bootstrap*, 1993) and a **within-quarter block placebo**
  that respects the clustering — both still reject at 20 days.
- **Selection on a famous rule.** Harvey, Liu & Zhu (2016, *…and the Cross-Section of Expected
  Returns*, RFS) and McLean & Pontiff (2016, *Does academic research destroy stock return
  predictability?*, JF) caution that documented anomalies decay post-publication; PEAD has
  decayed but not vanished — consistent with our thin-but-significant large-cap result.

## Method lineage (the desk's shared engine)

- **Quintile long-short + one-sample t.** [`strategy.long_short_drift`](../pead_drift/strategy.py)
  and [`strategy.ttest_vs_zero`](../pead_drift/strategy.py) — top-minus-bottom surprise-quintile
  drift and its significance against zero.
- **Label-shuffle placebo.** [`strategy.placebo_pvalue`](../pead_drift/strategy.py) — 20,000
  random re-sorts of the same drifts; the honest small-effect null.
- **Deterministic synthetic control.** [`data.synthetic_pead`](../pead_drift/data.py) plants a
  known post-event drift proportional to the surprise; with the edge set to zero the inference
  must NOT manufacture significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted closes for a fixed 30-name large-cap basket + per-name
  `Ticker.get_earnings_dates` (reported EPS surprise), 2005-01-11 → 2026-06-10, cached under
  `_cache/pead_prices.csv` and `_cache/pead_events.csv`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- The **research-method demos** (data-mining-roulette, multiple-testing, look-ahead) frame why a
  *t* alone is not enough — PEAD is the counter-example that *does* clear the bar after the
  placebo and clustering checks.
