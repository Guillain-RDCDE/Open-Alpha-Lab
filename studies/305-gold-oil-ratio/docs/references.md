# References & literature map — Study 305 (Gold-Oil-Ratio)

## The claim under test

- **The "gold/oil ratio is a macro regime gauge" narrative.** The folk pitch, a sibling of
  the copper/gold "Dr. Copper" story, holds that the gold-to-oil ratio (ounces of oil per
  ounce of gold, or simply GLD/USO) reads the business cycle: a **high** ratio (gold
  expensive vs oil) signals risk-off / contraction / deflation fear, while a **low** ratio
  signals reflation and growth. Practitioners (e.g. macro commentary at Charlie Bilello /
  Compound Capital Advisors, and the "barrels of oil per ounce of gold" charts that
  periodically go viral on finance Twitter) suggest you can use it to *time* equity exposure:
  go defensive when the ratio spikes. We take the strongest tradable version — a binary
  risk-on/risk-off SPY-vs-cash switch driven by the standardised gold/oil-ratio deviation —
  and ask whether it beats buy-and-hold on a risk-adjusted, excess-of-cash basis.

## The real macro relationships the claim leans on

- **Oil as a growth/inflation barometer.** Hamilton (1983), *Oil and the Macroeconomy Since
  World War II* (Journal of Political Economy) — oil-price shocks precede recessions; Kilian
  (2009), *Not All Oil Price Shocks Are Alike* (American Economic Review) — disentangling
  demand- vs supply-driven oil moves, which is exactly why a raw oil level (or ratio) is a
  noisy growth signal.
- **Gold as a risk-off / real-rate asset.** Baur & Lucey (2010), *Is Gold a Hedge or a Safe
  Haven?* (Financial Review) — gold's safe-haven behaviour in equity sell-offs, the half of
  the ratio that rises in a crisis.
- **Commodity ratios and the cycle.** The copper/gold "Dr. Copper" literature (Gundlach's
  popularisations; and on this desk, **Study 85 — Dr-Copper**) is the direct intellectual
  parent: a relative-price of a growth-sensitive commodity to a safe-haven one as a cycle
  gauge. The gold/oil ratio is the same idea with oil (demand-and-supply driven) standing in
  for copper.

## Why timing claims usually fail — the predictability literature

- **Out-of-sample return predictability is hard.** Goyal & Welch (2008), *A Comprehensive
  Look at the Empirical Performance of Equity Premium Prediction* (Review of Financial
  Studies) — most in-sample equity-return predictors fail out-of-sample against the simple
  historical mean. A gold/oil timing rule is exactly the kind of single-variable predictor
  that this critique targets.
- **Market timing vs buy-and-hold.** Sharpe (1975), *Likely Gains from Market Timing*
  (Financial Analysts Journal) — the high bar a timing rule must clear to beat staying
  invested, given that it spends time in cash and pays costs. The right yardstick is
  risk-adjusted and excess-of-cash, not raw return.
- **Drawdown reduction ≠ alpha.** A switch that sits in cash part-time is a lower-beta book;
  comparing its raw Sharpe to a fully-invested book's *excess* Sharpe is a rigged race (the
  desk's house rule). We compare excess-of-cash Sharpe to excess-of-cash Sharpe, and we add a
  random-timing control with the identical cash fraction to isolate *information* from *beta*.

## How this study is distinct from its desk cousins (dedup)

- **Study 85 — Dr-Copper** runs *predictive regressions* of the copper/gold ratio change on
  forward equity returns and yields (does the ratio *forecast*?), and separates the
  contemporaneous from the predictive link. This study (305) does **not** regress; it builds a
  *tradable binary timing switch* and races it, excess-vs-excess, against buy-and-hold and a
  random-timing control. Different commodity pair (gold/oil vs copper/gold), different
  apparatus (timing backtest vs forecasting regression), different question (can you *trade*
  the regime call vs does the ratio *forecast*).
- **Study 113 — Gold-Silver-Ratio** trades a *z-score pairs / mean-reversion* between two
  *metals* (long one, short the other, betting the ratio snaps back), and tests stationarity
  (ADF, cointegration, OU half-life). This study (305) is **not** a pairs trade and makes no
  mean-reversion bet on the ratio itself; the ratio is only a *signal* for an equity-vs-cash
  allocation. Different instruments (equities vs metals), different mechanism (regime timing
  vs ratio reversion).

## Method lineage (the desk's shared engine)

- **HAC / Newey-West *t*-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  used for the *t* on the daily excess-return difference (see
  [`strategy._hac_t_mean`](../gold_oil_ratio/strategy.py)).
- **Circular block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap* (JASA);
  Künsch (1989) — block resampling preserves the autocorrelation i.i.d. resampling destroys;
  used for the CI on the mean daily excess of switch-minus-benchmark.
- **Excess-of-cash Sharpe race & the random-timing control.** The discipline of comparing a
  part-time-in-cash book to buy-and-hold on a matched excess basis, with a same-cash-fraction
  random control, follows the desk's house rules in [`METHODOLOGY.md`](../../../METHODOLOGY.md).

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`), auto-adjusted close: GLD (gold), USO (oil),
  SPY (equity, total-return-adjusted close), ^IRX (13-week T-bill yield, the cash leg). Common
  history 2006-04-10 → 2026-05-29. All headline numbers are pinned with an as-of date and
  content fingerprints (see [`docs/results.md`](results.md)). The offline reproducible core and
  test-suite run on the deterministic [`data.synthetic_daily`](../gold_oil_ratio/data.py)
  generator, never the network.

## Related desk studies

- **[Study 85 — Dr-Copper](../../85-dr-copper/)** — the copper/gold forecasting cousin (Weak
  signal, coincident-only). The closest sibling; read it for the regression angle this study
  deliberately avoids.
- **[Study 113 — Gold-Silver-Ratio](../../113-gold-silver-ratio/)** — the metals pairs-trade
  cousin (Weak / Fragile). The mean-reversion angle this study deliberately avoids.
- **[Study 97 — Balancing-Act](../../97-balancing-act/)** — the 60/40 study, the desk's
  reference for an honest excess-of-cash allocation race.
