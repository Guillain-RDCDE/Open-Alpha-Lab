# References & literature map — Study 110 (Faber-Timing)

## The claim under test

- **Faber (2007).** Mebane T. Faber, *A Quantitative Approach to Tactical Asset Allocation*,
  Journal of Wealth Management, vol. 9 no. 4, 2007 (also SSRN 962461). The canonical paper:
  hold the index when its end-of-month price is above the 10-month simple moving average;
  move to cash (T-bills) otherwise. Faber reports substantially improved Sharpe and greatly
  reduced drawdowns vs buy-and-hold on the MSCI World and five global asset classes, 1900–2006.
  This paper has been downloaded over 1 million times — it may be the most-cited tactical
  asset allocation rule ever published.
- **The steelman.** The rule concedes that timing the market is hard; it does not predict
  *when* the market will fall — it simply responds to a confirmed trend. The 10-month SMA
  is a momentum filter: when price is below its own 10-month average, the market has been in
  a sustained downtrend, and the rule acknowledges this and moves to safety. The claim is
  not that it generates alpha, but that it avoids catastrophic drawdowns that destroy
  investors' ability to stay in the market. This framing is honest and testable.

## The statistical and financial literature behind the rule

- **Trend-following / time-series momentum.** Moskowitz, Ooi & Pedersen (2012), *Time Series
  Momentum*, Journal of Financial Economics — documents that being long (short) assets with
  a positive (negative) past 12-month return earns a significant risk premium across 58 liquid
  instruments. The Faber rule is a binary version of this for equities only.
- **Moving-average timing rules.** Brock, Lakonishok & LeBaron (1992), *Simple Technical
  Trading Rules and the Stochastic Properties of Stock Returns*, Journal of Finance — among
  the first systematic tests of MA crossover rules; reported outperformance in pre-1987 data.
  Bessembinder & Chan (1998), *The Profitability of Technical Trading Rules in the Asian
  Stock Markets*, Pacific-Basin Finance Journal — replicated in emerging markets.
- **Post-publication decay and data-snooping.** Han, Zhou & Zhu (2016), *A Trend Factor:
  Any Economic Gains from Using Information over Investment Horizons?*, Journal of Financial
  Economics — finds the MA rule's advantage is concentrated in the first few years of
  publication and decays thereafter. This is our sub-period story: the 2000s (dotcom + GFC)
  are the canonical showcase; the 2010s bull market was punishing for the rule.
- **Comparison with buy-and-hold on a risk-adjusted basis.** Zakamulin (2014), *The
  Real-Life Performance of Market Timing with Moving Average and Time-Series Momentum Rules*,
  Journal of Asset Management — finds that after properly adjusting for time-in-market and
  crediting the cash leg, the Sharpe advantage is smaller than reported in earlier papers,
  and often not statistically significant on post-2000 data alone.

## Why the rule can work — the theoretical mechanism

- **Regime-switching market structure.** Hamilton (1989), *A New Approach to the Economic
  Analysis of Nonstationary Time Series and the Business Cycle*, Econometrica — the canonical
  two-state Markov regime model. Our synthetic generator (`data.synthetic_daily`) is a direct
  implementation: a bull state with positive drift and low vol, a bear state with negative
  drift and high vol. When regimes are distinguishable, a lagged SMA can partially identify
  the current regime and adjust exposure accordingly.
- **Vol-clustering underpins the timing power.** The SMA rule doesn't just respond to trend —
  it inadvertently times volatility. Bear markets (crashes) come with high realized vol, and
  the SMA exits *during* high-vol periods, mechanically reducing portfolio variance more than
  its exit frequency would suggest. This is the "risk reduction" story that our real-tape
  results confirm: drawdown cuts from −55% to −22%, while CAGR only drops from +10.8% to
  +9.2%.
- **The honest caveat — the random-timing control.** The desk's key contribution: we run a
  *random-timing control* that matches the SMA rule's in-market fraction on random days. This
  control's Sharpe (+0.304) is far *lower* than the SMA rule (+0.729), confirming that the
  SMA's advantage comes from *which* days it avoids (specifically, the large drawdown periods),
  not merely from having less exposure. See [strategy.py](../faber_timing/strategy.py).

## Related desk studies

- **[Study 16 — Storm-Shy](../../16-storm-shy/)**: the Moreira–Muir (2017) *vol-targeting*
  overlay — a continuous version of the same idea (scale position by inverse realized vol
  rather than a binary in/out). Storm-Shy shows that vol-targeting also improves Sharpe
  over buy-and-hold, but via a different mechanism (always partially invested, not binary).
  The comparison is illuminating: both exploit vol clustering; the SMA rule is simpler and
  more binary; vol-targeting is smoother but requires leverage in calm periods.
- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: the 50/200 daily SMA *crossover* entry
  signal (trend-following timing), tested as a pure long signal. Different from Faber's
  binary in/out — more like a "golden cross" entry filter. Same MA family, different application.
- **[Study 68 — All-Weather](../../68-all-weather/)**: the Ray Dalio Risk-Parity allocation
  — a *structural* diversification approach vs Faber's *tactical* timing. Both aim to reduce
  drawdown; they do it differently. Risk parity is always diversified; Faber is binary and concentrated.
- **[Study 72 — Loaded-Dice](../../72-loaded-dice/)**: the SMA(5/10) 5-minute scalp — the
  *opposite* frequency of the Faber rule. At 5-minute fidelity, MA crossovers add nothing.
  At 10-month fidelity, they add drawdown protection. The contrast in outcomes across
  time-horizons is one of this desk's clearest illustrations that the bar for "does it work?"
  is frequency-dependent.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica —
  [`strategy.summary`](../faber_timing/strategy.py) and [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Return-difference t-stat (Sharpe comparison).** Jobson & Korkie (1981), *Performance
  Hypothesis Testing with the Sharpe and Treynor Measures*, Journal of Finance —
  implemented via [`strategy.sharpe_diff_tstat`](../faber_timing/strategy.py).
- **Block bootstrap CI.** Politis & Romano (1994), *The Stationary Bootstrap*, JASA —
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).

## Data sources

- **SPY daily total-return closes** (via `yfinance`, `auto_adjust=True`), 1993-01-29 to
  2026-06-12. The S&P 500 ETF is the canonical Faber instrument for post-1993 tests; pre-1993
  Faber used the Cowles Commission / CRSP index series. Split and dividend adjustments are
  essential for multi-decade buy-and-hold comparisons — an unadjusted series would
  understate buy-and-hold total return by several percentage points per year in the 1990s.
- **Cash rate proxy:** a flat 4%/yr annual rate (equivalent to ~SOFR/Fed funds in the long
  run). FRED and CBOE endpoints are unavailable in this sandbox; the flat rate is
  slightly conservative vs the actual Fed funds path (which ranged from 0% to 5.5% over
  the sample). Sensitivity: at 0% cash yield the Sharpe advantage narrows (timing lags
  more in bull markets); at 5% cash yield the CAGR gap narrows further.
