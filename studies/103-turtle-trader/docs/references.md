# References & literature map — Study 103 (Turtle-Trader)

## The claim under test

- **The Turtle Trading System.** In 1983–1984, Richard Dennis trained a group of
  novice traders ("Turtles") on a mechanical breakout rule. System 1: buy when the
  daily close breaks above the 20-day high (a new 20-day high), exit when it falls
  below the 10-day low; short on a 20-day low break, exit on a 10-day high. System 2:
  the same with a 55-day entry and 20-day exit. Position sizing via ATR-based units
  (*N*). The recipe was kept secret until 2003, when Curtis Faith published it in full.
  The core claim is that **channel breakouts capture the beginning of sustained trends**
  across a diversified basket of futures markets, producing positive long-run expectancy
  net of costs. We steelman this as: *a Donchian breakout entry into a new N-day high/low
  predicts a further price move in the same direction, measured to the exit channel, that
  is distinguishable from a random-entry baseline.*

## The underlying trend-following effect

- **Momentum — the AQR / Jegadeesh-Titman family.** Jegadeesh & Titman (1993),
  *Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency*
  (Journal of Finance) — the foundational momentum paper; 3-12 month winners continue to
  outperform. The Turtle rule is a trend filter that exploits this at the instrument level.
- **Time-series momentum (TSMOM).** Moskowitz, Ooi & Pedersen (2012), *Time Series Momentum*
  (Journal of Financial Economics) — documents positive 12-month autocorrelation in futures
  returns across asset classes; the Turtle system is a mechanical implementation of this
  idea one decade earlier.
- **Trend-following in managed futures.** Hurst, Ooi & Pedersen (2017), *A Century of
  Evidence on Trend-Following Investing* (Journal of Portfolio Management) — 100-year back-
  test of trend-following across asset classes; broadly consistent positive Sharpe before
  costs, with clear post-2009 weakness.
- **Donchian channels.** Richard Donchian (1960), *Commodities Close-Up* (Futures Magazine)
  — original description of the 4-week (≈20-day) channel breakout rule. The Turtle system
  is the most famous systematic implementation of Donchian's idea.

## The Turtle Trading System (primary sources)

- **Faith, C. (2003).** *The Original Turtle Trading Rules* (self-published; available at
  turtletrader.com) — the first public disclosure of the exact entry/exit parameters by one
  of the original Turtle traders.
- **Faith, C. (2007).** *Way of the Turtle: The Secret Methods that Turned Ordinary People
  into Legendary Traders.* McGraw-Hill. — the full narrative and system description.
- **Covel, M. (2004).** *Trend Following.* Prentice Hall. — the practitioner account of
  the Turtle legacy and why trend-following works (or worked).
- **Dennis, R. (cited).** In various magazine interviews (1987, 1989), Richard Dennis
  attributed the Turtle program's success to the mechanical nature of the system — removing
  emotion from the breakout signal.

## Post-publication decay

- **McLean, R.D. & Pontiff, J. (2016).** *Does Academic Research Destroy Stock Return
  Predictability?* (Journal of Finance) — systematic study showing anomalies earn ~26%
  less after publication, as arbitrage capital flows in. Our pre/post-2003 split shows
  ~40% decay in the Turtle edge (703 → 303 bps/trade) consistent with this pattern.
- **Menkhoff, L., Sarno, L., Schmeling, M. & Schrimpf, A. (2012).** *Currency Momentum
  Strategies* (Journal of Financial Economics) — momentum documented in FX; consistent with
  the Turtle basket including USD proxy (UUP).

## Why shorts in an equity/commodity ETF basket are structurally wrong

- **Equity risk premium.** Dimson, Marsh & Staunton (2002), *Triumph of the Optimists*
  (Princeton University Press) — 100-year global equity premium; the long-term upward drift
  of equity markets makes short entries on 20-day lows structurally disadvantaged vs longs.
- **Commodity ETF roll costs.** Gorton & Rouwenhorst (2006), *Facts and Fantasies about
  Commodity Futures* (Financial Analysts Journal) — commodity futures roll yield explains
  much of the commodity return distribution; ETF proxies (USO, DBA) face negative roll in
  contango, favouring longs during spot-price rallies and making shorts harder to sustain.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.summarize`](../turtle_trader/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Donchian channel implementation.** The rolling-high entry with a shifted lookback
  (preventing look-ahead) mirrors the desk's standard in
  [`strategy.donchian_signals`](../turtle_trader/strategy.py).
- **Reproducibility stamp.** [`quantlab/repro.py`](../../../quantlab/repro.py) — the as-of
  freeze and content fingerprint each headline run carries.

## Data sources

- **Yahoo! Finance daily bars** (via `yfinance`), 1993–2026 for SPY; shorter windows for
  GLD/TLT/USO/UUP/IEF/DBA as ETFs launched later. All bars auto-adjusted for splits and
  dividends. Daily granularity gives long windows for the pre/post split and cost analysis.

## Related desk studies

- **[Study 20 — Freight-Train](../../20-freight-train/)**: the general trend-following
  decomposition — same family as the Turtle system.
- **[Study 31 — Trade-Winds](../../31-trade-winds/)**: trend-following across currencies —
  confirms the directional asymmetry in systematic trend.
- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: the 50/200 day moving-average cross —
  a slower trend filter on the same asset class.
- **[Study 86 — Tail-Radar](../../86-tail-radar/)**: volatility-index signals that share the
  "new high / new low" breakout intuition.
