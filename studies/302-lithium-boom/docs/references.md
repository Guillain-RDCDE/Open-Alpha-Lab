# References & literature map — Study 302 (Lithium-Boom)

## The claim under test

- **The battery-metals super-cycle pitch.** The thesis, repeated across sell-side notes and
  thematic-ETF marketing through the late 2010s and the 2021–22 mania: the electrification of
  transport would drive a structural, multi-year bull market in lithium and battery materials,
  and a trend-follower riding the **Global X Lithium & Battery Tech ETF (LIT)** would have
  caught the boom and — by following the trend — stepped aside before the bust. Global X,
  *LIT — Global X Lithium & Battery Tech ETF* fund literature; and the broad "lithium is the
  new oil" commodity-super-cycle commentary (e.g. Goldman Sachs and Benchmark Mineral
  Intelligence cycle notes, 2021–2022). We take that literally: hold LIT (and the miners ALB,
  SQM) when price is above its 200-day moving average, else sit in cash, and ask whether the
  *timing* paid — and whether it beat simply owning the thing, or the S&P 500.

## The real effect the rule leans on — time-series (trend) momentum

- **Time-series momentum.** Moskowitz, Ooi & Pedersen (2012), *Time Series Momentum* (Journal
  of Financial Economics) — an asset's own past 12-month return predicts its next-month return
  across 58 instruments; the canonical academic foundation for trend-following. The 200-day
  moving-average rule is a coarse, widely-used proxy for the same signal.
- **The moving-average timing rule.** Faber (2007), *A Quantitative Approach to Tactical Asset
  Allocation* (Journal of Wealth Management) — the 10-month / 200-day SMA long-flat rule that
  this study implements directly; Faber's headline result is *risk reduction* (drawdown and
  volatility), not excess return, which is exactly the distinction this study turns on. The
  desk has already torn this rule down on broad indices in
  [Study 116 — Faber-Timing](../../116-faber-timing/).
- **Trend-following on commodities.** Hurst, Ooi & Pedersen (2017), *A Century of Evidence on
  Trend-Following Investing* (Journal of Portfolio Management) — trend works across asset
  classes including commodities, but the Sharpe of a single-instrument sleeve is thin and
  diversification across many trends is what makes it investable.

## Why the headline (boom-riding return) is the wrong frame

- **Return vs risk-reduction.** A long/flat moving-average overlay mechanically cuts drawdowns
  — it is out of the market during sustained declines by construction — *whether or not* the
  underlying has any exploitable persistence. Zakamulin (2014), *The Real-Life Performance of
  Market Timing with Moving Average and Time-Series Momentum Rules* (Journal of Asset
  Management) — moving-average timing's apparent edge is largely a volatility/exposure effect,
  not alpha, and is fragile to the choice of window and to costs. This study's synthetic
  control makes the point starkly: at zero planted persistence the overlay still halves the
  drawdown while adding zero Sharpe.
- **Excess-of-cash, not raw, races.** Because the overlay is in cash part-time, comparing its
  raw Sharpe to a fully-invested buy-and-hold's *raw* Sharpe is the classic apples-to-oranges
  error the desk guards against (METHODOLOGY → House rules); we race excess-of-cash to
  excess-of-cash throughout.
- **Single-theme concentration.** A thematic ETF concentrates idiosyncratic and
  commodity-price risk; the relevant benchmark for "did it pay?" is not just buy-and-hold the
  theme but the **opportunity cost** of simply owning the broad index (SPY).

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.hac_tstat`](../lithium_boom/strategy.py).
- **Circular block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap* (JASA), and
  the block-bootstrap CI for the Sharpe ratio — preserves volatility clustering that i.i.d.
  resampling destroys ([`strategy.block_bootstrap_sharpe_ci`](../lithium_boom/strategy.py)).
- **Sharpe-ratio inference.** Lo (2002), *The Statistics of Sharpe Ratios* (Financial Analysts
  Journal) — why a Sharpe needs a standard error before it is compared.

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`), `auto_adjust=True` total-return. Basket:
  LIT (from 2010-07-23, the fund's inception), ALB, SQM, and SPY as benchmark; all from
  2010 to the as-of. All headline numbers are pinned with an as-of date and content
  fingerprint (see [`docs/results.md`](results.md)). The offline reproducible core and
  test-suite run on the deterministic [`data.synthetic_theme`](../lithium_boom/data.py)
  generator, never the network.

## Related desk studies

- **[Study 116 — Faber-Timing](../../116-faber-timing/)**: the same 200-day SMA long-flat rule
  on broad indices — the parent method. This study is its single-theme commodity application.
- **[Study 20 — Freight-Train](../../20-freight-train/)** and
  **[Study 31 — Trade-Winds](../../31-trade-winds/)**: time-series momentum across *many*
  markets, where diversification is what (barely) rescues a thin per-sleeve Sharpe. Lithium-Boom
  is the one-sleeve, one-theme case — and shows exactly why a single trend sleeve is fragile.
- **[Study 208 — Gold-Miners](../../208-gold-miners/)**: a thematic-commodity-equity teardown in
  the same spirit (are miners just leveraged metal?) — Weak/Mirage.
- **[Study 35 — Contango](../../35-contango/)**: commodity carry/roll, the other half of the
  "carry, curves & commodities" bench family.
