# References & literature map — Study 337 (Covered-Call-ETF)

## The claim under test

- **"Yield without giving up returns."** The marketing pitch behind the buy-write ETF boom —
  J.P. Morgan's **JEPI** (JPMorgan Equity Premium Income) and **JEPQ**, Global X's **QYLD**
  (Nasdaq-100 Covered Call), **XYLD** (S&P 500 Covered Call) and **RYLD** (Russell 2000 Covered
  Call). The funds advertise high, steady monthly **distribution yields** (often 8–12%) as
  "income" while holding the underlying equity index, and are sold to retirees and income
  investors as a lower-risk, higher-cash-flow way to own stocks. The testable hypothesis: a
  buy-write delivers comparable total return to its equity benchmark *plus* a distribution
  cushion — i.e. you keep the returns and get the yield on top. We test it against SPY total
  return, with a HAC-robust spread, and we decompose the "income" into NAV vs return of capital.

## Covered calls — the established theory and prior evidence

- **Israelov & Nielsen (2015), *Covered Calls Uncovered* (Financial Analysts Journal).** The
  definitive teardown: a covered call is mechanically long equity + short a call, so its return
  is equity beta minus the calls' negative-convexity drag. The "income" from writing calls is
  *not* free yield — it is compensation for capping the upside, and over time it is closely
  matched by foregone capital gains. This is the source of the "the distribution is gains given
  back" framing.
- **Whaley (2002), *Return and Risk of CBOE Buy Write Monthly Index* (Journal of Derivatives).**
  The BXM index study — early evidence that systematic at-the-money call writing on the S&P 500
  trades return for a modestly higher Sharpe in calm regimes, and gives back the trade in trends.
- **Israelov (2017), *Pathetic Protection: The Elusive Benefits of Protective Puts*** and the
  AQR option-overlay literature — the symmetric point: option overlays mostly relocate risk and
  return rather than create alpha.
- **Volatility risk premium.** Bakshi & Kapadia (2003), *Delta-Hedged Gains and the Negative
  Market Volatility Risk Premium* (Review of Financial Studies) — index options are, on average,
  richly priced, so *selling* them earns a premium. The buy-write harvests this VRP, but the
  empirical question is whether the premium exceeds the upside it forfeits (here: it does not,
  net of the cap, vs SPY).

## The income illusion — distribution yield vs total return

- **Return of capital / NAV erosion.** A distribution that exceeds the fund's economic earnings
  is financed by liquidating principal: the NAV falls while the headline "yield" stays high. The
  decomposition total return = price (NAV) return + distribution yield makes this visible —
  when the price CAGR is negative under a double-digit distribution, the *entire* distribution is
  return of capital. This is the central, distinct object of this study.
- **The dividend/"income" fallacy.** Hartzmark & Solomon (2019), *The Dividend Disconnect*
  (Journal of Finance), and Baker, Nagel & Wurgler (2007) — investors treat distributions as
  separate "income" rather than self-financed sales, the behavioural reason buy-write "income"
  is appealing despite being NAV-neutral at best. Mirrors the desk's [Study 57 — Yield-Trap]
  (../../57-yield-trap/) and [Study 143 — Dividend-Capture](../../143-dividend-capture/).

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) — applied
  to the monthly return spread in [`strategy.race`](../covered_call_etf/strategy.py).
- **Circular block bootstrap.** Politis & Romano (1992) — block resampling preserves the serial
  dependence i.i.d. resampling destroys; used for the spread's 95% CI.
- **Up/down capture.** The standard Morningstar capture-ratio construction, computed on aligned
  monthly returns conditioned on the benchmark's sign.

## Data sources used here

- **Yahoo! Finance monthly bars** (via `yfinance`). The race uses `auto_adjust=True` (total
  return, distributions reinvested, fund fees inside); the income-illusion split uses the
  split-only close + the dividend stream so the price (NAV) leg can be separated from the
  distribution. Tickers: JEPI, JEPQ, QYLD, XYLD, RYLD, SPY, QQQ. As-of **2026-05-31**, partial
  current month dropped, content-fingerprinted (see [`docs/results.md`](results.md)). The offline
  reproducible core and the test-suite run entirely on the deterministic
  [`data.synthetic_buywrite`](../covered_call_etf/data.py) replicator, never the network.

## Related desk studies

- **[Study 62 — Premium-Seller](../../62-premium-seller/)** — the prior covered-call study:
  QYLD raced against *its own underlying* (QQQ), focused on the upside/downside-capture
  asymmetry that explains the QQQ shortfall. **This study is the distinct, complementary
  angle:** the *income illusion* (distribution decomposed into NAV vs return-of-capital) across
  the broader JEPI/JEPQ/XYLD/RYLD generation, raced against **SPY total return**, with a
  mechanical synthetic buy-write *replicator* as the control. Where 62 asks "does it beat the
  index it holds?", 337 asks "is the *income* real, and is the newer generation any different?".
- **[Study 57 — Yield-Trap](../../57-yield-trap/)** — high-dividend stocks on total return; same
  "the yield is not free" lesson on single names.
- **[Study 35 — Contango](../../35-contango/)** and the desk's vol-carry studies — the
  volatility/term-premium-harvesting family the buy-write belongs to.
