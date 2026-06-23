# References & literature map — Study 377 (Bid-Ask-Bounce)

## The claim under test

- **Short-horizon mean reversion as "edge."** A perennial retail and quant-lite claim: stocks
  that fall today bounce back tomorrow, so a "buy the losers / sell the winners" book at the
  one-day horizon is reliable alpha. The pattern is genuinely visible — daily returns show
  **negative lag-1 autocorrelation**, especially in small-caps — and short-term contrarian
  strategies backtest beautifully *on close-to-close prices*.
- **The rival explanation (Roll, 1984).** Richard Roll, *A Simple Implicit Measure of the
  Effective Bid-Ask Spread in an Efficient Market*, **Journal of Finance 39(4), 1127–1139
  (1984)**. Roll showed that even when the *true* price is a pure random walk (no predictability
  at all), **transaction** prices bounce between the bid and the ask, injecting a mechanical
  negative serial covariance into observed returns: **cov₁ = −(s/2)²**, where *s* is the
  effective spread. The apparent "mean reversion" is then an artefact of *measuring price at the
  trade*, not a property of the efficient price — and it cannot be harvested, because capturing
  it means crossing the very spread *s* that creates it. Roll inverted the identity into a famous
  **spread estimator**, *s* = 2·√(−cov₁).

## Why the bounce is not free money — the microstructure literature

- **Bid-ask spread components.** Hasbrouck, *Empirical Market Microstructure* (Oxford, 2007);
  Stoll, *The Supply of Dealer Services in Securities Markets*, JF (1978); Glosten & Milgrom,
  *Bid, Ask and Transaction Prices in a Specialist Market with Heterogeneously Informed Traders*,
  JFE (1985) — the spread compensates the liquidity provider (order-processing, inventory,
  adverse-selection costs). A liquidity *taker* who trades against the bounce pays exactly this,
  every round trip, which is why the apparent reversion has negative *net* expected return.
- **Refinements of Roll's estimator.** Hasbrouck (2009), *Trading Costs and Returns for US
  Equities: Estimating Effective Costs from Daily Data* (JF) — a Gibbs-sampler "effective cost"
  built on Roll's identity; Corwin & Schultz (2012), *A Simple Way to Estimate Bid-Ask Spreads
  from Daily High and Low Prices* (JF); Abdi & Ranaldo (2017), *A Simple Estimation of Bid-Ask
  Spreads from Daily Close, High, and Low Prices* (RFS). All inherit Roll's core insight that
  daily-frequency autocovariance encodes the spread.
- **Short-term reversal as a documented anomaly — and its cost sensitivity.** Jegadeesh (1990),
  *Evidence of Predictable Behavior of Security Returns* (JF) and Lehmann (1990), *Fads,
  Martingales, and Market Efficiency* (QJE) — the original one-week/one-month reversal results;
  Avramov, Chordia & Goyal (2006), *Liquidity and Autocorrelations in Individual Stock Returns*
  (JF) — the reversal is **strongest exactly where illiquidity (and the bounce) is largest**, and
  shrinks once you account for it; Nagel (2012), *Evaporating Liquidity*, **Review of Financial
  Studies 25(7)** — short-term reversal returns behave like the compensation a liquidity
  *provider* earns (they spike when liquidity is scarce), i.e. they are the spread, not free alpha.

## Why a gross backtest lies here — inference & execution

- **The gross/net gap is the whole study.** A close-to-close backtest of a daily-flip reversal
  book trades against the bounce but is *credited* at the print, never charged the spread it
  crosses. The desk's house rules (one execution lag; one-way cost × turnover; gross labelled
  gross and net labelled net — [METHODOLOGY.md](../../METHODOLOGY.md)) exist precisely to stop a
  microstructure artefact from being sold as alpha.
- **Block bootstrap for autocorrelated PnL.** Reversal PnL is itself serially dependent, so the
  significance of the mean daily PnL is judged with a **circular/moving-block bootstrap** (Künsch
  1989; Politis & Romano 1994), not an i.i.d. *t* alone — [`strategy.block_bootstrap_p`](../bid_ask_bounce/strategy.py).

## Method lineage (the desk's shared engine)

- **Roll model + planted-edge control.** [`data.synthetic_roll`](../bid_ask_bounce/data.py) builds a
  true AR(1) price (zero reversion at `edge = 0`) plus a fair-coin bid/ask bounce; the offline
  core runs with no network. The control proves the bounce reproduces cov₁ = −(s/2)² *and* that a
  gross-positive reversal book net of the spread makes money **only** when a genuine reversion is
  planted.
- **Roll spread estimator + autocorrelation.** [`strategy.roll_spread`](../bid_ask_bounce/strategy.py)
  inverts the identity *s* = 2·√(−cov₁); [`strategy.lag1_autocorr`](../bid_ask_bounce/strategy.py) is the
  apparent-reversion headline.
- **Reversal book with execution lag + one-way costs.**
  [`strategy.reversal_book`](../bid_ask_bounce/strategy.py) forms a dollar-neutral one-day contrarian
  book, enters one day after the signal, and charges half-spread × turnover;
  [`strategy.summarize_book`](../bid_ask_bounce/strategy.py) reports gross/net Sharpe, win-rate, *t*, and
  the bootstrap *p*.

## Data sources used here

- **yfinance** daily adjusted closes for a fixed 35-name small/mid-cap US basket + IWM,
  2005-08-11 → 2026-06-18, cached under `_cache/smallcap_prices.csv`. yfinance has **no** bid/ask,
  so the effective spread is **inferred** via Roll's estimator and labelled a proxy throughout.
  All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 329 — One-Month-Reversal](../../329-one-month-reversal/)**: the lower-frequency cousin —
  whether the monthly reversal premium survives once it is charged realistic costs.
- **[Study 140 — Amihud-Illiquidity](../../140-amihud-illiquidity/)**: the illiquidity premium whose
  shadow this study lives in — short-term reversal is concentrated in exactly the illiquid names
  where the bounce is largest.
